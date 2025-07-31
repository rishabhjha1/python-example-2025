#!/usr/bin/env python

# Enhanced Chagas disease detection for PhysioNet Challenge 2025
# Implements ResNet-based ensemble with careful handling of class imbalance

import os
import numpy as np
import pandas as pd
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

from helper_code import *

# Optimized constants
TARGET_SAMPLING_RATE = 400  # Standard rate across datasets
SIGNAL_LENGTH = 4096  # Standard length (zero-padded)
NUM_LEADS = 12
BATCH_SIZE = 64
NUM_EPOCHS = 30
NUM_ENSEMBLE_MODELS = 5  # Ensemble size
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ResNet block for 1D signals
class ResBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1):
        super(ResBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride),
                nn.BatchNorm1d(out_channels)
            )
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.dropout(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# Enhanced ResNet for ECG
class ECGResNet(nn.Module):
    def __init__(self, num_leads=12, num_classes=1):
        super(ECGResNet, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv1d(num_leads, 64, kernel_size=15, stride=2, padding=7)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(3, stride=2, padding=1)
        
        # ResNet blocks with increasing channels
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Global pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.5)
        
        # Multi-scale feature aggregation
        self.fc1 = nn.Linear(512 + 64 + 2, 256)  # 512 from ResNet + 64 from attention + 2 demographics
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, num_classes)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(512, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(ResBlock1D(in_channels, out_channels, stride=stride))
        for _ in range(1, num_blocks):
            layers.append(ResBlock1D(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x, demographics):
        # Initial feature extraction
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.pool1(out)
        
        # ResNet blocks
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        # Global pooling
        out_pooled = self.global_pool(out).squeeze(-1)  # Shape: (batch, 512)
        
        # Attention mechanism - fixed dimension handling
        attention_scores = self.attention(out_pooled)  # Shape: (batch, 1)
        attention_weights = F.softmax(attention_scores, dim=0)  # Softmax over batch
        
        # Apply attention as a scalar weight per sample
        attended_features = out_pooled * attention_weights.expand_as(out_pooled)
        
        # Extract additional features from attended representation
        attended_summary = torch.sum(attended_features, dim=1, keepdim=True).expand(-1, 64)  # Shape: (batch, 64)
        
        # Combine features
        combined = torch.cat([out_pooled, attended_summary, demographics], dim=1)  # Shape: (batch, 512+64+2)
        
        # Classification
        out = F.relu(self.fc1(combined))
        out = self.dropout(out)
        out = F.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.fc3(out)
        
        return torch.sigmoid(out)

# Custom dataset for ECG data
class ECGDataset(Dataset):
    def __init__(self, signals, demographics, labels, augment=True):
        self.signals = torch.FloatTensor(signals)
        self.demographics = torch.FloatTensor(demographics)
        self.labels = torch.FloatTensor(labels).unsqueeze(1)
        self.augment = augment
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        signal = self.signals[idx]
        demo = self.demographics[idx]
        label = self.labels[idx]
        
        # Data augmentation during training
        if self.augment and torch.rand(1).item() > 0.5:
            # Add Gaussian noise
            noise = torch.randn_like(signal) * 0.05
            signal = signal + noise
            
            # Random amplitude scaling
            scale = 0.9 + torch.rand(1).item() * 0.2
            signal = signal * scale
        
        return signal, demo, label

# Focal loss for class imbalance
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()

def train_model(data_folder, model_folder, verbose):
    """
    Train ensemble of ResNet models optimized for top 5% ranking
    """
    if verbose:
        print("Training enhanced Chagas detection ensemble...")
        print(f"Using device: {DEVICE}")
    
    os.makedirs(model_folder, exist_ok=True)
    
    # Load data with robust handling
    signals, labels, demographics = load_data_comprehensive(data_folder, verbose)
    
    if len(signals) < 100:
        if verbose:
            print(f"Warning: Only {len(signals)} samples found. Creating baseline model.")
        return create_baseline_model(model_folder, verbose)
    
    # Train ensemble
    train_ensemble(signals, labels, demographics, model_folder, verbose)

def load_data_comprehensive(data_folder, verbose):
    """
    Load data from multiple sources with careful label handling
    """
    all_signals = []
    all_labels = []
    all_demographics = []
    
    # Try different data sources
    sources = [
        ('HDF5', load_from_hdf5_enhanced),
        ('WFDB', load_from_wfdb_enhanced),
        ('Direct', load_from_directory)
    ]
    
    for source_name, loader_func in sources:
        try:
            if verbose:
                print(f"Attempting to load from {source_name}...")
            signals, labels, demos = loader_func(data_folder, verbose)
            
            if len(signals) > 0:
                all_signals.extend(signals)
                all_labels.extend(labels)
                all_demographics.extend(demos)
                
                if verbose:
                    print(f"Loaded {len(signals)} samples from {source_name}")
        except Exception as e:
            if verbose:
                print(f"Failed to load from {source_name}: {str(e)}")
    
    if verbose:
        print(f"\nTotal samples loaded: {len(all_signals)}")
        if len(all_labels) > 0:
            pos_rate = np.mean(all_labels) * 100
            print(f"Positive rate: {pos_rate:.2f}%")
    
    return all_signals, all_labels, all_demographics

def load_from_hdf5_enhanced(data_folder, verbose):
    """
    Enhanced HDF5 loading with better label detection
    """
    signals, labels, demographics = [], [], []
    
    hdf5_path = os.path.join(data_folder, 'exams.hdf5')
    if not os.path.exists(hdf5_path):
        return signals, labels, demographics
    
    try:
        # Load metadata
        exams_csv_path = os.path.join(data_folder, 'exams.csv')
        if not os.path.exists(exams_csv_path):
            if verbose:
                print("exams.csv not found")
            return signals, labels, demographics
            
        exams_df = pd.read_csv(exams_csv_path, nrows=50000)
        
        # Load Chagas labels from multiple possible sources
        chagas_labels = load_chagas_labels(data_folder, verbose)
        
        with h5py.File(hdf5_path, 'r') as hdf:
            # Find the dataset
            dataset_key = None
            for key in ['tracings', 'exams', 'signals', 'ecg']:
                if key in hdf:
                    dataset_key = key
                    break
            
            if dataset_key is None and len(hdf.keys()) > 0:
                dataset_key = list(hdf.keys())[0]
            
            if dataset_key is None:
                return signals, labels, demographics
            
            dataset = hdf[dataset_key]
            
            # Track label distribution
            pos_count = 0
            neg_count = 0
            
            for idx, row in exams_df.iterrows():
                if len(signals) >= 10000:  # Limit for memory
                    break
                
                try:
                    exam_id = row.get('exam_id', row.get('id', idx))
                    
                    # Get label with smart defaults
                    label = get_label_smart(exam_id, row, chagas_labels)
                    if label is None:
                        continue
                    
                    # Skip if we have too many of one class (balance during loading)
                    if label == 1 and pos_count > neg_count + 1000:
                        continue
                    elif label == 0 and neg_count > pos_count + 1000:
                        continue
                    
                    # Get signal
                    signal = extract_signal_safe(dataset, idx, exam_id)
                    if signal is None:
                        continue
                    
                    # Process signal
                    processed_signal = process_signal_enhanced(signal)
                    if processed_signal is None:
                        continue
                    
                    # Extract demographics
                    demo = extract_demographics_enhanced(row)
                    
                    signals.append(processed_signal)
                    labels.append(label)
                    demographics.append(demo)
                    
                    if label == 1:
                        pos_count += 1
                    else:
                        neg_count += 1
                    
                except Exception as e:
                    continue
            
            if verbose:
                print(f"HDF5: Loaded {pos_count} positive and {neg_count} negative samples")
    
    except Exception as e:
        if verbose:
            print(f"HDF5 loading error: {e}")
    
    return signals, labels, demographics

def load_from_wfdb_enhanced(data_folder, verbose):
    """
    Enhanced WFDB loading
    """
    signals, labels, demographics = [], [], []
    
    try:
        records = find_records(data_folder)
        
        for record_name in records[:50000]:
            try:
                record_path = os.path.join(data_folder, record_name)
                
                # Load signal
                signal, fields = load_signals(record_path)
                processed_signal = process_signal_enhanced(signal)
                if processed_signal is None:
                    continue
                
                # Load label
                label = load_label(record_path)
                if label is None:
                    # Try to infer from filename or metadata
                    label = infer_label_from_record(record_name, record_path)
                    if label is None:
                        continue
                
                # Extract demographics
                header = load_header(record_path)
                demo = extract_demographics_from_header(header)
                
                signals.append(processed_signal)
                labels.append(int(label))
                demographics.append(demo)
                
            except Exception:
                continue
    
    except Exception as e:
        if verbose:
            print(f"WFDB loading error: {e}")
    
    return signals, labels, demographics

def load_from_directory(data_folder, verbose):
    """
    Load from directory structure (backup method)
    """
    signals, labels, demographics = [], [], []
    
    # Check for subdirectories by dataset
    dataset_dirs = ['samitrop', 'ptbxl', 'code-15', 'code15']
    
    for dataset_name in dataset_dirs:
        dataset_path = os.path.join(data_folder, dataset_name)
        if os.path.exists(dataset_path):
            # Load based on known dataset characteristics
            if 'samitrop' in dataset_name:
                default_label = 1  # All positive
            elif 'ptb' in dataset_name:
                default_label = 0  # All negative
            else:
                default_label = None  # Need to check
            
            s, l, d = load_dataset_specific(dataset_path, default_label, verbose)
            signals.extend(s)
            labels.extend(l)
            demographics.extend(d)
    
    return signals, labels, demographics

def process_signal_enhanced(signal):
    """
    Enhanced signal processing with better normalization
    """
    try:
        signal = np.array(signal, dtype=np.float32)
        
        # Handle shape
        if len(signal.shape) == 1:
            signal = signal.reshape(-1, 1)
        elif signal.shape[0] <= 12 and signal.shape[1] > 12:
            signal = signal.T
        
        # Ensure 12 leads
        if signal.shape[1] > 12:
            signal = signal[:, :12]
        elif signal.shape[1] < 12:
            # Pad with zeros (missing leads)
            padding = np.zeros((signal.shape[0], 12 - signal.shape[1]))
            signal = np.hstack([signal, padding])
        
        # Resample to target length
        signal = resample_signal_advanced(signal, SIGNAL_LENGTH)
        
        # Advanced preprocessing
        signal = remove_baseline_wander(signal)
        signal = normalize_signal_advanced(signal)
        
        # Transpose for PyTorch (channels first)
        signal = signal.T  # Shape: (12, 4096)
        
        return signal.astype(np.float32)
    
    except Exception:
        return None

def remove_baseline_wander(signal):
    """
    Remove baseline wander using median filter
    """
    from scipy.signal import medfilt
    
    window_size = int(0.2 * 400)  # 200ms window at 400Hz
    if window_size % 2 == 0:
        window_size += 1
    
    for i in range(signal.shape[1]):
        baseline = medfilt(signal[:, i], window_size)
        signal[:, i] = signal[:, i] - baseline
    
    return signal

def normalize_signal_advanced(signal):
    """
    Advanced normalization preserving clinical features
    """
    for i in range(signal.shape[1]):
        # Remove DC offset
        signal[:, i] = signal[:, i] - np.mean(signal[:, i])
        
        # Robust scaling using percentiles
        p5, p95 = np.percentile(signal[:, i], [5, 95])
        scale = p95 - p5
        
        if scale > 1e-6:
            signal[:, i] = (signal[:, i] - p5) / scale - 0.5
        
        # Clip outliers
        signal[:, i] = np.clip(signal[:, i], -3, 3)
    
    return signal

def train_ensemble(signals, labels, demographics, model_folder, verbose):
    """
    Train ensemble of models with different initializations
    """
    # Convert to numpy arrays
    X_signal = np.array(signals, dtype=np.float32)
    X_demo = np.array(demographics, dtype=np.float32)
    y = np.array(labels, dtype=np.float32)
    
    # Handle extreme imbalance
    pos_rate = np.mean(y)
    if pos_rate < 0.01 or pos_rate > 0.99:
        if verbose:
            print(f"Extreme class imbalance detected: {pos_rate:.2%} positive")
        X_signal, X_demo, y = balance_dataset(X_signal, X_demo, y, verbose)
    
    # Split data
    X_sig_train, X_sig_val, X_demo_train, X_demo_val, y_train, y_val = train_test_split(
        X_signal, X_demo, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale demographics
    demo_scaler = StandardScaler()
    X_demo_train = demo_scaler.fit_transform(X_demo_train)
    X_demo_val = demo_scaler.transform(X_demo_val)
    
    # Create datasets
    train_dataset = ECGDataset(X_sig_train, X_demo_train, y_train, augment=True)
    val_dataset = ECGDataset(X_sig_val, X_demo_val, y_val, augment=False)
    
    # Weighted sampling for class balance
    train_labels = torch.FloatTensor(y_train)
    class_weights = compute_class_weights(train_labels)
    sample_weights = torch.tensor([class_weights[int(label)] for label in y_train])
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Train ensemble
    ensemble_models = []
    
    for model_idx in range(NUM_ENSEMBLE_MODELS):
        if verbose:
            print(f"\nTraining model {model_idx + 1}/{NUM_ENSEMBLE_MODELS}")
        
        # Initialize model with different seed
        torch.manual_seed(42 + model_idx)
        model = ECGResNet().to(DEVICE)
        
        # Loss and optimizer
        criterion = FocalLoss(alpha=0.25, gamma=2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
        
        # Train model
        best_val_score = 0
        patience_counter = 0
        
        for epoch in range(NUM_EPOCHS):
            # Training
            model.train()
            train_loss = 0
            
            for signals_batch, demo_batch, labels_batch in train_loader:
                signals_batch = signals_batch.to(DEVICE)
                demo_batch = demo_batch.to(DEVICE)
                labels_batch = labels_batch.to(DEVICE)
                
                optimizer.zero_grad()
                outputs = model(signals_batch, demo_batch)
                loss = criterion(outputs, labels_batch)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_probs = []
            val_labels = []
            
            with torch.no_grad():
                for signals_batch, demo_batch, labels_batch in val_loader:
                    signals_batch = signals_batch.to(DEVICE)
                    demo_batch = demo_batch.to(DEVICE)
                    
                    outputs = model(signals_batch, demo_batch)
                    val_probs.extend(outputs.cpu().numpy())
                    val_labels.extend(labels_batch.numpy())
            
            # Calculate top 5% metric
            val_score = calculate_top_k_metric(np.array(val_probs), np.array(val_labels), k=0.05)
            
            if verbose and epoch % 5 == 0:
                print(f"Epoch {epoch}: Loss={train_loss/len(train_loader):.4f}, Top5%={val_score:.4f}")
            
            # Early stopping
            if val_score > best_val_score:
                best_val_score = val_score
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), os.path.join(model_folder, f'model_{model_idx}.pth'))
            else:
                patience_counter += 1
                if patience_counter > 5:
                    break
            
            scheduler.step()
        
        # Load best model
        model.load_state_dict(torch.load(os.path.join(model_folder, f'model_{model_idx}.pth')))
        ensemble_models.append(model)
    
    # Save ensemble configuration
    save_ensemble_config(model_folder, demo_scaler, verbose)

def calculate_top_k_metric(probs, labels, k=0.05):
    """
    Calculate the fraction of positive cases in top k% of predictions
    """
    n_samples = len(probs)
    n_top = int(np.ceil(n_samples * k))
    
    # Sort by probability (descending)
    sorted_indices = np.argsort(-probs.squeeze())
    top_indices = sorted_indices[:n_top]
    
    # Calculate metric
    n_positive_total = np.sum(labels)
    n_positive_top = np.sum(labels[top_indices])
    
    if n_positive_total == 0:
        return 0.0
    
    return n_positive_top / n_positive_total

def compute_class_weights(labels):
    """
    Compute balanced class weights
    """
    n_samples = len(labels)
    n_classes = 2
    
    class_counts = torch.bincount(labels.long())
    weights = n_samples / (n_classes * class_counts.float())
    
    return weights

def save_ensemble_config(model_folder, demo_scaler, verbose):
    """
    Save configuration and preprocessing objects
    """
    import pickle
    import json
    
    # Save scaler
    with open(os.path.join(model_folder, 'demo_scaler.pkl'), 'wb') as f:
        pickle.dump(demo_scaler, f)
    
    # Save config
    config = {
        'signal_length': SIGNAL_LENGTH,
        'num_leads': NUM_LEADS,
        'num_ensemble_models': NUM_ENSEMBLE_MODELS,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    with open(os.path.join(model_folder, 'config.json'), 'w') as f:
        json.dump(config, f)
    
    if verbose:
        print(f"Ensemble saved to {model_folder}")

def load_model(model_folder, verbose=False):
    """
    Load the trained ensemble
    """
    import pickle
    import json
    
    if verbose:
        print(f"Loading ensemble from {model_folder}")
    
    # Load config
    with open(os.path.join(model_folder, 'config.json'), 'r') as f:
        config = json.load(f)
    
    # Load scaler
    with open(os.path.join(model_folder, 'demo_scaler.pkl'), 'rb') as f:
        demo_scaler = pickle.load(f)
    
    # Load ensemble models
    ensemble_models = []
    for i in range(config['num_ensemble_models']):
        model = ECGResNet().to(DEVICE)
        model.load_state_dict(torch.load(os.path.join(model_folder, f'model_{i}.pth'), 
                                       map_location=DEVICE))
        model.eval()
        ensemble_models.append(model)
    
    return {
        'models': ensemble_models,
        'demo_scaler': demo_scaler,
        'config': config
    }

def run_model(record, model_data, verbose=False):
    """
    Run ensemble prediction on a single record
    """
    try:
        models = model_data['models']
        demo_scaler = model_data['demo_scaler']
        config = model_data['config']
        
        # Load and process signal
        try:
            signal, fields = load_signals(record)
            processed_signal = process_signal_enhanced(signal)
            
            if processed_signal is None:
                raise ValueError("Signal processing failed")
        except Exception as e:
            if verbose:
                print(f"Signal loading failed: {e}")
            # Create default signal
            processed_signal = np.zeros((NUM_LEADS, SIGNAL_LENGTH), dtype=np.float32)
        
        # Extract demographics
        try:
            header = load_header(record)
            demographics = extract_demographics_from_header(header)
        except:
            demographics = np.array([0.5, 0.5])  # Default: middle age, unknown sex
        
        # Prepare inputs
        signal_tensor = torch.FloatTensor(processed_signal).unsqueeze(0).to(DEVICE)
        demo_scaled = demo_scaler.transform(demographics.reshape(1, -1))
        demo_tensor = torch.FloatTensor(demo_scaled).to(DEVICE)
        
        # Ensemble prediction
        predictions = []
        
        with torch.no_grad():
            for model in models:
                prob = model(signal_tensor, demo_tensor).cpu().numpy()[0, 0]
                predictions.append(prob)
        
        # Average predictions
        ensemble_prob = np.mean(predictions)
        
        # Binary prediction (can be optimized based on validation)
        threshold = 0.5
        binary_pred = 1 if ensemble_prob >= threshold else 0
        
        return binary_pred, float(ensemble_prob)
        
    except Exception as e:
        if verbose:
            print(f"Error in run_model: {e}")
        # Conservative prediction for errors
        return 0, 0.1

# Additional helper functions

def load_chagas_labels(data_folder, verbose):
    """
    Load Chagas labels from various possible files
    """
    chagas_labels = {}
    
    label_files = [
        'chagas_labels.csv',
        'samitrop_chagas_labels.csv', 
        'code15_chagas_labels.csv',
        'labels.csv'
    ]
    
    for label_file in label_files:
        label_path = os.path.join(data_folder, label_file)
        if os.path.exists(label_path):
            try:
                df = pd.read_csv(label_path)
                for _, row in df.iterrows():
                    exam_id = row.get('exam_id', row.get('id', None))
                    label = row.get('chagas', row.get('label', row.get('target', None)))
                    
                    if exam_id is not None and label is not None:
                        if isinstance(label, str):
                            label_binary = 1 if label.lower() in ['true', 'positive', 'yes', '1', 't', 'y'] else 0
                        else:
                            label_binary = int(float(label))
                        chagas_labels[exam_id] = label_binary
                
                if verbose:
                    print(f"Loaded {len(chagas_labels)} labels from {label_file}")
                break
            except Exception as e:
                if verbose:
                    print(f"Error loading {label_file}: {e}")
    
    return chagas_labels

def get_label_smart(exam_id, row, chagas_labels):
    """
    Get label with smart defaults based on data source
    """
    # Check explicit labels first
    if exam_id in chagas_labels:
        return chagas_labels[exam_id]
    
    # Check row data
    for col in ['chagas', 'label', 'target']:
        if col in row and pd.notna(row[col]):
            val = row[col]
            if isinstance(val, str):
                return 1 if val.lower() in ['true', 'positive', 'yes', '1', 't', 'y'] else 0
            else:
                return int(float(val))
    
    # Use source-based defaults
    source = str(row.get('source', '')).lower()
    if 'samitrop' in source or 'sami' in source:
        return 1  # SaMi-Trop is all positive
    elif 'ptb' in source:
        return 0  # PTB-XL is all negative
    
    return None  # Unknown

def extract_signal_safe(dataset, idx, exam_id):
    """
    Safely extract signal from HDF5 dataset
    """
    try:
        if hasattr(dataset, 'shape') and len(dataset.shape) >= 2:
            return dataset[idx]
        elif str(exam_id) in dataset:
            return dataset[str(exam_id)][:]
        elif str(idx) in dataset:
            return dataset[str(idx)][:]
        else:
            return None
    except:
        return None

def resample_signal_advanced(signal, target_length):
    """
    Advanced resampling using scipy
    """
    from scipy import signal as scipy_signal
    
    current_length = signal.shape[0]
    if current_length == target_length:
        return signal
    
    # Use scipy's resample for better quality
    resampled = np.zeros((target_length, signal.shape[1]))
    for i in range(signal.shape[1]):
        resampled[:, i] = scipy_signal.resample(signal[:, i], target_length)
    
    return resampled

def extract_demographics_enhanced(row):
    """
    Extract demographics with better handling
    """
    # Age
    age = row.get('age', row.get('Age', 50))
    if pd.isna(age) or age < 0 or age > 120:
        age = 50  # Default
    age_norm = age / 100.0
    
    # Sex
    sex = row.get('sex', row.get('Sex', row.get('is_male', 0.5)))
    if pd.isna(sex):
        sex_val = 0.5  # Unknown
    elif isinstance(sex, str):
        sex_lower = sex.lower()
        if sex_lower in ['m', 'male', 'man', '1', 'h', 'homem']:
            sex_val = 1.0
        elif sex_lower in ['f', 'female', 'woman', '0', 'm', 'mulher']:
            sex_val = 0.0
        else:
            sex_val = 0.5
    else:
        sex_val = float(sex)
    
    return np.array([age_norm, sex_val], dtype=np.float32)

def extract_demographics_from_header(header):
    """
    Extract demographics from WFDB header
    """
    age = get_age(header)
    sex = get_sex(header)
    
    age_norm = 0.5  # Default
    if age is not None and 0 <= age <= 120:
        age_norm = age / 100.0
    
    sex_val = 0.5  # Default unknown
    if sex is not None:
        if isinstance(sex, str):
            if sex.lower() in ['m', 'male']:
                sex_val = 1.0
            elif sex.lower() in ['f', 'female']:
                sex_val = 0.0
    
    return np.array([age_norm, sex_val], dtype=np.float32)

def infer_label_from_record(record_name, record_path):
    """
    Try to infer label from record name or metadata
    """
    record_lower = record_name.lower()
    
    # Check if it's from a known dataset
    if 'samitrop' in record_lower or 'sami' in record_lower:
        return 1
    elif 'ptb' in record_lower:
        return 0
    
    # Try to load from accompanying label file
    label_file = record_path + '.labels'
    if os.path.exists(label_file):
        try:
            with open(label_file, 'r') as f:
                label = int(float(f.read().strip()))
                return label
        except:
            pass
    
    return None

def balance_dataset(X_signal, X_demo, y, verbose):
    """
    Balance extremely imbalanced dataset
    """
    pos_indices = np.where(y == 1)[0]
    neg_indices = np.where(y == 0)[0]
    
    n_pos = len(pos_indices)
    n_neg = len(neg_indices)
    
    if n_pos == 0:
        # No positive samples - create synthetic ones
        if verbose:
            print("No positive samples found. Creating synthetic positive samples.")
        
        # Take 10% of negative samples and modify them
        n_synthetic = max(10, int(0.1 * n_neg))
        synthetic_indices = np.random.choice(neg_indices, n_synthetic, replace=False)
        
        # Create synthetic positives by adding characteristic patterns
        synthetic_signals = X_signal[synthetic_indices].copy()
        synthetic_signals = add_chagas_patterns(synthetic_signals)
        
        # Combine datasets
        X_signal = np.vstack([X_signal, synthetic_signals])
        X_demo = np.vstack([X_demo, X_demo[synthetic_indices]])
        y = np.hstack([y, np.ones(n_synthetic)])
        
    elif n_neg == 0:
        # No negative samples - create synthetic ones
        if verbose:
            print("No negative samples found. Creating synthetic negative samples.")
        
        n_synthetic = max(10, int(0.1 * n_pos))
        synthetic_indices = np.random.choice(pos_indices, n_synthetic, replace=False)
        
        # Create synthetic negatives by removing patterns
        synthetic_signals = X_signal[synthetic_indices].copy()
        synthetic_signals = remove_chagas_patterns(synthetic_signals)
        
        X_signal = np.vstack([X_signal, synthetic_signals])
        X_demo = np.vstack([X_demo, X_demo[synthetic_indices]])
        y = np.hstack([y, np.zeros(n_synthetic)])
    
    else:
        # Both classes present - balance by oversampling minority
        if n_pos < n_neg * 0.1:  # Less than 10% positive
            # Oversample positive class
            n_oversample = min(n_neg // 2, n_pos * 5) - n_pos
            if n_oversample > 0:
                oversample_indices = np.random.choice(pos_indices, n_oversample, replace=True)
                X_signal = np.vstack([X_signal, X_signal[oversample_indices]])
                X_demo = np.vstack([X_demo, X_demo[oversample_indices]])
                y = np.hstack([y, y[oversample_indices]])
                
                if verbose:
                    print(f"Oversampled {n_oversample} positive samples")
    
    return X_signal, X_demo, y

def add_chagas_patterns(signals):
    """
    Add synthetic Chagas patterns to signals
    """
    # Chagas often shows:
    # - QRS complex abnormalities
    # - T wave inversions
    # - Conduction delays
    
    modified = signals.copy()
    
    for i in range(modified.shape[0]):
        # Add QRS widening
        qrs_start = int(0.4 * SIGNAL_LENGTH)
        qrs_end = int(0.45 * SIGNAL_LENGTH)
        modified[i, :, qrs_start:qrs_end] *= 1.2
        
        # Add T wave inversion in some leads
        t_wave_start = int(0.5 * SIGNAL_LENGTH)
        t_wave_end = int(0.6 * SIGNAL_LENGTH)
        modified[i, [0, 1, 2], t_wave_start:t_wave_end] *= -0.8
        
        # Add some noise
        modified[i] += np.random.normal(0, 0.05, modified[i].shape)
    
    return modified

def remove_chagas_patterns(signals):
    """
    Remove patterns to create synthetic negatives
    """
    modified = signals.copy()
    
    # Smooth out abnormalities
    from scipy.ndimage import gaussian_filter1d
    
    for i in range(modified.shape[0]):
        for lead in range(modified.shape[1]):
            # Apply smoothing to remove sharp abnormalities
            modified[i, lead] = gaussian_filter1d(modified[i, lead], sigma=3)
            
            # Normalize QRS complexes
            qrs_regions = find_qrs_regions(modified[i, lead])
            for start, end in qrs_regions:
                if end - start > 0:
                    # Normalize amplitude
                    modified[i, lead, start:end] *= 0.8
        
        # Add slight noise to make it more realistic
        modified[i] += np.random.normal(0, 0.02, modified[i].shape)
    
    return modified

def find_qrs_regions(signal, window_size=50):
    """
    Simple QRS detection based on signal energy
    """
    # Calculate signal energy
    energy = signal ** 2
    
    # Find peaks
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(energy, height=np.percentile(energy, 90), distance=window_size)
    
    regions = []
    for peak in peaks:
        start = max(0, peak - window_size // 2)
        end = min(len(signal), peak + window_size // 2)
        regions.append((start, end))
    
    return regions

def load_dataset_specific(dataset_path, default_label, verbose):
    """
    Load data from a specific dataset directory
    """
    signals, labels, demographics = [], [], []
    
    # Try WFDB format first
    records = find_records(dataset_path)
    
    if len(records) > 0:
        for record_name in records[:10000]:  # Limit per dataset
            try:
                record_path = os.path.join(dataset_path, record_name)
                
                signal, _ = load_signals(record_path)
                processed_signal = process_signal_enhanced(signal)
                if processed_signal is None:
                    continue
                
                # Get label
                label = load_label(record_path)
                if label is None and default_label is not None:
                    label = default_label
                elif label is None:
                    continue
                
                # Get demographics
                header = load_header(record_path)
                demo = extract_demographics_from_header(header)
                
                signals.append(processed_signal)
                labels.append(int(label))
                demographics.append(demo)
                
            except:
                continue
    
    # Try CSV format
    csv_files = [f for f in os.listdir(dataset_path) if f.endswith('.csv')]
    if len(csv_files) > 0 and len(signals) == 0:
        # Implementation for CSV format if needed
        pass
    
    if verbose and len(signals) > 0:
        print(f"Loaded {len(signals)} samples from {os.path.basename(dataset_path)}")
    
    return signals, labels, demographics

def create_baseline_model(model_folder, verbose):
    """
    Create a baseline model when training data is insufficient
    """
    if verbose:
        print("Creating baseline ensemble model...")
    
    # Create simple models
    for i in range(NUM_ENSEMBLE_MODELS):
        model = ECGResNet().to(DEVICE)
        
        # Initialize with small random weights
        for param in model.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param, gain=0.1)
        
        torch.save(model.state_dict(), os.path.join(model_folder, f'model_{i}.pth'))
    
    # Create dummy scaler
    demo_scaler = StandardScaler()
    demo_scaler.fit(np.random.randn(100, 2))
    
    save_ensemble_config(model_folder, demo_scaler, verbose)
    
    return True
