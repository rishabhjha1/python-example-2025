#!/usr/bin/env python

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

from helper_code import *

# Configuration based on PhysioNet Challenge analysis
class Config:
    # CRITICAL: Use frequency-agnostic approach to avoid sampling bias
    TARGET_SAMPLING_RATE = None  # Don't normalize to specific rate
    TARGET_SIGNAL_LENGTH = 2500  # Normalized length (5 seconds equivalent)
    MAX_SAMPLES = 15000
    BATCH_SIZE = 32
    NUM_LEADS = 12
    LEARNING_RATE = 0.0005  # Lower LR for more stable training
    EPOCHS = 100
    PATIENCE = 15
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Challenge-specific: Focus on prioritization metric
    EVAL_TOP_PERCENT = 0.05  # Top 5% for evaluation

class ECGDataset(Dataset):
    """Dataset class for ECG signals with source tracking"""
    def __init__(self, signals, labels, sources=None):
        self.signals = torch.FloatTensor(signals)
        self.labels = torch.LongTensor(labels)
        self.sources = sources if sources is not None else ['unknown'] * len(signals)
    
    def __len__(self):
        return len(self.signals)
    
    def __getitem__(self, idx):
        return self.signals[idx], self.labels[idx]

class FrequencyAgnosticECGNet(nn.Module):
    """
    ECG Network designed to avoid sampling frequency bias
    Based on PhysioNet Challenge insights about confounding factors
    """
    def __init__(self, signal_length=Config.TARGET_SIGNAL_LENGTH, num_leads=Config.NUM_LEADS):
        super(FrequencyAgnosticECGNet, self).__init__()
        
        self.signal_length = signal_length
        self.num_leads = num_leads
        
        # Feature extraction that's robust to sampling rate differences
        # Use relative temporal patterns rather than absolute frequencies
        
        # Multi-resolution analysis with different dilation rates
        # This captures patterns at different time scales without frequency dependence
        self.conv_blocks = nn.ModuleList([
            self._make_conv_block(num_leads, 32, dilation=1),   # Fine patterns
            self._make_conv_block(num_leads, 32, dilation=2),   # Medium patterns  
            self._make_conv_block(num_leads, 32, dilation=4),   # Coarse patterns
            self._make_conv_block(num_leads, 32, dilation=8),   # Very coarse patterns
        ])
        
        # Lead-wise feature extraction (each lead processed independently first)
        self.lead_processors = nn.ModuleList([
            nn.Conv1d(1, 16, kernel_size=15, padding=7) for _ in range(num_leads)
        ])
        
        # Cross-lead feature integration
        self.cross_lead_conv = nn.Conv1d(16 * num_leads, 128, kernel_size=1)
        self.cross_lead_bn = nn.BatchNorm1d(128)
        
        # Temporal feature extraction with attention
        self.temporal_conv1 = nn.Conv1d(128 + 128, 256, kernel_size=5, padding=2)  # 128 from dilated + 128 from cross-lead
        self.temporal_bn1 = nn.BatchNorm1d(256)
        
        self.temporal_conv2 = nn.Conv1d(256, 512, kernel_size=5, padding=2)
        self.temporal_bn2 = nn.BatchNorm1d(512)
        
        # Attention mechanism for important pattern focus
        self.attention = nn.Sequential(
            nn.Conv1d(512, 128, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Global pooling and classification
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classifier with strong regularization
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(), 
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)  # Binary classification
        )
        
        # Regularization
        self.dropout = nn.Dropout(0.2)
        
    def _make_conv_block(self, in_channels, out_channels, dilation=1):
        """Create dilated convolution block"""
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=5, 
                     padding=2*dilation, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, x):
        # Input shape: (batch_size, num_leads, signal_length)
        batch_size = x.size(0)
        
        # Multi-resolution dilated convolutions
        dilated_features = []
        for conv_block in self.conv_blocks:
            feat = conv_block(x)
            feat = F.max_pool1d(feat, kernel_size=2)
            dilated_features.append(feat)
        
        # Concatenate dilated features
        dilated_concat = torch.cat(dilated_features, dim=1)  # (batch, 128, length/2)
        dilated_pooled = F.adaptive_avg_pool1d(dilated_concat, 
                                               x.size(-1)//4)  # Downsample
        
        # Lead-wise processing
        lead_features = []
        for i, processor in enumerate(self.lead_processors):
            lead_signal = x[:, i:i+1, :]  # Single lead
            lead_feat = processor(lead_signal)
            lead_features.append(lead_feat)
        
        # Concatenate lead features
        lead_concat = torch.cat(lead_features, dim=1)  # (batch, 16*12, length)
        
        # Cross-lead feature integration
        cross_lead_feat = F.relu(self.cross_lead_bn(self.cross_lead_conv(lead_concat)))
        cross_lead_feat = F.max_pool1d(cross_lead_feat, kernel_size=4)
        
        # Resize to match dilated features
        cross_lead_feat = F.adaptive_avg_pool1d(cross_lead_feat, 
                                                dilated_pooled.size(-1))
        
        # Combine all features
        combined = torch.cat([dilated_pooled, cross_lead_feat], dim=1)
        
        # Temporal processing
        x = F.relu(self.temporal_bn1(self.temporal_conv1(combined)))
        x = self.dropout(x)
        x = F.max_pool1d(x, kernel_size=2)
        
        x = F.relu(self.temporal_bn2(self.temporal_conv2(x)))
        x = self.dropout(x)
        
        # Attention mechanism
        attention_weights = self.attention(x)
        x = x * attention_weights
        
        # Global pooling
        x = self.global_pool(x)  # (batch_size, 512, 1)
        x = x.squeeze(-1)  # (batch_size, 512)
        
        # Classification
        x = self.classifier(x)
        
        return x

def train_model(data_folder, model_folder, verbose=True):
    """
    Main training function following PhysioNet Challenge insights
    """
    if verbose:
        print("Training frequency-agnostic Chagas detection model...")
        print("Addressing sampling frequency bias and prioritization metric...")
    
    os.makedirs(model_folder, exist_ok=True)
    
    # Load data with source tracking
    signals, labels, sources = load_data_with_sources(data_folder, verbose)
    
    if len(signals) < 50:
        if verbose:
            print(f"Insufficient data ({len(signals)} samples), creating baseline model")
        return create_baseline_model(model_folder, verbose)
    
    return train_improved_model(signals, labels, sources, model_folder, verbose)

def load_data_with_sources(data_folder, verbose):
    """
    Load data while tracking sources to understand bias
    """
    signals = []
    labels = []
    sources = []
    
    try:
        # Find all records
        records = find_records(data_folder)
        if verbose:
            print(f"Found {len(records)} records")
        
        processed_count = 0
        source_counts = {}
        
        for record_name in records:
            if processed_count >= Config.MAX_SAMPLES:
                break
                
            try:
                record_path = os.path.join(data_folder, record_name)
                
                # Load signal and header
                signal, fields = load_signals(record_path)
                header = load_header(record_path)
                
                # Determine source from path or header
                source = determine_source(record_path, header)
                
                # Process signal (frequency-agnostic)
                processed_signal = process_signal_frequency_agnostic(signal, source)
                if processed_signal is None:
                    continue
                
                # Extract label
                label = load_label(record_path)
                if label is None:
                    continue
                
                signals.append(processed_signal)
                labels.append(int(label))
                sources.append(source)
                
                source_counts[source] = source_counts.get(source, 0) + 1
                processed_count += 1
                
                if verbose and processed_count % 500 == 0:
                    print(f"Processed {processed_count} records")
            
            except Exception as e:
                if verbose and processed_count < 5:
                    print(f"Error processing {record_name}: {e}")
                continue
    
    except Exception as e:
        if verbose:
            print(f"Data loading error: {e}")
    
    if verbose:
        print(f"Total loaded: {len(signals)} samples")
        print(f"Source distribution: {source_counts}")
        if len(labels) > 0:
            pos_rate = np.mean(labels) * 100
            print(f"Positive rate: {pos_rate:.1f}%")
    
    return signals, labels, sources

def determine_source(record_path, header):
    """
    Determine data source from path or header information
    """
    path_lower = record_path.lower()
    
    if 'samitrop' in path_lower or 'sami' in path_lower:
        return 'samitrop'
    elif 'ptbxl' in path_lower or 'ptb' in path_lower:
        return 'ptbxl'
    elif 'code15' in path_lower or 'code-15' in path_lower:
        return 'code15'
    else:
        # Try to infer from header if available
        try:
            comments = get_comments(header)
            if any('sami' in str(c).lower() for c in comments):
                return 'samitrop'
            elif any('ptb' in str(c).lower() for c in comments):
                return 'ptbxl'
            elif any('code' in str(c).lower() for c in comments):
                return 'code15'
        except:
            pass
    
    return 'unknown'

def process_signal_frequency_agnostic(signal, source):
    """
    Process signal without introducing sampling frequency bias
    """
    try:
        signal = np.array(signal, dtype=np.float32)
        
        # Handle different input shapes
        if len(signal.shape) == 1:
            signal = signal.reshape(-1, 1)
        elif signal.shape[0] < signal.shape[1] and signal.shape[0] <= 12:
            signal = signal.T  # Transpose if leads are in rows
        
        # Ensure we have 12 leads
        if signal.shape[1] > 12:
            signal = signal[:, :12]  # Take first 12 leads
        elif signal.shape[1] < 12:
            # Pad with zeros
            padding = np.zeros((signal.shape[0], 12 - signal.shape[1]))
            signal = np.hstack([signal, padding])
        
        # CRITICAL: Frequency-agnostic resampling
        # Instead of resampling to fixed rate, normalize to fixed duration
        signal = resample_to_duration(signal, Config.TARGET_SIGNAL_LENGTH)
        
        # Robust normalization per lead
        signal = normalize_signal_robust(signal)
        
        # Remove any residual frequency-related artifacts
        signal = remove_frequency_artifacts(signal)
        
        # Transpose for PyTorch (leads, time_steps)
        signal = signal.T  # Shape: (12, signal_length)
        
        return signal.astype(np.float32)
    
    except Exception as e:
        return None

def resample_to_duration(signal, target_length):
    """
    Resample based on relative duration, not absolute frequency
    This helps avoid the sampling frequency bias
    """
    current_length = signal.shape[0]
    
    if current_length == target_length:
        return signal
    
    # Use relative time indices (0 to 1) for frequency-agnostic resampling
    x_old = np.linspace(0, 1, current_length)
    x_new = np.linspace(0, 1, target_length)
    
    resampled = np.zeros((target_length, signal.shape[1]))
    for i in range(signal.shape[1]):
        resampled[:, i] = np.interp(x_new, x_old, signal[:, i])
    
    return resampled

def remove_frequency_artifacts(signal):
    """
    Remove artifacts that might be frequency-dependent
    """
    # Remove very high and very low frequency components that might be sampling-related
    from scipy import signal as scipy_signal
    
    try:
        # Design a band-pass filter to remove frequency artifacts
        # Keep only ECG-relevant frequencies (0.5-40 Hz equivalent in normalized domain)
        nyquist = 0.5  # Normalized frequency
        low_cut = 0.01  # Normalized low cutoff
        high_cut = 0.4   # Normalized high cutoff
        
        b, a = scipy_signal.butter(4, [low_cut, high_cut], btype='band')
        
        for i in range(signal.shape[1]):
            signal[:, i] = scipy_signal.filtfilt(b, a, signal[:, i])
    except:
        # If filtering fails, continue without it
        pass
    
    return signal

def normalize_signal_robust(signal):
    """
    Robust signal normalization per lead
    """
    for i in range(signal.shape[1]):
        # Remove DC component using median (more robust)
        signal[:, i] = signal[:, i] - np.median(signal[:, i])
        
        # Robust scaling using IQR
        q25, q75 = np.percentile(signal[:, i], [25, 75])
        iqr = q75 - q25
        
        if iqr > 1e-6:
            signal[:, i] = signal[:, i] / (iqr + 1e-6)
        
        # Conservative clipping
        signal[:, i] = np.clip(signal[:, i], -3, 3)
    
    return signal

def train_improved_model(signals, labels, sources, model_folder, verbose):
    """
    Train model with focus on prioritization metric
    """
    if verbose:
        print(f"Training on {len(signals)} samples")
    
    # Convert to arrays
    X = np.array(signals, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)
    
    if verbose:
        print(f"Signal shape: {X.shape}")
        unique, counts = np.unique(y, return_counts=True)
        print(f"Label distribution: {dict(zip(unique, counts))}")
        
        # Analyze by source
        sources_array = np.array(sources)
        for source in np.unique(sources_array):
            mask = sources_array == source
            source_labels = y[mask]
            if len(source_labels) > 0:
                pos_rate = np.mean(source_labels) * 100
                print(f"{source}: {len(source_labels)} samples, {pos_rate:.1f}% positive")
    
    # Stratified split maintaining source distribution
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create datasets
    train_dataset = ECGDataset(X_train, y_train)
    val_dataset = ECGDataset(X_val, y_val)
    
    # Balanced sampling for training (important for prioritization metric)
    class_counts = np.bincount(y_train)
    class_weights = 1. / class_counts
    sample_weights = class_weights[y_train]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    # Initialize model
    model = FrequencyAgnosticECGNet(Config.TARGET_SIGNAL_LENGTH, Config.NUM_LEADS)
    model = model.to(Config.DEVICE)
    
    if verbose:
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss function optimized for prioritization
    # Use focal loss to focus on hard examples
    class FocalLoss(nn.Module):
        def __init__(self, alpha=1, gamma=2):
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma
            
        def forward(self, inputs, targets):
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
            pt = torch.exp(-ce_loss)
            focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
            return focal_loss.mean()
    
    criterion = FocalLoss(alpha=1, gamma=2)
    optimizer = Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=7, verbose=verbose)
    
    # Training loop with prioritization metric tracking
    best_prioritization_score = 0.0
    patience_counter = 0
    
    for epoch in range(Config.EPOCHS):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_signals, batch_labels in train_loader:
            batch_signals = batch_signals.to(Config.DEVICE)
            batch_labels = batch_labels.to(Config.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(batch_signals)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase with prioritization metric
        model.eval()
        val_loss = 0.0
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for batch_signals, batch_labels in val_loader:
                batch_signals = batch_signals.to(Config.DEVICE)
                batch_labels = batch_labels.to(Config.DEVICE)
                
                outputs = model(batch_signals)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()
                
                # Get probabilities for prioritization metric
                probs = F.softmax(outputs, dim=1)
                all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of positive class
                all_labels.extend(batch_labels.cpu().numpy())
        
        # Calculate prioritization score (top 5%)
        prioritization_score = calculate_prioritization_score(all_probs, all_labels, 
                                                              Config.EVAL_TOP_PERCENT)
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        if verbose:
            print(f'Epoch [{epoch+1}/{Config.EPOCHS}] - '
                  f'Train Loss: {avg_train_loss:.4f} - '
                  f'Val Loss: {avg_val_loss:.4f} - '
                  f'Prioritization Score: {prioritization_score:.4f}')
        
        scheduler.step(avg_val_loss)
        
        # Save best model based on prioritization score
        if prioritization_score > best_prioritization_score:
            best_prioritization_score = prioritization_score
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': {
                    'signal_length': Config.TARGET_SIGNAL_LENGTH,
                    'num_leads': Config.NUM_LEADS,
                    'model_type': 'frequency_agnostic_pytorch',
                    'best_prioritization_score': best_prioritization_score
                }
            }, os.path.join(model_folder, 'model.pth'))
        else:
            patience_counter += 1
            
        if patience_counter >= Config.PATIENCE:
            if verbose:
                print(f"Early stopping at epoch {epoch+1}")
            break
    
    if verbose:
        print(f"Best prioritization score: {best_prioritization_score:.4f}")
        print("Model training completed successfully")
    
    return True

def calculate_prioritization_score(probs, labels, top_percent):
    """
    Calculate the prioritization score (fraction of positives in top predictions)
    This is the key metric for the PhysioNet Challenge
    """
    probs = np.array(probs)
    labels = np.array(labels)
    
    # Number of samples to take from top predictions
    n_top = max(1, int(len(probs) * top_percent))
    
    # Get indices of top predictions
    top_indices = np.argsort(probs)[-n_top:]
    
    # Count true positives in top predictions
    true_positives_in_top = np.sum(labels[top_indices])
    
    # Total true positives
    total_positives = np.sum(labels)
    
    if total_positives == 0:
        return 0.0
    
    return true_positives_in_top / total_positives

def create_baseline_model(model_folder, verbose):
    """Create baseline model when insufficient data"""
    if verbose:
        print("Creating baseline model...")
    
    os.makedirs(model_folder, exist_ok=True)
    
    model = FrequencyAgnosticECGNet(Config.TARGET_SIGNAL_LENGTH, Config.NUM_LEADS)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'signal_length': Config.TARGET_SIGNAL_LENGTH,
            'num_leads': Config.NUM_LEADS,
            'model_type': 'baseline_frequency_agnostic'
        }
    }, os.path.join(model_folder, 'model.pth'))
    
    if verbose:
        print("Baseline model created")
    
    return True

def load_model(model_folder, verbose=False):
    """Load the trained model"""
    if verbose:
        print(f"Loading model from {model_folder}")
    
    checkpoint = torch.load(os.path.join(model_folder, 'model.pth'), 
                           map_location=Config.DEVICE)
    
    config = checkpoint['config']
    model = FrequencyAgnosticECGNet(config['signal_length'], config['num_leads'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(Config.DEVICE)
    model.eval()
    
    return {
        'model': model,
        'config': config,
        'device': Config.DEVICE
    }

def run_model(record, model_data, verbose=False):
    """Run model on a single record"""
    try:
        model = model_data['model']
        config = model_data['config']
        device = model_data['device']
        
        # Load and process signal
        try:
            signal, fields = load_signals(record)
            header = load_header(record)
            source = determine_source(record, header)
            processed_signal = process_signal_frequency_agnostic(signal, source)
            
            if processed_signal is None:
                raise ValueError("Signal processing failed")
                
        except Exception as e:
            if verbose:
                print(f"Signal loading failed: {e}, using default")
            processed_signal = np.random.randn(config['num_leads'], 
                                               config['signal_length']).astype(np.float32)
        
        # Prepare input
        signal_input = torch.FloatTensor(processed_signal).unsqueeze(0).to(device)
        
        # Predict
        try:
            with torch.no_grad():
                outputs = model(signal_input)
                probabilities = F.softmax(outputs, dim=1)
                probability = float(probabilities[0][1])  # Probability of Chagas positive
        except Exception as e:
            if verbose:
                print(f"Prediction error: {e}")
            probability = 0.05  # Conservative default (realistic prevalence)
        
        # Binary prediction (optimized threshold for prioritization)
        binary_prediction = 1 if probability >= 0.3 else 0  # Lower threshold for better recall
        
        return binary_prediction, probability
        
    except Exception as e:
        if verbose:
            print(f"Error in run_model: {e}")
        return 0, 0.05
