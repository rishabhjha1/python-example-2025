#!/usr/bin/env python

# Memory-optimized Chagas disease detection model
# Implements batch processing and memory-efficient data loading

import os
import numpy as np
import pandas as pd
import h5py
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Dense, Dropout, Conv1D, Input, concatenate, 
                                   BatchNormalization, GlobalAveragePooling1D, 
                                   MultiHeadAttention, LayerNormalization, Add, ReLU)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score
import warnings
import gc
warnings.filterwarnings('ignore')

from helper_code import *

# Memory-optimized constants
TARGET_SAMPLING_RATE = 500
TARGET_SIGNAL_LENGTH = 5000  # 10 seconds at 500Hz
BATCH_SIZE = 8  # Reduced from 16
NUM_LEADS = 12
CHUNK_SIZE = 100  # Process data in chunks

def train_model(data_folder, model_folder, verbose):
    """
    Main training function with memory optimization
    """
    if verbose:
        print("Training Chagas detection model...")
    
    os.makedirs(model_folder, exist_ok=True)
    
    # Load data in chunks to avoid memory overflow
    return train_enhanced_model_chunked(data_folder, model_folder, verbose)

def train_enhanced_model_chunked(data_folder, model_folder, verbose):
    """
    Train model using chunked data loading
    """
    # Initialize data generators
    train_gen = ChunkedDataGenerator(data_folder, chunk_size=CHUNK_SIZE, verbose=verbose)
    
    if train_gen.total_samples == 0:
        raise ValueError("No data loaded. Check your data folder and format.")
    
    if verbose:
        print(f"Total samples found: {train_gen.total_samples}")
    
    # Build model architecture (same as before but optimized)
    model = build_memory_efficient_model()
    
    if verbose:
        print("Model built successfully")
        print(f"Model parameters: {model.count_params():,}")
    
    # Compile with memory-efficient settings
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Memory-efficient training
    callbacks = [
        EarlyStopping(monitor='loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-6),
        tf.keras.callbacks.TerminateOnNaN()
    ]
    
    # Train in batches
    history = train_in_batches(model, train_gen, callbacks, verbose)
    
    # Save model
    save_model_files(model_folder, model, train_gen.demo_scaler, train_gen.ecg_scaler, verbose)
    
    return True

class ChunkedDataGenerator:
    """
    Memory-efficient data generator that loads data in chunks
    """
    
    def __init__(self, data_folder, chunk_size=100, verbose=False):
        self.data_folder = data_folder
        self.chunk_size = chunk_size
        self.verbose = verbose
        self.demo_scaler = RobustScaler()
        self.ecg_scaler = RobustScaler()
        
        # Find all available data
        self.data_files = self._find_data_files()
        self.total_samples = len(self.data_files)
        
        if verbose:
            print(f"Found {self.total_samples} data files")
        
        # Fit scalers on a subset
        self._fit_scalers()
    
    def _find_data_files(self):
        """Find all available data files"""
        data_files = []
        
        # Try HDF5 first
        hdf5_path = os.path.join(self.data_folder, 'exams.hdf5')
        exams_path = os.path.join(self.data_folder, 'exams.csv')
        
        if os.path.exists(hdf5_path) and os.path.exists(exams_path):
            try:
                exams_df = pd.read_csv(exams_path)
                chagas_labels = self._load_chagas_labels()
                
                for idx, row in exams_df.iterrows():
                    exam_id = row.get('exam_id', row.get('id', idx))
                    label = self._get_label_for_exam(exam_id, row, chagas_labels)
                    
                    if label is not None:
                        data_files.append({
                            'type': 'hdf5',
                            'index': idx,
                            'exam_id': exam_id,
                            'label': label,
                            'row': row
                        })
            except Exception as e:
                if self.verbose:
                    print(f"HDF5 indexing error: {e}")
        
        # Try WFDB records
        try:
            records = find_records(self.data_folder)
            for record_name in records:
                record_path = os.path.join(self.data_folder, record_name)
                try:
                    label = load_label(record_path)
                    if label is not None:
                        data_files.append({
                            'type': 'wfdb',
                            'path': record_path,
                            'label': int(label)
                        })
                except:
                    continue
        except Exception as e:
            if self.verbose:
                print(f"WFDB indexing error: {e}")
        
        return data_files
    
    def _load_chagas_labels(self):
        """Load Chagas labels from CSV files"""
        chagas_labels = {}
        label_files = ['samitrop_chagas_labels.csv', 'code15_chagas_labels.csv', 'chagas_labels.csv']
        
        for label_file in label_files:
            label_path = os.path.join(self.data_folder, label_file)
            if os.path.exists(label_path):
                try:
                    label_df = pd.read_csv(label_path)
                    for _, row in label_df.iterrows():
                        exam_id = row.get('exam_id', row.get('id', None))
                        chagas = row.get('chagas', row.get('label', None))
                        
                        if exam_id is not None and chagas is not None:
                            if isinstance(chagas, str):
                                chagas_binary = 1 if chagas.lower() in ['true', 'positive', 'yes', '1'] else 0
                            else:
                                chagas_binary = int(float(chagas))
                            chagas_labels[exam_id] = chagas_binary
                    break
                except Exception as e:
                    continue
        
        return chagas_labels
    
    def _get_label_for_exam(self, exam_id, row, chagas_labels):
        """Get label for exam with source-based inference"""
        if exam_id in chagas_labels:
            return chagas_labels[exam_id]
        
        source = str(row.get('source', '')).lower()
        if 'samitrop' in source or 'sami-trop' in source:
            return 1
        if any(keyword in source for keyword in ['ptb', 'chapman', 'georgia']):
            return 0
        
        return None
    
    def _fit_scalers(self):
        """Fit scalers on a small subset of data"""
        demo_samples = []
        ecg_samples = []
        
        # Use first 50 samples for fitting scalers
        fit_samples = min(50, len(self.data_files))
        
        for i in range(fit_samples):
            try:
                data = self._load_single_sample(i)
                if data is not None:
                    demo_samples.append(data['demographics'])
                    ecg_samples.append(data['ecg_features'])
            except:
                continue
        
        if demo_samples:
            self.demo_scaler.fit(np.array(demo_samples))
            self.ecg_scaler.fit(np.array(ecg_samples))
    
    def _load_single_sample(self, index):
        """Load a single sample by index"""
        if index >= len(self.data_files):
            return None
        
        file_info = self.data_files[index]
        
        try:
            if file_info['type'] == 'hdf5':
                return self._load_hdf5_sample(file_info)
            else:
                return self._load_wfdb_sample(file_info)
        except Exception as e:
            return None
    
    def _load_hdf5_sample(self, file_info):
        """Load single HDF5 sample"""
        hdf5_path = os.path.join(self.data_folder, 'exams.hdf5')
        
        with h5py.File(hdf5_path, 'r') as hdf:
            if 'tracings' in hdf:
                dataset = hdf['tracings']
            elif 'exams' in hdf:
                dataset = hdf['exams']
            else:
                dataset = hdf[list(hdf.keys())[0]]
            
            # Get signal
            if hasattr(dataset, 'shape') and len(dataset.shape) == 3:
                signal = dataset[file_info['index']]
            elif str(file_info['exam_id']) in dataset:
                signal = dataset[str(file_info['exam_id'])][:]
            else:
                return None
            
            # Process signal
            processed_signal = process_signal_memory_efficient(signal)
            if processed_signal is None:
                return None
            
            # Extract features
            demographics = extract_demographics(file_info['row'])
            ecg_features = extract_ecg_features_efficient(processed_signal)
            
            return {
                'signal': processed_signal,
                'demographics': demographics,
                'ecg_features': ecg_features,
                'label': file_info['label']
            }
    
    def _load_wfdb_sample(self, file_info):
        """Load single WFDB sample"""
        signal, fields = load_signals(file_info['path'])
        header = load_header(file_info['path'])
        
        processed_signal = process_signal_memory_efficient(signal)
        if processed_signal is None:
            return None
        
        demographics = extract_demographics_wfdb(header)
        ecg_features = extract_ecg_features_efficient(processed_signal)
        
        return {
            'signal': processed_signal,
            'demographics': demographics,
            'ecg_features': ecg_features,
            'label': file_info['label']
        }
    
    def get_batch(self, indices):
        """Get a batch of samples"""
        signals = []
        demographics = []
        ecg_features = []
        labels = []
        
        for idx in indices:
            try:
                data = self._load_single_sample(idx)
                if data is not None:
                    signals.append(data['signal'])
                    demographics.append(data['demographics'])
                    ecg_features.append(data['ecg_features'])
                    labels.append(data['label'])
            except:
                continue
        
        if not signals:
            return None
        
        # Convert to arrays
        X_signal = np.array(signals, dtype=np.float32)
        X_demo = self.demo_scaler.transform(np.array(demographics, dtype=np.float32))
        X_ecg = self.ecg_scaler.transform(np.array(ecg_features, dtype=np.float32))
        y = np.array(labels, dtype=np.int32)
        
        return [X_signal, X_demo, X_ecg], y

def process_signal_memory_efficient(signal):
    """Memory-efficient signal processing"""
    try:
        # Convert to float32 early to save memory
        signal = np.array(signal, dtype=np.float32)
        
        # Handle shape
        if len(signal.shape) == 1:
            signal = signal.reshape(-1, 1)
        elif signal.shape[0] < signal.shape[1] and signal.shape[0] <= 12:
            signal = signal.T
        
        # Ensure 12 leads
        if signal.shape[1] > NUM_LEADS:
            signal = signal[:, :NUM_LEADS]
        elif signal.shape[1] < NUM_LEADS:
            padding_needed = NUM_LEADS - signal.shape[1]
            if signal.shape[1] > 0:
                last_lead = signal[:, -1:] 
                padding = np.tile(last_lead, (1, padding_needed))
                signal = np.hstack([signal, padding])
            else:
                signal = np.zeros((signal.shape[0], NUM_LEADS), dtype=np.float32)
        
        # Resample more efficiently
        signal = resample_signal_efficient(signal, TARGET_SIGNAL_LENGTH)
        
        # Normalize in-place
        normalize_signal_inplace(signal)
        
        return signal
    
    except Exception as e:
        return None

def resample_signal_efficient(signal, target_length):
    """Memory-efficient resampling"""
    current_length = signal.shape[0]
    
    if current_length == target_length:
        return signal
    
    # Use integer indices when possible
    if current_length > target_length:
        # Downsample by selecting every nth sample
        step = current_length / target_length
        indices = np.round(np.arange(0, current_length, step)[:target_length]).astype(int)
        indices = np.clip(indices, 0, current_length - 1)
        return signal[indices]
    else:
        # Upsample using linear interpolation
        x_old = np.linspace(0, 1, current_length)
        x_new = np.linspace(0, 1, target_length)
        
        resampled = np.zeros((target_length, signal.shape[1]), dtype=np.float32)
        for i in range(signal.shape[1]):
            resampled[:, i] = np.interp(x_new, x_old, signal[:, i])
        
        return resampled

def normalize_signal_inplace(signal):
    """In-place signal normalization to save memory"""
    for i in range(signal.shape[1]):
        # Remove DC
        median_val = np.median(signal[:, i])
        signal[:, i] -= median_val
        
        # Robust scaling using MAD
        mad = np.median(np.abs(signal[:, i]))
        
        if mad > 1e-6:
            signal[:, i] /= (mad * 1.4826)
        
        # Clip extreme values
        np.clip(signal[:, i], -5, 5, out=signal[:, i])

def extract_ecg_features_efficient(signal):
    """Memory-efficient ECG feature extraction"""
    features = np.zeros(11, dtype=np.float32)
    
    # Use lead II or lead I
    lead_signal = signal[:, 1] if signal.shape[1] > 1 else signal[:, 0]
    
    # Simple heart rate estimation
    try:
        threshold = np.std(lead_signal) * 0.5
        min_distance = TARGET_SAMPLING_RATE // 3
        
        # Find peaks more efficiently
        above_threshold = lead_signal > threshold
        peaks = []
        
        i = min_distance
        while i < len(lead_signal) - min_distance:
            if (above_threshold[i] and 
                lead_signal[i] > lead_signal[i-1] and 
                lead_signal[i] > lead_signal[i+1]):
                peaks.append(i)
                i += min_distance  # Skip ahead
            else:
                i += 1
        
        if len(peaks) >= 2:
            rr_intervals = np.diff(peaks) / TARGET_SAMPLING_RATE
            features[0] = 60.0 / np.mean(rr_intervals)  # HR
            features[1] = np.std(rr_intervals) * 1000   # HRV
        else:
            features[0] = 70.0
            features[1] = 20.0
            
    except:
        features[0] = 70.0
        features[1] = 20.0
    
    # Signal statistics for first 3 leads
    for i in range(min(3, signal.shape[1])):
        lead = signal[:, i]
        base_idx = 2 + i * 3
        features[base_idx] = np.mean(lead)
        features[base_idx + 1] = np.std(lead)
        features[base_idx + 2] = np.ptp(lead)  # Peak-to-peak
    
    return features

def build_memory_efficient_model():
    """Build a more memory-efficient model"""
    # Smaller model to reduce memory usage
    signal_input = Input(shape=(TARGET_SIGNAL_LENGTH, NUM_LEADS), name='signal_input')
    
    # Lighter CNN
    x = Conv1D(32, 15, strides=2, padding='same')(signal_input)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    # Fewer residual blocks
    for filters in [64, 128]:
        residual = x
        
        x = Conv1D(filters, 7, padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        
        x = Conv1D(filters, 7, padding='same')(x)
        x = BatchNormalization()(x)
        
        if residual.shape[-1] != filters:
            residual = Conv1D(filters, 1, padding='same')(residual)
            residual = BatchNormalization()(residual)
        
        x = Add()([x, residual])
        x = ReLU()(x)
        x = Conv1D(filters, 3, strides=2, padding='same')(x)
    
    # Simpler attention
    x = MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
    x = LayerNormalization()(x)
    
    # Global pooling
    x = GlobalAveragePooling1D()(x)
    signal_features = Dense(64, activation='relu')(x)
    signal_features = Dropout(0.3)(signal_features)
    
    # Demographics branch
    demo_input = Input(shape=(2,), name='demo_input')
    demo_branch = Dense(8, activation='relu')(demo_input)
    
    # ECG features branch
    ecg_input = Input(shape=(11,), name='ecg_input')
    ecg_branch = Dense(16, activation='relu')(ecg_input)
    
    # Fusion
    combined = concatenate([signal_features, demo_branch, ecg_branch])
    combined = Dense(32, activation='relu')(combined)
    combined = Dropout(0.4)(combined)
    
    # Output
    output = Dense(1, activation='sigmoid')(combined)
    
    model = Model(inputs=[signal_input, demo_input, ecg_input], outputs=output)
    return model

def train_in_batches(model, data_gen, callbacks, verbose):
    """Train model in memory-efficient batches"""
    # Create balanced batches
    indices = np.arange(data_gen.total_samples)
    np.random.shuffle(indices)
    
    # Calculate class weights from a sample
    sample_labels = []
    for i in range(0, min(500, len(indices)), 10):
        try:
            data = data_gen._load_single_sample(indices[i])
            if data:
                sample_labels.append(data['label'])
        except:
            continue
    
    if sample_labels:
        class_weights = compute_class_weight('balanced', classes=np.unique(sample_labels), y=sample_labels)
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    else:
        class_weight_dict = {0: 1.0, 1: 1.0}
    
    # Training loop
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(50):  # Reduced epochs
        if verbose:
            print(f"Epoch {epoch + 1}/30")
        
        epoch_losses = []
        np.random.shuffle(indices)
        
        # Process in chunks
        for i in range(0, len(indices), BATCH_SIZE):
            batch_indices = indices[i:i + BATCH_SIZE]
            
            try:
                batch_data = data_gen.get_batch(batch_indices)
                if batch_data is None:
                    continue
                
                X_batch, y_batch = batch_data
                
                # Train on batch
                loss = model.train_on_batch(X_batch, y_batch, class_weight=class_weight_dict)
                epoch_losses.append(loss)
                
                # Cleanup
                del X_batch, y_batch
                gc.collect()
                
            except Exception as e:
                if verbose:
                    print(f"Batch error: {e}")
                continue
        
        if epoch_losses:
            avg_loss = np.mean(epoch_losses)
            if verbose:
                print(f"Average loss: {avg_loss:.4f}")
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= 10:
                if verbose:
                    print("Early stopping triggered")
                break
        
        # Manual garbage collection
        gc.collect()
    
    return None

def extract_demographics(row):
    """Extract demographic features"""
    age = row.get('age', 50.0)
    if pd.isna(age):
        age = 50.0
    age_norm = np.clip(float(age) / 100.0, 0.0, 1.0)
    
    sex = row.get('sex', row.get('is_male', 0))
    if pd.isna(sex):
        sex_male = 0.5
    else:
        if isinstance(sex, str):
            sex_male = 1.0 if sex.lower().startswith('m') else 0.0
        else:
            sex_male = float(sex)
    
    return np.array([age_norm, sex_male], dtype=np.float32)

def extract_demographics_wfdb(header):
    """Extract demographics from WFDB header"""
    age = get_age(header)
    sex = get_sex(header)
    
    age_norm = 0.5
    if age is not None:
        age_norm = np.clip(float(age) / 100.0, 0.0, 1.0)
    
    sex_male = 0.5
    if sex is not None:
        sex_male = 1.0 if sex.lower().startswith('m') else 0.0
    
    return np.array([age_norm, sex_male], dtype=np.float32)

def save_model_files(model_folder, model, demo_scaler, ecg_scaler, verbose):
    """Save model and scalers"""
    model.save(os.path.join(model_folder, 'model.keras'))
    
    import joblib
    joblib.dump(demo_scaler, os.path.join(model_folder, 'demo_scaler.pkl'))
    joblib.dump(ecg_scaler, os.path.join(model_folder, 'ecg_scaler.pkl'))
    
    config = {
        'signal_length': TARGET_SIGNAL_LENGTH,
        'num_leads': NUM_LEADS,
        'sampling_rate': TARGET_SAMPLING_RATE,
        'demo_features': 2,
        'ecg_features': 11
    }
    
    import json
    with open(os.path.join(model_folder, 'config.json'), 'w') as f:
        json.dump(config, f)
    
    if verbose:
        print(f"Model saved to {model_folder}")

def load_model(model_folder, verbose=False):
    """Load trained model"""
    model = tf.keras.models.load_model(os.path.join(model_folder, 'model.keras'))
    
    import joblib
    demo_scaler = joblib.load(os.path.join(model_folder, 'demo_scaler.pkl'))
    ecg_scaler = joblib.load(os.path.join(model_folder, 'ecg_scaler.pkl'))
    
    import json
    with open(os.path.join(model_folder, 'config.json'), 'r') as f:
        config = json.load(f)
    
    return {
        'model': model,
        'demo_scaler': demo_scaler,
        'ecg_scaler': ecg_scaler,
        'config': config
    }

def run_model(record, model_data, verbose=False):
    """Run model on single record"""
    try:
        model = model_data['model']
        demo_scaler = model_data['demo_scaler']
        ecg_scaler = model_data['ecg_scaler']
        config = model_data['config']
        
        # Load and process signal
        signal, fields = load_signals(record)
        processed_signal = process_signal_memory_efficient(signal)
        
        if processed_signal is None:
            processed_signal = np.random.randn(config['signal_length'], config['num_leads']).astype(np.float32)
        
        # Extract features
        ecg_features = extract_ecg_features_efficient(processed_signal)
        header = load_header(record)
        demographics = extract_demographics_wfdb(header)
        
        # Prepare inputs
        signal_input = processed_signal.reshape(1, config['signal_length'], config['num_leads'])
        demo_input = demo_scaler.transform(demographics.reshape(1, -1))
        ecg_input = ecg_scaler.transform(ecg_features.reshape(1, -1))
        
        # Predict
        probability = float(model.predict([signal_input, demo_input, ecg_input], verbose=0)[0][0])
        binary_prediction = 1 if probability >= 0.3 else 0
        
        return binary_prediction, probability
        
    except Exception as e:
        if verbose:
            print(f"Error in run_model: {e}")
        return 0, 0.1
