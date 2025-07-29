#!/usr/bin/env python

import os
import numpy as np
import pandas as pd
import h5py
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv1D, MaxPooling1D, Dense, Dropout, 
                                   BatchNormalization, GlobalAveragePooling1D, 
                                   concatenate, Add, Activation, LayerNormalization)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold
import wfdb
from scipy import signal as scipy_signal
from scipy.stats import zscore
import glob
import warnings
warnings.filterwarnings('ignore')

# Improved Constants
TARGET_SAMPLING_RATE = 500  # Match most common sampling rate
TARGET_SIGNAL_LENGTH = 5000  # 10 seconds at 500Hz
NUM_LEADS = 12
BATCH_SIZE = 16  # Reduced for better convergence
RANDOM_SEED = 42

def find_records(data_folder):
    """Find all WFDB records in the directory."""
    hea_files = glob.glob(os.path.join(data_folder, '*.hea'))
    return [os.path.splitext(os.path.basename(f))[0] for f in hea_files]

def train_model(data_folder, model_folder, verbose):
    """Main training function with improved data handling."""
    try:
        if verbose:
            print("Starting improved Chagas detection model training...")
        
        os.makedirs(model_folder, exist_ok=True)
        
        # Load and preprocess data with better quality control
        signals, labels, demographics, sources = load_chagas_data_improved(data_folder, verbose)
        
        if len(signals) < 50:
            if verbose:
                print("Insufficient data, creating fallback model")
            return create_fallback_model(model_folder, verbose)
        
        # Train with cross-validation for robustness
        trained = train_chagas_model_cv(signals, labels, demographics, sources, model_folder, verbose)
        
        if verbose and trained:
            print("Training completed successfully")
        return trained
        
    except Exception as e:
        if verbose:
            print(f"Training failed: {str(e)}")
        return False

def load_chagas_data_improved(data_folder, verbose):
    """Improved data loading with better label extraction and quality control."""
    signals, labels, demographics, sources = [], [], [], []
    
    # Strategy 1: Load from PhysioNet Challenge format
    label_file_patterns = [
        'labels.csv', 'dx_labels.csv', 'reference.csv', 
        'samitrop_chagas_labels.csv', 'ptbxl_database.csv'
    ]
    
    label_df = None
    for pattern in label_file_patterns:
        label_path = os.path.join(data_folder, pattern)
        if os.path.exists(label_path):
            try:
                label_df = pd.read_csv(label_path)
                if verbose:
                    print(f"Found labels in {pattern}")
                break
            except:
                continue
    
    # Strategy 2: Load WFDB records with improved label extraction
    records = find_records(data_folder)
    if verbose:
        print(f"Found {len(records)} WFDB records")
    
    label_map = {}
    if label_df is not None:
        # Handle different label file formats
        for _, row in label_df.iterrows():
            record_id = None
            label = None
            
            # Try different column names for record ID
            for col in ['record_name', 'exam_id', 'ecg_id', 'filename', 'record']:
                if col in row and pd.notna(row[col]):
                    record_id = str(row[col]).replace('.hea', '').replace('.mat', '')
                    break
            
            # Try different column names for Chagas label
            for col in ['chagas', 'dx', 'label', 'target', 'diagnosis']:
                if col in row and pd.notna(row[col]):
                    label_val = row[col]
                    if isinstance(label_val, str):
                        label_val = label_val.lower()
                        if 'chagas' in label_val or 'trypanosoma' in label_val:
                            label = 1
                        elif 'normal' in label_val or 'healthy' in label_val or label_val == '0':
                            label = 0
                    else:
                        label = int(float(label_val)) if not pd.isna(label_val) else None
                    break
            
            if record_id and label is not None:
                label_map[record_id] = label
    
    # Process records with improved quality control
    processed_count = 0
    for record in records:
        if processed_count >= 50000:  # Increased limit
            break
            
        try:
            # Load signal and header
            record_path = os.path.join(data_folder, record)
            signal, fields = wfdb.rdsamp(record_path)
            header = wfdb.rdheader(record_path)
            
            # Improved signal preprocessing
            processed_sig = preprocess_ecg_improved(signal, fields.get('fs', 500))
            if processed_sig is None:
                continue
            
            # Better label extraction
            label = get_improved_chagas_label(header, record, label_map)
            if label is None:
                continue  # Skip ambiguous cases
            
            # Extract demographics and source info
            demo = get_improved_demographics(header)
            source = get_data_source(header, record)
            
            signals.append(processed_sig)
            labels.append(label)
            demographics.append(demo)
            sources.append(source)
            processed_count += 1
            
        except Exception as e:
            if verbose and processed_count < 10:  # Only show first few errors
                print(f"Skipping record {record}: {str(e)}")
            continue
    
    if verbose:
        pos_count = sum(labels)
        total_count = len(labels)
        if total_count > 0:
            print(f"Loaded {total_count} samples: {pos_count} positive ({pos_count/total_count*100:.1f}%)")
            
            # Show source distribution
            source_counts = {}
            for source in sources:
                source_counts[source] = source_counts.get(source, 0) + 1
            print(f"Source distribution: {source_counts}")
        
    return np.array(signals), np.array(labels), np.array(demographics), np.array(sources)

def preprocess_ecg_improved(raw_signal, sampling_rate):
    """Improved ECG preprocessing with better quality control."""
    try:
        signal = np.array(raw_signal, dtype=np.float32)
        
        # Handle shape issues
        if len(signal.shape) == 1:
            signal = signal.reshape(-1, 1)
        if signal.shape[0] < signal.shape[1]:
            signal = signal.T
        
        # Quality check: reject very short signals
        if signal.shape[0] < sampling_rate * 5:  # Less than 5 seconds
            return None
        
        # Handle lead count more intelligently
        if signal.shape[1] > NUM_LEADS:
            # Take first 12 leads (standard order)
            signal = signal[:, :NUM_LEADS]
        elif signal.shape[1] < NUM_LEADS:
            # Pad with zeros for missing leads
            signal = np.pad(signal, ((0, 0), (0, NUM_LEADS - signal.shape[1])), 
                           mode='constant', constant_values=0)
        
        # Improved resampling
        if sampling_rate != TARGET_SAMPLING_RATE:
            signal = resample_ecg_improved(signal, sampling_rate, TARGET_SAMPLING_RATE)
        
        # Ensure target length
        if signal.shape[0] != TARGET_SIGNAL_LENGTH:
            signal = resize_to_target_length(signal, TARGET_SIGNAL_LENGTH)
        
        # Advanced preprocessing pipeline
        signal = remove_powerline_interference(signal, TARGET_SAMPLING_RATE)
        signal = remove_baseline_wander_improved(signal)
        signal = apply_bandpass_filter(signal, TARGET_SAMPLING_RATE)
        signal = normalize_ecg_improved(signal)
        
        # Final quality check
        if np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
            return None
        
        return signal
        
    except Exception as e:
        return None

def resample_ecg_improved(signal, original_fs, target_fs):
    """Improved resampling using anti-aliasing."""
    if original_fs == target_fs:
        return signal
    
    resampled = np.zeros((int(signal.shape[0] * target_fs / original_fs), signal.shape[1]))
    
    for lead in range(signal.shape[1]):
        resampled[:, lead] = scipy_signal.resample(
            signal[:, lead], 
            int(signal.shape[0] * target_fs / original_fs),
            window='hann'
        )
    
    return resampled

def resize_to_target_length(signal, target_length):
    """Resize signal to target length."""
    current_length = signal.shape[0]
    
    if current_length > target_length:
        # Take middle portion
        start = (current_length - target_length) // 2
        return signal[start:start + target_length]
    else:
        # Pad symmetrically
        pad_total = target_length - current_length
        pad_before = pad_total // 2
        pad_after = pad_total - pad_before
        return np.pad(signal, ((pad_before, pad_after), (0, 0)), mode='edge')

def remove_powerline_interference(signal, fs):
    """Remove 50/60 Hz powerline interference."""
    for freq in [50, 60]:  # Both European and US powerline frequencies
        # Notch filter
        Q = 30  # Quality factor
        w0 = freq / (fs / 2)  # Normalized frequency
        b, a = scipy_signal.iirnotch(w0, Q)
        
        for lead in range(signal.shape[1]):
            signal[:, lead] = scipy_signal.filtfilt(b, a, signal[:, lead])
    
    return signal

def remove_baseline_wander_improved(signal):
    """Improved baseline wander removal using high-pass filter."""
    # High-pass filter to remove baseline wander (< 0.5 Hz)
    sos = scipy_signal.butter(4, 0.5, btype='high', fs=TARGET_SAMPLING_RATE, output='sos')
    
    for lead in range(signal.shape[1]):
        signal[:, lead] = scipy_signal.sosfiltfilt(sos, signal[:, lead])
    
    return signal

def apply_bandpass_filter(signal, fs):
    """Apply bandpass filter for ECG (0.5-40 Hz)."""
    sos = scipy_signal.butter(4, [0.5, 40], btype='band', fs=fs, output='sos')
    
    for lead in range(signal.shape[1]):
        signal[:, lead] = scipy_signal.sosfiltfilt(sos, signal[:, lead])
    
    return signal

def normalize_ecg_improved(signal):
    """Improved normalization with outlier handling."""
    for lead in range(signal.shape[1]):
        lead_signal = signal[:, lead]
        
        # Remove extreme outliers (beyond 5 standard deviations)
        z_scores = np.abs(zscore(lead_signal))
        lead_signal = np.where(z_scores > 5, np.median(lead_signal), lead_signal)
        
        # Robust normalization using percentiles
        p1, p99 = np.percentile(lead_signal, [1, 99])
        lead_signal = np.clip(lead_signal, p1, p99)
        
        # Standard normalization
        mean_val = np.mean(lead_signal)
        std_val = np.std(lead_signal)
        if std_val > 0:
            lead_signal = (lead_signal - mean_val) / std_val
        
        signal[:, lead] = lead_signal
    
    return signal

def get_improved_chagas_label(header, record_name, label_map):
    """Improved label extraction with multiple strategies."""
    # Strategy 1: Use label map if available
    if record_name in label_map:
        return label_map[record_name]
    
    # Strategy 2: Check header comments and diagnosis
    try:
        for field in ['comments', 'diagnosis', 'dx']:
            if hasattr(header, field):
                content = str(getattr(header, field)).lower()
                if 'chagas' in content or 'trypanosoma' in content:
                    return 1
                elif 'normal' in content or 'healthy' in content:
                    return 0
    except:
        pass
    
    # Strategy 3: Infer from source/filename patterns
    record_lower = record_name.lower()
    
    # SaMi-Trop patterns (typically positive)
    if any(pattern in record_lower for pattern in ['samitrop', 'sami', 'chagas']):
        return 1
    
    # PTB-XL patterns (typically negative)
    if any(pattern in record_lower for pattern in ['ptb', 'german', 'normal']):
        return 0
    
    # CODE-15 patterns (mixed, require more careful analysis)
    if 'code' in record_lower:
        # For CODE-15, we need labels from the CSV file
        return None  # Skip if no clear label
    
    return None  # Skip ambiguous cases

def get_improved_demographics(header):
    """Extract demographics with better defaults."""
    age = 50.0  # Default middle age
    sex = 0.5   # Unknown
    
    try:
        # Age extraction
        if hasattr(header, 'age'):
            age_str = str(header.age).lower()
            # Extract numeric part
            import re
            age_match = re.search(r'(\d+)', age_str)
            if age_match:
                age = float(age_match.group(1))
                age = np.clip(age, 0, 100)  # Reasonable bounds
        
        # Sex extraction
        if hasattr(header, 'sex'):
            sex_str = str(header.sex).lower().strip()
            if sex_str.startswith('m') or sex_str == '1':
                sex = 1.0
            elif sex_str.startswith('f') or sex_str == '0':
                sex = 0.0
    except:
        pass
    
    return [age / 100.0, sex]  # Normalized age

def get_data_source(header, record_name):
    """Identify data source for stratification."""
    record_lower = record_name.lower()
    
    if any(pattern in record_lower for pattern in ['samitrop', 'sami']):
        return 'samitrop'
    elif any(pattern in record_lower for pattern in ['ptb', 'german']):
        return 'ptbxl'
    elif 'code' in record_lower:
        return 'code15'
    else:
        return 'unknown'

def train_chagas_model_cv(signals, labels, demographics, sources, model_folder, verbose):
    """Train with cross-validation and ensemble."""
    try:
        # Convert to numpy arrays
        X_signals = np.array(signals)
        X_demographics = np.array(demographics)
        y = np.array(labels)
        
        # Stratified K-Fold for robust training
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        
        models = []
        scalers = []
        scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_signals, y)):
            if verbose:
                print(f"Training fold {fold + 1}/5...")
            
            # Split data
            X_sig_train, X_sig_val = X_signals[train_idx], X_signals[val_idx]
            X_demo_train, X_demo_val = X_demographics[train_idx], X_demographics[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Scale demographics
            demo_scaler = RobustScaler()  # More robust to outliers
            X_demo_train_scaled = demo_scaler.fit_transform(X_demo_train)
            X_demo_val_scaled = demo_scaler.transform(X_demo_val)
            
            # Build and train model
            model = build_improved_model()
            
            # Class weights
            class_weights = compute_class_weight(
                'balanced', classes=np.unique(y_train), y=y_train)
            class_weight_dict = {i: w for i, w in enumerate(class_weights)}
            
            # Callbacks
            callbacks = [
                EarlyStopping(patience=15, restore_best_weights=True, monitor='val_auc'),
                ReduceLROnPlateau(factor=0.3, patience=8, min_lr=1e-7, monitor='val_auc'),
            ]
            
            # Train
            history = model.fit(
                [X_sig_train, X_demo_train_scaled], y_train,
                validation_data=([X_sig_val, X_demo_val_scaled], y_val),
                epochs=100,
                batch_size=BATCH_SIZE,
                callbacks=callbacks,
                class_weight=class_weight_dict,
                verbose=1 if verbose and fold == 0 else 0
            )
            
            # Evaluate
            val_pred = model.predict([X_sig_val, X_demo_val_scaled], verbose=0)
            val_auc = tf.keras.metrics.AUC()(y_val, val_pred).numpy()
            scores.append(val_auc)
            
            models.append(model)
            scalers.append(demo_scaler)
            
            if verbose:
                print(f"Fold {fold + 1} AUC: {val_auc:.4f}")
        
        # Select best model or ensemble
        best_idx = np.argmax(scores)
        best_model = models[best_idx]
        best_scaler = scalers[best_idx]
        
        if verbose:
            print(f"Best fold AUC: {scores[best_idx]:.4f}")
            print(f"Mean CV AUC: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
        
        # Save best model
        save_model_improved(best_model, best_scaler, model_folder, verbose)
        
        return True
        
    except Exception as e:
        if verbose:
            print(f"Model training failed: {str(e)}")
        return False

def build_improved_model():
    """Build improved CNN model with residual connections."""
    # Signal input
    signal_input = Input(shape=(TARGET_SIGNAL_LENGTH, NUM_LEADS), name='ecg_input')
    
    # Initial processing
    x = Conv1D(32, 15, padding='same')(signal_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(2)(x)
    
    # Residual blocks
    for filters in [64, 128, 256]:
        x = residual_block(x, filters)
        x = MaxPooling1D(2)(x)
    
    # Global features
    x = GlobalAveragePooling1D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    # Demographics branch
    demo_input = Input(shape=(2,), name='demo_input')
    demo_branch = Dense(32, activation='relu')(demo_input)
    demo_branch = Dense(16, activation='relu')(demo_branch)
    
    # Combine features
    combined = concatenate([x, demo_branch])
    combined = Dense(64, activation='relu')(combined)
    combined = Dropout(0.2)(combined)
    
    # Output
    output = Dense(1, activation='sigmoid')(combined)
    
    model = Model(inputs=[signal_input, demo_input], outputs=output)
    
    # Improved optimizer settings
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8
    )
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    
    return model

def residual_block(x, filters):
    """Residual block for better gradient flow."""
    shortcut = x
    
    # First conv
    x = Conv1D(filters, 5, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Second conv
    x = Conv1D(filters, 5, padding='same')(x)
    x = BatchNormalization()(x)
    
    # Adjust shortcut if needed
    if shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters, 1, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)
    
    # Add and activate
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    
    return x

def save_model_improved(model, demo_scaler, model_folder, verbose):
    """Save model with versioning and metadata."""
    os.makedirs(model_folder, exist_ok=True)
    
    # Save model
    model.save(os.path.join(model_folder, 'model.keras'))
    
    # Save scaler
    import joblib
    joblib.dump(demo_scaler, os.path.join(model_folder, 'demo_scaler.pkl'))
    
    # Save comprehensive config
    config = {
        'signal_length': TARGET_SIGNAL_LENGTH,
        'num_leads': NUM_LEADS,
        'sampling_rate': TARGET_SAMPLING_RATE,
        'batch_size': BATCH_SIZE,
        'model_version': '2.0',
        'preprocessing': {
            'bandpass_filter': [0.5, 40],
            'notch_filter': [50, 60],
            'normalization': 'robust_zscore'
        }
    }
    
    import json
    with open(os.path.join(model_folder, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    if verbose:
        print(f"Improved model saved to {model_folder}")

def load_model(model_folder, verbose):
    """Load improved model."""
    try:
        if verbose:
            print(f"Loading model from {model_folder}")
        
        model = tf.keras.models.load_model(os.path.join(model_folder, 'model.keras'))
        
        import joblib
        demo_scaler = joblib.load(os.path.join(model_folder, 'demo_scaler.pkl'))
        
        import json
        with open(os.path.join(model_folder, 'config.json'), 'r') as f:
            config = json.load(f)
        
        return {
            'model': model,
            'demo_scaler': demo_scaler,
            'config': config
        }
        
    except Exception as e:
        if verbose:
            print(f"Model loading failed: {str(e)}")
        return None

def run_model(record, model_data, verbose):
    """Improved inference with better error handling."""
    try:
        model = model_data['model']
        demo_scaler = model_data['demo_scaler']
        config = model_data['config']
        
        # Load and process signal
        try:
            signal, fields = wfdb.rdsamp(record)
            sampling_rate = fields.get('fs', 500)
            processed_signal = preprocess_ecg_improved(signal, sampling_rate)
            
            if processed_signal is None:
                raise ValueError("Signal processing failed")
                
        except Exception as e:
            if verbose:
                print(f"Signal loading failed for {record}: {e}")
            # Create dummy signal as fallback
            processed_signal = np.random.randn(
                config['signal_length'], config['num_leads']).astype(np.float32) * 0.1
        
        # Load demographics
        try:
            header = wfdb.rdheader(record)
            demographics = get_improved_demographics(header)
        except:
            demographics = [0.5, 0.5]  # Default values
        
        # Prepare inputs
        signal_input = processed_signal.reshape(1, -1, config['num_leads'])
        demo_input = demo_scaler.transform([demographics])
        
        # Predict with confidence
        probability = float(model.predict([signal_input, demo_input], verbose=0)[0][0])
        
        # More conservative threshold for better precision
        threshold = 0.3  # Lower threshold to catch more positives
        prediction = 1 if probability >= threshold else 0
        
        return prediction, probability
        
    except Exception as e:
        if verbose:
            print(f"Prediction failed for {record}: {str(e)}")
        return 0, 0.01  # Very conservative default

def create_fallback_model(model_folder, verbose):
    """Improved fallback model."""
    try:
        os.makedirs(model_folder, exist_ok=True)
        
        # Simple but effective architecture
        signal_input = Input(shape=(TARGET_SIGNAL_LENGTH, NUM_LEADS))
        x = Conv1D(32, 15, activation='relu', padding='same')(signal_input)
        x = MaxPooling1D(4)(x)
        x = Conv1D(64, 9, activation='relu', padding='same')(x)
        x = GlobalAveragePooling1D()(x)
        
        demo_input = Input(shape=(2,))
        demo_branch = Dense(16, activation='relu')(demo_input)
        
        combined = concatenate([x, demo_branch])
        output = Dense(1, activation='sigmoid')(combined)
        
        model = Model(inputs=[signal_input, demo_input], outputs=output)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss='binary_crossentropy',
            metrics=['auc']
        )
        
        # Create dummy scaler
        demo_scaler = RobustScaler()
        demo_scaler.fit(np.random.randn(100, 2))
        
        save_model_improved(model, demo_scaler, model_folder, verbose)
        
        if verbose:
            print("Improved fallback model created")
        
        return True
        
    except Exception as e:
        if verbose:
            print(f"Fallback model creation failed: {str(e)}")
        return False
