#!/usr/bin/env python

# Simplified and focused Chagas disease detection model
# Clean implementation without baseline/dummy complexity

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
warnings.filterwarnings('ignore')

from helper_code import *

# Constants
TARGET_SAMPLING_RATE = 500
TARGET_SIGNAL_LENGTH = 5000  # 10 seconds at 500Hz
BATCH_SIZE = 16
NUM_LEADS = 12

def train_model(data_folder, model_folder, verbose):
    """
    Main training function
    """
    if verbose:
        print("Training Chagas detection model...")
    
    os.makedirs(model_folder, exist_ok=True)
    
    # Load data
    signals, labels, demographics = load_data(data_folder, verbose)
    
    if len(signals) == 0:
        raise ValueError("No data loaded. Check your data folder and format.")
    
    return train_enhanced_model(signals, labels, demographics, model_folder, verbose)

def load_data(data_folder, verbose):
    """
    Load data from available formats
    """
    signals = []
    labels = []
    demographics = []
    
    # Try HDF5 first
    hdf5_path = os.path.join(data_folder, 'exams.hdf5')
    if os.path.exists(hdf5_path):
        if verbose:
            print("Loading from HDF5...")
        s, l, d = load_from_hdf5(data_folder, verbose)
        signals.extend(s)
        labels.extend(l)
        demographics.extend(d)
    
    # Try WFDB records
    if verbose:
        print("Loading from WFDB records...")
    s, l, d = load_from_wfdb(data_folder, verbose)
    signals.extend(s)
    labels.extend(l)
    demographics.extend(d)
    
    if verbose:
        print(f"Total loaded: {len(signals)} samples")
        if len(labels) > 0:
            pos_rate = np.mean(labels) * 100
            print(f"Positive rate: {pos_rate:.1f}%")
    
    return signals, labels, demographics

def load_from_hdf5(data_folder, verbose):
    """
    Load from HDF5 format
    """
    signals = []
    labels = []
    demographics = []
    
    try:
        exams_path = os.path.join(data_folder, 'exams.csv')
        if not os.path.exists(exams_path):
            return signals, labels, demographics
        
        exams_df = pd.read_csv(exams_path)
        
        # Load Chagas labels
        chagas_labels = load_chagas_labels(data_folder, verbose)
        
        # Load HDF5 signals
        hdf5_path = os.path.join(data_folder, 'exams.hdf5')
        with h5py.File(hdf5_path, 'r') as hdf:
            # Get main dataset
            if 'tracings' in hdf:
                dataset = hdf['tracings']
            elif 'exams' in hdf:
                dataset = hdf['exams']
            else:
                dataset = hdf[list(hdf.keys())[0]]
            
            for idx, row in exams_df.iterrows():
                try:
                    exam_id = row.get('exam_id', row.get('id', idx))
                    
                    # Get label
                    label = get_label_for_exam(exam_id, row, chagas_labels)
                    if label is None:
                        continue
                    
                    # Extract signal
                    if hasattr(dataset, 'shape') and len(dataset.shape) == 3:
                        signal = dataset[idx]
                    elif str(exam_id) in dataset:
                        signal = dataset[str(exam_id)][:]
                    else:
                        continue
                    
                    # Process signal
                    processed_signal = process_signal(signal)
                    if processed_signal is None:
                        continue
                    
                    # Extract demographics
                    demo = extract_demographics(row)
                    
                    signals.append(processed_signal)
                    labels.append(label)
                    demographics.append(demo)
                    
                    if verbose and len(signals) % 500 == 0:
                        print(f"Processed {len(signals)} HDF5 samples")
                
                except Exception as e:
                    continue
    
    except Exception as e:
        if verbose:
            print(f"HDF5 loading error: {e}")
    
    return signals, labels, demographics

def load_chagas_labels(data_folder, verbose):
    """
    Load Chagas labels from CSV files
    """
    chagas_labels = {}
    
    label_files = ['samitrop_chagas_labels.csv', 'code15_chagas_labels.csv', 'chagas_labels.csv']
    
    for label_file in label_files:
        label_path = os.path.join(data_folder, label_file)
        if os.path.exists(label_path):
            try:
                label_df = pd.read_csv(label_path)
                if verbose:
                    print(f"Found label file: {label_file}")
                
                for _, row in label_df.iterrows():
                    exam_id = row.get('exam_id', row.get('id', None))
                    chagas = row.get('chagas', row.get('label', None))
                    
                    if exam_id is not None and chagas is not None:
                        if isinstance(chagas, str):
                            chagas_binary = 1 if chagas.lower() in ['true', 'positive', 'yes', '1'] else 0
                        else:
                            chagas_binary = int(float(chagas))
                        chagas_labels[exam_id] = chagas_binary
                
                if verbose and chagas_labels:
                    pos_count = sum(chagas_labels.values())
                    print(f"Loaded {len(chagas_labels)} labels, {pos_count} positive")
                    break
                    
            except Exception as e:
                if verbose:
                    print(f"Error loading {label_file}: {e}")
                continue
    
    return chagas_labels

def get_label_for_exam(exam_id, row, chagas_labels):
    """
    Get label for exam with source-based inference
    """
    # Direct label lookup
    if exam_id in chagas_labels:
        return chagas_labels[exam_id]
    
    # Source-based inference
    source = str(row.get('source', '')).lower()
    
    # SaMi-Trop dataset - all Chagas positive
    if 'samitrop' in source or 'sami-trop' in source:
        return 1
    
    # PTB-XL, Chapman, etc. - typically Chagas negative
    if any(keyword in source for keyword in ['ptb', 'chapman', 'georgia']):
        return 0
    
    return None

def load_from_wfdb(data_folder, verbose):
    """
    Load from WFDB format
    """
    signals = []
    labels = []
    demographics = []
    
    try:
        records = find_records(data_folder)
        if verbose:
            print(f"Found {len(records)} WFDB records")
        
        for record_name in records:
            try:
                record_path = os.path.join(data_folder, record_name)
                
                # Load signal and header
                signal, fields = load_signals(record_path)
                header = load_header(record_path)
                
                # Process signal
                processed_signal = process_signal(signal)
                if processed_signal is None:
                    continue
                
                # Extract label
                label = load_label(record_path)
                if label is None:
                    continue
                
                # Extract demographics
                demo = extract_demographics_wfdb(header)
                
                signals.append(processed_signal)
                labels.append(int(label))
                demographics.append(demo)
                
                if verbose and len(signals) % 100 == 0:
                    print(f"Processed {len(signals)} WFDB records")
            
            except Exception as e:
                continue
    
    except Exception as e:
        if verbose:
            print(f"WFDB loading error: {e}")
    
    return signals, labels, demographics

def process_signal(signal):
    """
    Process ECG signal
    """
    try:
        signal = np.array(signal, dtype=np.float32)
        
        # Handle shape
        if len(signal.shape) == 1:
            signal = signal.reshape(-1, 1)
        elif signal.shape[0] < signal.shape[1] and signal.shape[0] <= 12:
            signal = signal.T
        
        # Ensure 12 leads
        if signal.shape[1] > 12:
            signal = signal[:, :12]
        elif signal.shape[1] < 12:
            # Repeat last lead to fill 12 leads
            padding_needed = 12 - signal.shape[1]
            last_lead = signal[:, -1:] if signal.shape[1] > 0 else np.zeros((signal.shape[0], 1))
            padding = np.repeat(last_lead, padding_needed, axis=1)
            signal = np.hstack([signal, padding])
        
        # Resample to target length
        signal = resample_signal(signal, TARGET_SIGNAL_LENGTH)
        
        # Normalize
        signal = normalize_signal(signal)
        
        return signal.astype(np.float32)
    
    except Exception as e:
        return None

def resample_signal(signal, target_length):
    """
    Resample signal to target length
    """
    current_length = signal.shape[0]
    
    if current_length == target_length:
        return signal
    
    # Linear interpolation
    x_old = np.linspace(0, 1, current_length)
    x_new = np.linspace(0, 1, target_length)
    
    resampled = np.zeros((target_length, signal.shape[1]))
    for i in range(signal.shape[1]):
        resampled[:, i] = np.interp(x_new, x_old, signal[:, i])
    
    return resampled

def normalize_signal(signal):
    """
    Normalize signal using robust statistics
    """
    for i in range(signal.shape[1]):
        # Remove DC
        signal[:, i] = signal[:, i] - np.median(signal[:, i])
        
        # Robust scaling using MAD
        mad = np.median(np.abs(signal[:, i] - np.median(signal[:, i])))
        
        if mad > 1e-6:
            signal[:, i] = signal[:, i] / (mad * 1.4826)
        
        # Clip extreme values
        signal[:, i] = np.clip(signal[:, i], -5, 5)
    
    return signal

def extract_demographics(row):
    """
    Extract demographic features
    """
    # Age
    age = row.get('age', 50.0)
    if pd.isna(age):
        age = 50.0
    age_norm = np.clip(float(age) / 100.0, 0.0, 1.0)
    
    # Sex
    sex = row.get('sex', row.get('is_male', 0))
    if pd.isna(sex):
        sex_male = 0.5
    else:
        if isinstance(sex, str):
            sex_male = 1.0 if sex.lower().startswith('m') else 0.0
        else:
            sex_male = float(sex)
    
    return np.array([age_norm, sex_male])

def extract_demographics_wfdb(header):
    """
    Extract demographics from WFDB header
    """
    age = get_age(header)
    sex = get_sex(header)
    
    age_norm = 0.5
    if age is not None:
        age_norm = np.clip(float(age) / 100.0, 0.0, 1.0)
    
    sex_male = 0.5
    if sex is not None:
        sex_male = 1.0 if sex.lower().startswith('m') else 0.0
    
    return np.array([age_norm, sex_male])

def extract_ecg_features(signal):
    """
    Extract simple ECG features
    """
    features = []
    
    # Use lead II or lead I
    lead_signal = signal[:, 1] if signal.shape[1] > 1 else signal[:, 0]
    
    # Simple heart rate estimation
    try:
        # Find peaks (simple approach)
        threshold = np.std(lead_signal) * 0.5
        peaks = []
        min_distance = TARGET_SAMPLING_RATE // 3  # 200ms
        
        for i in range(min_distance, len(lead_signal) - min_distance):
            if (lead_signal[i] > threshold and 
                lead_signal[i] > lead_signal[i-1] and 
                lead_signal[i] > lead_signal[i+1]):
                if not peaks or i - peaks[-1] >= min_distance:
                    peaks.append(i)
        
        if len(peaks) >= 2:
            rr_intervals = np.diff(peaks) / TARGET_SAMPLING_RATE
            mean_hr = 60.0 / np.mean(rr_intervals)
            hrv = np.std(rr_intervals) * 1000
        else:
            mean_hr = 70.0
            hrv = 20.0
            
        features.extend([mean_hr, hrv])
        
    except:
        features.extend([70.0, 20.0])
    
    # Signal statistics for each lead
    for i in range(min(3, signal.shape[1])):  # First 3 leads
        lead = signal[:, i]
        features.extend([
            np.mean(lead),
            np.std(lead),
            np.max(lead) - np.min(lead)  # Peak-to-peak
        ])
    
    # Pad to fixed length
    while len(features) < 11:
        features.append(0.0)
    
    return np.array(features[:11])

def build_model(signal_shape, demo_features, ecg_features):
    """
    Build enhanced CNN model with attention
    """
    # Signal input branch
    signal_input = Input(shape=signal_shape, name='signal_input')
    
    # CNN with residual connections
    x = Conv1D(64, 15, strides=2, padding='same')(signal_input)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    # Residual blocks
    for filters in [64, 128, 256]:
        residual = x
        
        x = Conv1D(filters, 7, padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        
        x = Conv1D(filters, 7, padding='same')(x)
        x = BatchNormalization()(x)
        
        # Adjust residual if needed
        if residual.shape[-1] != filters:
            residual = Conv1D(filters, 1, padding='same')(residual)
            residual = BatchNormalization()(residual)
        
        x = Add()([x, residual])
        x = ReLU()(x)
        
        # Downsample
        if filters > 64:
            x = Conv1D(filters, 3, strides=2, padding='same')(x)
    
    # Attention mechanism
    x = MultiHeadAttention(num_heads=8, key_dim=32)(x, x)
    x = LayerNormalization()(x)
    
    # Global pooling
    x = GlobalAveragePooling1D()(x)
    signal_features = Dense(128, activation='relu')(x)
    signal_features = Dropout(0.3)(signal_features)
    
    # Demographics branch
    demo_input = Input(shape=(demo_features,), name='demo_input')
    demo_branch = Dense(16, activation='relu')(demo_input)
    demo_branch = Dropout(0.2)(demo_branch)
    
    # ECG features branch
    ecg_input = Input(shape=(ecg_features,), name='ecg_input')
    ecg_branch = Dense(32, activation='relu')(ecg_input)
    ecg_branch = Dropout(0.2)(ecg_branch)
    
    # Fusion
    combined = concatenate([signal_features, demo_branch, ecg_branch])
    combined = Dense(64, activation='relu')(combined)
    combined = Dropout(0.4)(combined)
    combined = Dense(32, activation='relu')(combined)
    combined = Dropout(0.3)(combined)
    
    # Output
    output = Dense(1, activation='sigmoid')(combined)
    
    model = Model(inputs=[signal_input, demo_input, ecg_input], outputs=output)
    return model

def train_enhanced_model(signals, labels, demographics, model_folder, verbose):
    """
    Train the model
    """
    if verbose:
        print(f"Training on {len(signals)} samples")
    
    # Convert to arrays
    X_signal = np.array(signals, dtype=np.float32)
    X_demo = np.array(demographics, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    
    # Extract ECG features
    if verbose:
        print("Extracting ECG features...")
    
    ecg_features_list = []
    for signal in signals:
        ecg_feat = extract_ecg_features(signal)
        ecg_features_list.append(ecg_feat)
    
    X_ecg = np.array(ecg_features_list, dtype=np.float32)
    
    if verbose:
        print(f"Signal shape: {X_signal.shape}")
        print(f"Demographics shape: {X_demo.shape}")
        print(f"ECG features shape: {X_ecg.shape}")
        unique, counts = np.unique(y, return_counts=True)
        print(f"Label distribution: {dict(zip(unique, counts))}")
    
    # Split data
    X_sig_train, X_sig_test, X_demo_train, X_demo_test, X_ecg_train, X_ecg_test, y_train, y_test = train_test_split(
        X_signal, X_demo, X_ecg, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    demo_scaler = RobustScaler()
    ecg_scaler = RobustScaler()
    
    X_demo_train_scaled = demo_scaler.fit_transform(X_demo_train)
    X_demo_test_scaled = demo_scaler.transform(X_demo_test)
    
    X_ecg_train_scaled = ecg_scaler.fit_transform(X_ecg_train)
    X_ecg_test_scaled = ecg_scaler.transform(X_ecg_test)
    
    # Build model
    model = build_model(X_signal.shape[1:], X_demo.shape[1], X_ecg.shape[1])
    
    if verbose:
        print("Model summary:")
        model.summary()
    
    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    # Class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6)
    ]
    
    # Train
    history = model.fit(
        [X_sig_train, X_demo_train_scaled, X_ecg_train_scaled], y_train,
        validation_data=([X_sig_test, X_demo_test_scaled, X_ecg_test_scaled], y_test),
        epochs=100,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1 if verbose else 0
    )
    
    # Evaluate
    if verbose:
        y_pred_prob = model.predict([X_sig_test, X_demo_test_scaled, X_ecg_test_scaled])
        val_auc = roc_auc_score(y_test, y_pred_prob)
        print(f"Validation AUC: {val_auc:.3f}")
    
    # Save model
    save_model_files(model_folder, model, demo_scaler, ecg_scaler, verbose)
    
    return True

def save_model_files(model_folder, model, demo_scaler, ecg_scaler, verbose):
    """
    Save model and scalers
    """
    # Save model
    model.save(os.path.join(model_folder, 'model.keras'))
    
    # Save scalers
    import joblib
    joblib.dump(demo_scaler, os.path.join(model_folder, 'demo_scaler.pkl'))
    joblib.dump(ecg_scaler, os.path.join(model_folder, 'ecg_scaler.pkl'))
    
    # Save config
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
    """
    Load trained model
    """
    # Load model
    model = tf.keras.models.load_model(os.path.join(model_folder, 'model.keras'))
    
    # Load scalers
    import joblib
    demo_scaler = joblib.load(os.path.join(model_folder, 'demo_scaler.pkl'))
    ecg_scaler = joblib.load(os.path.join(model_folder, 'ecg_scaler.pkl'))
    
    # Load config
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
    """
    Run model on single record
    """
    try:
        model = model_data['model']
        demo_scaler = model_data['demo_scaler']
        ecg_scaler = model_data['ecg_scaler']
        config = model_data['config']
        
        # Load and process signal
        signal, fields = load_signals(record)
        processed_signal = process_signal(signal)
        
        if processed_signal is None:
            # Use default signal
            processed_signal = np.random.randn(config['signal_length'], config['num_leads']).astype(np.float32)
        
        # Extract ECG features
        ecg_features = extract_ecg_features(processed_signal)
        
        # Extract demographics
        header = load_header(record)
        demographics = extract_demographics_wfdb(header)
        
        # Prepare inputs
        signal_input = processed_signal.reshape(1, config['signal_length'], config['num_leads'])
        demo_input = demo_scaler.transform(demographics.reshape(1, -1))
        ecg_input = ecg_scaler.transform(ecg_features.reshape(1, -1))
        
        # Predict
        probability = float(model.predict([signal_input, demo_input, ecg_input], verbose=0)[0][0])
        
        # Binary prediction with optimized threshold
        binary_prediction = 1 if probability >= 0.3 else 0
        
        return binary_prediction, probability
        
    except Exception as e:
        if verbose:
            print(f"Error in run_model: {e}")
        return 0, 0.1
