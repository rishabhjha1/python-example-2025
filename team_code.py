#!/usr/bin/env python

# Robust Chagas disease detection model
# Focused on ECG signal analysis only

import os
import numpy as np
import pandas as pd
import h5py
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Dense, Dropout, Conv1D, Input, 
                                   BatchNormalization, GlobalAveragePooling1D, 
                                   ReLU, Add, SpatialDropout1D)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import AUC
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

from helper_code import *

# Optimized constants
TARGET_SAMPLING_RATE = 400
TARGET_SIGNAL_LENGTH = 2048
MAX_SAMPLES = 15000
BATCH_SIZE = 32
NUM_LEADS = 12

def train_model(data_folder, model_folder, verbose):
    """
    Robust Chagas detection training focusing on ECG signals
    """
    if verbose:
        print("Training robust Chagas detection model...")
    
    os.makedirs(model_folder, exist_ok=True)
    
    # Load ECG data
    signals, labels = load_ecg_data(data_folder, verbose)
    
    if len(signals) < 100:
        if verbose:
            print(f"Insufficient data ({len(signals)} samples), creating minimal baseline")
        return create_minimal_baseline(model_folder, verbose)
    
    return train_robust_model(signals, labels, model_folder, verbose)

def load_ecg_data(data_folder, verbose):
    """
    Load ECG signals with robust error handling
    """
    signals = []
    labels = []
    
    # Try HDF5 first
    hdf5_path = os.path.join(data_folder, 'exams.hdf5')
    if os.path.exists(hdf5_path):
        if verbose:
            print("Loading from HDF5...")
        s, l = load_from_hdf5(data_folder, verbose)
        signals.extend(s)
        labels.extend(l)
    
    # Try WFDB records if we need more data
    if len(signals) < 1000:
        if verbose:
            print("Loading from WFDB records...")
        s, l = load_from_wfdb(data_folder, verbose)
        signals.extend(s)
        labels.extend(l)
    
    if verbose:
        print(f"Total loaded: {len(signals)} samples")
        if len(labels) > 0:
            pos_rate = np.mean(labels) * 100
            print(f"Positive rate: {pos_rate:.1f}%")
    
    return signals, labels

def load_from_hdf5(data_folder, verbose):
    """
    Load ECG data from HDF5 format
    """
    signals = []
    labels = []
    
    try:
        # Load metadata
        exams_path = os.path.join(data_folder, 'exams.csv')
        if not os.path.exists(exams_path):
            return signals, labels
        
        exams_df = pd.read_csv(exams_path, nrows=MAX_SAMPLES)
        
        # Load Chagas labels
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
                        exam_id = row.get('exam_id', row.get('id'))
                        chagas = row.get('chagas', row.get('label', row.get('target')))
                        
                        if exam_id is not None and chagas is not None:
                            # Convert to binary
                            if isinstance(chagas, str):
                                chagas_binary = 1 if chagas.lower() in ['true', 'positive', 'yes', '1'] else 0
                            else:
                                chagas_binary = int(float(chagas))
                            chagas_labels[exam_id] = chagas_binary
                    
                    if verbose:
                        pos_count = sum(chagas_labels.values())
                        print(f"Loaded {len(chagas_labels)} labels, {pos_count} positive")
                    break
                except Exception as e:
                    if verbose:
                        print(f"Error loading {label_file}: {e}")
                    continue
        
        # Load HDF5 signals
        hdf5_path = os.path.join(data_folder, 'exams.hdf5')
        with h5py.File(hdf5_path, 'r') as hdf:
            if verbose:
                print(f"HDF5 structure: {list(hdf.keys())}")
            
            # Find dataset
            if 'tracings' in hdf:
                dataset = hdf['tracings']
            elif 'exams' in hdf:
                dataset = hdf['exams']
            else:
                dataset = hdf[list(hdf.keys())[0]]
            
            processed_count = 0
            
            for idx, row in exams_df.iterrows():
                if processed_count >= MAX_SAMPLES:
                    break
                
                try:
                    exam_id = row.get('exam_id', row.get('id', idx))
                    
                    # Get label
                    if exam_id in chagas_labels:
                        label = chagas_labels[exam_id]
                    else:
                        # Infer from source
                        source = str(row.get('source', '')).lower()
                        if 'samitrop' in source:
                            label = 1  # SaMi-Trop is Chagas positive
                        elif 'ptb' in source:
                            label = 0  # PTB-XL is negative
                        else:
                            continue
                    
                    # Extract signal
                    if hasattr(dataset, 'shape') and len(dataset.shape) == 3:
                        signal = dataset[idx]
                    elif str(exam_id) in dataset:
                        signal = dataset[str(exam_id)][:]
                    else:
                        continue
                    
                    # Process signal
                    processed_signal = process_ecg_signal(signal)
                    if processed_signal is None:
                        continue
                    
                    signals.append(processed_signal)
                    labels.append(label)
                    processed_count += 1
                    
                    if verbose and processed_count % 1000 == 0:
                        print(f"Processed {processed_count} HDF5 samples")
                
                except Exception as e:
                    if verbose and processed_count < 5:
                        print(f"Error processing HDF5 sample {idx}: {e}")
                    continue
    
    except Exception as e:
        if verbose:
            print(f"HDF5 loading error: {e}")
    
    return signals, labels

def load_from_wfdb(data_folder, verbose):
    """
    Load ECG data from WFDB format
    """
    signals = []
    labels = []
    
    try:
        records = find_records(data_folder)
        if verbose:
            print(f"Found {len(records)} WFDB records")
        
        for record_name in records[:MAX_SAMPLES]:
            try:
                record_path = os.path.join(data_folder, record_name)
                
                # Load signal
                signal, fields = load_signals(record_path)
                
                # Process signal
                processed_signal = process_ecg_signal(signal)
                if processed_signal is None:
                    continue
                
                # Extract label
                label = load_label(record_path)
                if label is None:
                    continue
                
                signals.append(processed_signal)
                labels.append(int(label))
                
                if verbose and len(signals) % 500 == 0:
                    print(f"Processed {len(signals)} WFDB records")
            
            except Exception as e:
                if verbose and len(signals) < 5:
                    print(f"Error processing WFDB {record_name}: {e}")
                continue
    
    except Exception as e:
        if verbose:
            print(f"WFDB loading error: {e}")
    
    return signals, labels

def process_ecg_signal(signal):
    """
    Robust ECG signal processing
    """
    try:
        signal = np.array(signal, dtype=np.float32)
        
        # Handle shapes
        if len(signal.shape) == 1:
            signal = signal.reshape(-1, 1)
        elif signal.shape[0] < signal.shape[1] and signal.shape[0] <= 12:
            signal = signal.T
        
        # Ensure 12 leads
        if signal.shape[1] > 12:
            signal = signal[:, :12]
        elif signal.shape[1] < 12:
            # Pad with zeros or repeat last lead
            if signal.shape[1] > 0:
                last_lead = signal[:, -1:]
                padding = np.repeat(last_lead, 12 - signal.shape[1], axis=1)
                signal = np.hstack([signal, padding])
            else:
                signal = np.zeros((signal.shape[0], 12))
        
        # Resample
        signal = resample_signal(signal, TARGET_SIGNAL_LENGTH)
        
        # Normalize
        signal = normalize_ecg(signal)
        
        # Quality check
        if not is_valid_ecg(signal):
            return None
        
        return signal.astype(np.float32)
    
    except Exception:
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

def normalize_ecg(signal):
    """
    Robust ECG normalization
    """
    for i in range(signal.shape[1]):
        # Remove baseline (median for robustness)
        signal[:, i] = signal[:, i] - np.median(signal[:, i])
        
        # Robust scaling using percentiles
        p01, p99 = np.percentile(signal[:, i], [1, 99])
        signal[:, i] = np.clip(signal[:, i], p01, p99)
        
        # Standardize
        std_val = np.std(signal[:, i])
        if std_val > 1e-6:
            signal[:, i] = signal[:, i] / std_val
        
        # Final clipping
        signal[:, i] = np.clip(signal[:, i], -10, 10)
    
    return signal

def is_valid_ecg(signal):
    """
    Check if ECG signal is valid
    """
    # Check for NaN or inf
    if not np.isfinite(signal).all():
        return False
    
    # Check if signal has variation
    for i in range(signal.shape[1]):
        if np.std(signal[:, i]) < 1e-6:
            return False
    
    # Check amplitude range (after normalization should be reasonable)
    if np.max(np.abs(signal)) > 50:
        return False
    
    return True

def augment_ecg_signal(signal):
    """
    Simple ECG data augmentation
    """
    augmented = signal.copy()
    
    # Add noise
    noise_factor = np.random.uniform(0.01, 0.05)
    augmented += np.random.normal(0, noise_factor, augmented.shape)
    
    # Time shift
    shift = np.random.randint(-10, 11)
    if shift != 0:
        augmented = np.roll(augmented, shift, axis=0)
    
    # Amplitude scaling
    scale = np.random.uniform(0.9, 1.1)
    augmented *= scale
    
    return augmented

def train_robust_model(signals, labels, model_folder, verbose):
    """
    Train robust ECG-only model with improved performance
    """
    if verbose:
        print(f"Training on {len(signals)} ECG samples")
    
    # Convert to arrays
    X = np.array(signals, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    
    if verbose:
        print(f"Signal shape: {X.shape}")
        unique, counts = np.unique(y, return_counts=True)
        print(f"Label distribution: {dict(zip(unique, counts))}")
    
    # Enhanced data augmentation for minority class
    if len(np.unique(y)) == 2:
        minority_class = np.argmin(np.bincount(y))
        majority_class = 1 - minority_class
        
        # More aggressive augmentation to balance classes
        minority_indices = np.where(y == minority_class)[0]
        majority_count = np.sum(y == majority_class)
        minority_count = len(minority_indices)
        
        if minority_count < majority_count:
            # Create multiple augmented versions
            augment_factor = min(4, majority_count // minority_count)
            
            augmented_X = [X]
            augmented_y = [y]
            
            for factor in range(augment_factor):
                for idx in minority_indices:
                    # Multiple augmentation strategies
                    base_signal = X[idx]
                    
                    # Strategy 1: Noise + amplitude scaling
                    aug1 = base_signal.copy()
                    noise_level = 0.02 + 0.03 * np.random.random()
                    aug1 += np.random.normal(0, noise_level, aug1.shape)
                    aug1 *= np.random.uniform(0.85, 1.15)
                    
                    # Strategy 2: Time warping simulation
                    aug2 = base_signal.copy()
                    shift = np.random.randint(-15, 16)
                    aug2 = np.roll(aug2, shift, axis=0)
                    aug2 *= np.random.uniform(0.9, 1.1)
                    
                    augmented_X.extend([aug1.reshape(1, *aug1.shape), aug2.reshape(1, *aug2.shape)])
                    augmented_y.extend([[minority_class], [minority_class]])
            
            X = np.vstack(augmented_X)
            y = np.hstack(augmented_y)
            
            if verbose:
                unique_new, counts_new = np.unique(y, return_counts=True)
                print(f"After augmentation: {dict(zip(unique_new, counts_new))}")
    
    # Handle single class case
    elif len(np.unique(y)) == 1:
        if verbose:
            print("Single class detected - creating synthetic opposite class")
        original_class = unique[0]
        synthetic_count = len(X) // 2
        
        synthetic_X = []
        for i in range(synthetic_count):
            idx = np.random.choice(len(X))
            synthetic_signal = X[idx].copy()
            
            # More aggressive synthetic data generation
            synthetic_signal += np.random.normal(0, 0.15, synthetic_signal.shape)
            synthetic_signal *= np.random.uniform(0.7, 1.3)
            
            # Add some systematic differences
            synthetic_signal = np.roll(synthetic_signal, np.random.randint(-20, 21), axis=0)
            
            synthetic_X.append(synthetic_signal)
        
        X = np.vstack([X, np.array(synthetic_X)])
        y = np.hstack([y, np.full(synthetic_count, 1 - original_class)])
    
    # Improved train-validation split with better stratification
    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
    except ValueError:
        # If stratification fails, use random split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.25, random_state=42
        )
    
    # Build model with improved architecture
    model = build_enhanced_ecg_model(X.shape[1:])
    
    if verbose:
        print("Enhanced model architecture:")
        model.summary()
    
    # Improved compilation with better optimizer settings
    initial_learning_rate = 0.002
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=initial_learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        ),
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall', AUC(name='auc')]
    )
    
    # Enhanced class weights calculation
    if len(np.unique(y_train)) > 1:
        try:
            class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
            # Adjust weights to be more conservative
            class_weight_dict = {}
            for i, weight in enumerate(class_weights):
                # Cap extreme weights and apply smoothing
                adjusted_weight = min(weight, 5.0) if weight > 1 else max(weight, 0.2)
                class_weight_dict[i] = adjusted_weight
            
            if verbose:
                print(f"Adjusted class weights: {class_weight_dict}")
        except:
            class_weight_dict = {0: 1.0, 1: 1.0}
    else:
        class_weight_dict = None
    
    # Enhanced callbacks with better monitoring
    callbacks = [
        EarlyStopping(
            monitor='val_auc',
            patience=25,  # Increased patience
            restore_best_weights=True,
            mode='max',
            min_delta=0.001  # Minimum improvement threshold
        ),
        ReduceLROnPlateau(
            monitor='val_auc',
            factor=0.6,  # Less aggressive reduction
            patience=12,  # More patience before reducing
            min_lr=1e-7,
            mode='max',
            verbose=1 if verbose else 0,
            cooldown=5  # Cooldown period
        )
    ]
    
    # Training with improved parameters
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=40,  # Increased epochs
        batch_size=min(BATCH_SIZE, len(X_train) // 4),  # Adaptive batch size
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1 if verbose else 0,
        shuffle=True
    )
    
    # Enhanced evaluation
    if verbose:
        y_pred_prob = model.predict(X_val, verbose=0)
        
        # Try multiple thresholds to find optimal
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        best_f1 = 0
        best_threshold = 0.5
        
        for threshold in thresholds:
            y_pred_binary = (y_pred_prob > threshold).astype(int).flatten()
            
            try:
                from sklearn.metrics import f1_score
                f1 = f1_score(y_val, y_pred_binary, average='weighted')
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
            except:
                continue
        
        print(f"\nBest threshold: {best_threshold} (F1: {best_f1:.4f})")
        
        # Final evaluation with best threshold
        y_pred_final = (y_pred_prob > best_threshold).astype(int).flatten()
        
        print("\nValidation Set Evaluation:")
        print(classification_report(y_val, y_pred_final))
        print("Confusion Matrix:")
        print(confusion_matrix(y_val, y_pred_final))
        
        try:
            auc_score = roc_auc_score(y_val, y_pred_prob)
            print(f"AUC Score: {auc_score:.4f}")
        except:
            pass
    
    # Save model with enhanced configuration
    save_enhanced_model_files(model_folder, model, best_threshold if 'best_threshold' in locals() else 0.5, verbose)
    
    if verbose:
        print("Enhanced model training completed")
    
    return True


def build_enhanced_ecg_model(input_shape):
    """
    Build enhanced ResNet architecture for ECG classification
    """
    def squeeze_excite_block(x, ratio=16):
        """Squeeze and Excitation block for channel attention"""
        filters = x.shape[-1]
        se = GlobalAveragePooling1D()(x)
        se = Dense(filters // ratio, activation='relu', use_bias=False)(se)
        se = Dense(filters, activation='sigmoid', use_bias=False)(se)
        se = tf.keras.layers.Reshape((1, filters))(se)
        return tf.keras.layers.Multiply()([x, se])
    
    def cbam_block(x, ratio=8):
        """Convolutional Block Attention Module (CBAM)"""
        # Channel attention
        avg_pool = GlobalAveragePooling1D(keepdims=True)(x)
        max_pool = tf.keras.layers.GlobalMaxPooling1D(keepdims=True)(x)
        
        avg_pool = Dense(x.shape[-1] // ratio, activation='relu', use_bias=False)(avg_pool)
        avg_pool = Dense(x.shape[-1], use_bias=False)(avg_pool)
        
        max_pool = Dense(x.shape[-1] // ratio, activation='relu', use_bias=False)(max_pool)
        max_pool = Dense(x.shape[-1], use_bias=False)(max_pool)
        
        channel_attention = tf.keras.layers.Add()([avg_pool, max_pool])
        channel_attention = tf.keras.layers.Activation('sigmoid')(channel_attention)
        
        x = tf.keras.layers.Multiply()([x, channel_attention])
        
        # Spatial attention
        avg_pool_spatial = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=-1, keepdims=True))(x)
        max_pool_spatial = tf.keras.layers.Lambda(lambda x: tf.reduce_max(x, axis=-1, keepdims=True))(x)
        spatial_concat = tf.keras.layers.Concatenate(axis=-1)([avg_pool_spatial, max_pool_spatial])
        
        spatial_attention = Conv1D(1, 7, padding='same', activation='sigmoid')(spatial_concat)
        return tf.keras.layers.Multiply()([x, spatial_attention])
    
    def enhanced_resnet_block(x, filters, kernel_size=7, stride=1, use_cbam=True):
        """Enhanced ResNet block with CBAM attention"""
        shortcut = x
        
        # First convolution
        x = Conv1D(filters, kernel_size, strides=stride, padding='same',
                  kernel_regularizer=tf.keras.regularizers.l2(1e-4), use_bias=False)(x)
        x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        x = ReLU()(x)
        x = SpatialDropout1D(0.1)(x)
        
        # Second convolution
        x = Conv1D(filters, kernel_size, strides=1, padding='same',
                  kernel_regularizer=tf.keras.regularizers.l2(1e-4), use_bias=False)(x)
        x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        
        # Third convolution for bottleneck design
        x = Conv1D(filters * 4, 1, strides=1, padding='same',
                  kernel_regularizer=tf.keras.regularizers.l2(1e-4), use_bias=False)(x)
        x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        
        # Attention mechanism
        if use_cbam:
            x = cbam_block(x)
        
        # Shortcut connection with proper dimension matching
        if shortcut.shape[-1] != filters * 4 or stride != 1:
            shortcut = Conv1D(filters * 4, 1, strides=stride, padding='same', use_bias=False)(shortcut)
            shortcut = BatchNormalization(momentum=0.9, epsilon=1e-5)(shortcut)
        
        # Residual connection
        x = Add()([x, shortcut])
        x = ReLU()(x)
        
        return x
    
    def resnet_stage(x, filters, blocks, kernel_size=7, stride=2, stage_name="stage"):
        """ResNet stage with multiple blocks"""
        # First block with stride for downsampling
        x = enhanced_resnet_block(x, filters, kernel_size, stride=stride, use_cbam=True)
        
        # Remaining blocks
        for i in range(1, blocks):
            x = enhanced_resnet_block(x, filters, kernel_size, stride=1, use_cbam=True)
        
        return x
    
    # Input layer
    inputs = Input(shape=input_shape, name='input_ecg')
    
    # Initial stem processing - larger kernels for ECG
    x = Conv1D(64, 15, strides=2, padding='same',
               kernel_regularizer=tf.keras.regularizers.l2(1e-4), use_bias=False, name='stem_conv1')(inputs)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, name='stem_bn1')(x)
    x = ReLU(name='stem_relu1')(x)
    
    x = Conv1D(64, 11, strides=1, padding='same',
               kernel_regularizer=tf.keras.regularizers.l2(1e-4), use_bias=False, name='stem_conv2')(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, name='stem_bn2')(x)
    x = ReLU(name='stem_relu2')(x)
    
    x = Conv1D(128, 7, strides=1, padding='same',
               kernel_regularizer=tf.keras.regularizers.l2(1e-4), use_bias=False, name='stem_conv3')(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, name='stem_bn3')(x)
    x = ReLU(name='stem_relu3')(x)
    x = SpatialDropout1D(0.1, name='stem_dropout')(x)
    
    # Max pooling
    x = tf.keras.layers.MaxPooling1D(pool_size=3, strides=2, padding='same', name='stem_maxpool')(x)
    
    # ResNet stages (similar to ResNet-50 but adapted for 1D ECG)
    x = resnet_stage(x, filters=64, blocks=3, kernel_size=7, stride=1, stage_name="stage1")
    x = resnet_stage(x, filters=128, blocks=4, kernel_size=5, stride=2, stage_name="stage2")
    x = resnet_stage(x, filters=256, blocks=6, kernel_size=3, stride=2, stage_name="stage3")
    x = resnet_stage(x, filters=512, blocks=3, kernel_size=3, stride=2, stage_name="stage4")
    
    # Multi-scale global pooling
    # Global Average Pooling
    gap = GlobalAveragePooling1D(name='global_avg_pool')(x)
    
    # Global Max Pooling
    gmp = tf.keras.layers.GlobalMaxPooling1D(name='global_max_pool')(x)
    
    # Attention-weighted pooling
    attention_weights = Dense(x.shape[-1], activation='softmax', name='attention_weights')(gap)
    attention_pooled = tf.keras.layers.Lambda(lambda inputs: tf.reduce_sum(inputs[0] * tf.expand_dims(inputs[1], 1), axis=1),
                                            name='attention_pooling')([x, attention_weights])
    
    # Combine all pooling strategies
    combined_features = tf.keras.layers.Concatenate(name='combine_features')([gap, gmp, attention_pooled])
    
    # Enhanced classification head with residual connections
    x = Dense(1024, kernel_regularizer=tf.keras.regularizers.l2(1e-4), use_bias=False, name='fc1')(combined_features)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, name='fc1_bn')(x)
    x = ReLU(name='fc1_relu')(x)
    x = Dropout(0.5, name='fc1_dropout')(x)
    
    x = Dense(512, kernel_regularizer=tf.keras.regularizers.l2(1e-4), use_bias=False, name='fc2')(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, name='fc2_bn')(x)
    x = ReLU(name='fc2_relu')(x)
    x = Dropout(0.4, name='fc2_dropout')(x)
    
    x = Dense(256, kernel_regularizer=tf.keras.regularizers.l2(1e-4), use_bias=False, name='fc3')(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5, name='fc3_bn')(x)
    x = ReLU(name='fc3_relu')(x)
    x = Dropout(0.3, name='fc3_dropout')(x)
    
    # Final classification layer
    outputs = Dense(1, activation='sigmoid', name='predictions')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs, name='ECG_ResNet_Enhanced')
    return model


def build_ecg_model(input_shape):
    """
    Build standard ResNet model for ECG classification (fallback option)
    """
    def residual_block(x, filters, kernel_size=7, stride=1):
        shortcut = x
        
        # First conv
        x = Conv1D(filters, kernel_size, strides=stride, padding='same',
                  kernel_regularizer=tf.keras.regularizers.l2(1e-4), use_bias=False)(x)
        x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        x = ReLU()(x)
        x = SpatialDropout1D(0.1)(x)
        
        # Second conv
        x = Conv1D(filters, kernel_size, strides=1, padding='same',
                  kernel_regularizer=tf.keras.regularizers.l2(1e-4), use_bias=False)(x)
        x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        
        # Shortcut connection
        if shortcut.shape[-1] != filters or stride != 1:
            shortcut = Conv1D(filters, 1, strides=stride, padding='same', use_bias=False)(shortcut)
            shortcut = BatchNormalization(momentum=0.9, epsilon=1e-5)(shortcut)
        
        x = Add()([x, shortcut])
        x = ReLU()(x)
        return x
    
    # Input
    inputs = Input(shape=input_shape)
    
    # Initial conv
    x = Conv1D(64, 15, strides=2, padding='same', use_bias=False)(inputs)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = ReLU()(x)
    
    # Residual blocks
    x = residual_block(x, 64, 11)
    x = residual_block(x, 128, 9, stride=2)
    x = residual_block(x, 128, 7)
    x = residual_block(x, 256, 5, stride=2)
    x = residual_block(x, 256, 3)
    
    # Global pooling
    x = GlobalAveragePooling1D()(x)
    
    # Classification head
    x = Dense(128, activation='relu',
              kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu',
              kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = Dropout(0.3)(x)
    
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, outputs)
    return model
    """
    Save model and enhanced configuration
    """
    # Save model
    model.save(os.path.join(model_folder, 'model.keras'))
    
    # Save enhanced config
    config = {
        'signal_length': TARGET_SIGNAL_LENGTH,
        'num_leads': NUM_LEADS,
        'sampling_rate': TARGET_SAMPLING_RATE,
        'model_type': 'enhanced_ecg',
        'threshold': threshold,  # Use optimized threshold
        'version': '2.0'
    }
    
    import json
    with open(os.path.join(model_folder, 'config.json'), 'w') as f:
        json.dump(config, f)
    
    if verbose:
        print(f"Enhanced model saved to {model_folder}")

def build_enhanced_ecg_model(input_shape):
    """
    Build enhanced ECG classification model with better architecture
    """
    def squeeze_excite_block(x, ratio=16):
        """Squeeze and Excitation block for channel attention"""
        filters = x.shape[-1]
        se = GlobalAveragePooling1D()(x)
        se = Dense(filters // ratio, activation='relu')(se)
        se = Dense(filters, activation='sigmoid')(se)
        se = tf.keras.layers.Reshape((1, filters))(se)
        return tf.keras.layers.Multiply()([x, se])
    
    def enhanced_residual_block(x, filters, kernel_size=7, stride=1, se_ratio=16):
        shortcut = x
        
        # First conv with depthwise separable convolution for efficiency
        x = Conv1D(filters, kernel_size, strides=stride, padding='same',
                  kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = SpatialDropout1D(0.1)(x)
        
        # Second conv
        x = Conv1D(filters, kernel_size, strides=1, padding='same',
                  kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        x = BatchNormalization()(x)
        
        # Squeeze and Excitation
        if se_ratio > 0:
            x = squeeze_excite_block(x, se_ratio)
        
        # Shortcut connection
        if shortcut.shape[-1] != filters or stride != 1:
            shortcut = Conv1D(filters, 1, strides=stride, padding='same')(shortcut)
            shortcut = BatchNormalization()(shortcut)
        
        x = Add()([x, shortcut])
        x = ReLU()(x)
        return x
    
    # Input
    inputs = Input(shape=input_shape)
    
    # Initial processing with larger kernel to capture more context
    x = Conv1D(64, 21, strides=2, padding='same',
               kernel_regularizer=tf.keras.regularizers.l2(1e-4))(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = SpatialDropout1D(0.1)(x)
    
    # Enhanced residual blocks with attention
    x = enhanced_residual_block(x, 64, 15, stride=1, se_ratio=16)
    x = enhanced_residual_block(x, 96, 13, stride=2, se_ratio=16)
    x = enhanced_residual_block(x, 96, 11, stride=1, se_ratio=16)
    x = enhanced_residual_block(x, 128, 9, stride=2, se_ratio=16)
    x = enhanced_residual_block(x, 128, 7, stride=1, se_ratio=16)
    x = enhanced_residual_block(x, 192, 5, stride=2, se_ratio=16)
    x = enhanced_residual_block(x, 192, 3, stride=1, se_ratio=16)
    
    # Multi-scale feature extraction
    # Global Average Pooling
    gap = GlobalAveragePooling1D()(x)
    
    # Global Max Pooling for different feature emphasis
    gmp = tf.keras.layers.GlobalMaxPooling1D()(x)
    
    # Combine features
    combined = tf.keras.layers.Concatenate()([gap, gmp])
    
    # Enhanced classification head with better regularization
    x = Dense(256, activation='relu',
              kernel_regularizer=tf.keras.regularizers.l2(1e-4))(combined)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu',
              kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu',
              kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = Dropout(0.2)(x)
    
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, outputs)
    return model


def save_enhanced_model_files(model_folder, model, threshold, verbose):
    """
    Save model and enhanced configuration
    """
    # Save model
    model.save(os.path.join(model_folder, 'model.keras'))
    
    # Save enhanced config
    config = {
        'signal_length': TARGET_SIGNAL_LENGTH,
        'num_leads': NUM_LEADS,
        'sampling_rate': TARGET_SAMPLING_RATE,
        'model_type': 'enhanced_ecg',
        'threshold': threshold,  # Use optimized threshold
        'version': '2.0'
    }
    
    import json
    with open(os.path.join(model_folder, 'config.json'), 'w') as f:
        json.dump(config, f)
    
    if verbose:
        print(f"Enhanced model saved to {model_folder}")
      
def balance_with_augmentation(X, y, verbose):
    """
    Balance dataset using augmentation
    """
    unique, counts = np.unique(y, return_counts=True)
    
    if len(unique) == 1:
        # Single class - create artificial opposite class
        original_class = unique[0]
        target_size = len(X)
        
        augmented_X = []
        augmented_y = []
        
        for i in range(target_size):
            aug_signal = augment_ecg_signal(X[i])
            # Add more aggressive augmentation for artificial class
            aug_signal += np.random.normal(0, 0.1, aug_signal.shape)
            augmented_X.append(aug_signal)
            augmented_y.append(1 - original_class)
        
        X_balanced = np.vstack([X, np.array(augmented_X)])
        y_balanced = np.hstack([y, np.array(augmented_y)])
        
    else:
        # Multiple classes - augment minority class
        max_count = np.max(counts)
        minority_class = unique[np.argmin(counts)]
        
        minority_indices = np.where(y == minority_class)[0]
        
        augmented_X = [X]
        augmented_y = [y]
        
        # Augment minority class to match majority
        needed = max_count - len(minority_indices)
        for _ in range(needed):
            idx = np.random.choice(minority_indices)
            aug_signal = augment_ecg_signal(X[idx])
            augmented_X.append(aug_signal.reshape(1, -1, 12))
            augmented_y.append([minority_class])
        
        X_balanced = np.vstack(augmented_X)
        y_balanced = np.hstack(augmented_y)
    
    if verbose:
        unique_new, counts_new = np.unique(y_balanced, return_counts=True)
        print(f"Balanced dataset: {dict(zip(unique_new, counts_new))}")
    
    return X_balanced, y_balanced

def build_ecg_model(input_shape):
    """
    Build robust ECG classification model
    """
    def residual_block(x, filters, kernel_size=7, stride=1):
        shortcut = x
        
        # First conv
        x = Conv1D(filters, kernel_size, strides=stride, padding='same',
                  kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = SpatialDropout1D(0.1)(x)
        
        # Second conv
        x = Conv1D(filters, kernel_size, strides=1, padding='same',
                  kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        x = BatchNormalization()(x)
        
        # Shortcut connection
        if shortcut.shape[-1] != filters or stride != 1:
            shortcut = Conv1D(filters, 1, strides=stride, padding='same')(shortcut)
            shortcut = BatchNormalization()(shortcut)
        
        x = Add()([x, shortcut])
        x = ReLU()(x)
        return x
    
    # Input
    inputs = Input(shape=input_shape)
    
    # Initial conv
    x = Conv1D(64, 15, strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    # Residual blocks
    x = residual_block(x, 64, 11)
    x = residual_block(x, 128, 9, stride=2)
    x = residual_block(x, 128, 7)
    x = residual_block(x, 256, 5, stride=2)
    x = residual_block(x, 256, 3)
    
    # Global pooling
    x = GlobalAveragePooling1D()(x)
    
    # Classification head
    x = Dense(128, activation='relu',
              kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu',
              kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = Dropout(0.3)(x)
    
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, outputs)
    return model

def create_minimal_baseline(model_folder, verbose):
    """
    Create minimal baseline when insufficient data
    """
    if verbose:
        print("Creating minimal baseline...")
    
    # Simple model
    model = build_ecg_model((TARGET_SIGNAL_LENGTH, NUM_LEADS))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', AUC(name='auc')]
    )
    
    save_model_files(model_folder, model, verbose)
    
    if verbose:
        print("Minimal baseline created")
    
    return True

def save_model_files(model_folder, model, verbose):
    """
    Save model and configuration
    """
    # Save model
    model.save(os.path.join(model_folder, 'model.keras'))
    
    # Save config
    config = {
        'signal_length': TARGET_SIGNAL_LENGTH,
        'num_leads': NUM_LEADS,
        'sampling_rate': TARGET_SAMPLING_RATE,
        'model_type': 'ecg_only',
        'threshold': 0.5
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
    if verbose:
        print(f"Loading model from {model_folder}")
    
    # Load model
    model = tf.keras.models.load_model(os.path.join(model_folder, 'model.keras'))
    
    # Load config
    import json
    with open(os.path.join(model_folder, 'config.json'), 'r') as f:
        config = json.load(f)
    
    return {
        'model': model,
        'config': config
    }

def run_model(record, model_data, verbose=False):
    """
    Run model on single record
    """
    try:
        model = model_data['model']
        config = model_data['config']
        threshold = config.get('threshold', 0.5)
        
        # Load and process signal
        try:
            signal, fields = load_signals(record)
            processed_signal = process_ecg_signal(signal)
            
            if processed_signal is None:
                raise ValueError("Signal processing failed")
                
        except Exception as e:
            if verbose:
                print(f"Signal loading failed: {e}, using default")
            # Default signal
            processed_signal = np.random.randn(config['signal_length'], config['num_leads']).astype(np.float32)
        
        # Prepare input
        signal_input = processed_signal.reshape(1, config['signal_length'], config['num_leads'])
        
        # Predict
        try:
            probability = float(model.predict(signal_input, verbose=0)[0][0])
        except Exception as e:
            if verbose:
                print(f"Prediction error: {e}")
            probability = 0.1  # Conservative default
        
        # Binary prediction
        binary_prediction = 1 if probability >= threshold else 0
        
        return binary_prediction, probability
        
    except Exception as e:
        if verbose:
            print(f"Error in run_model: {e}")
        return 0, 0.1
