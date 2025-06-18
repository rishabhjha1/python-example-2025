#!/usr/bin/env python

# Ultra memory-efficient Chagas disease detection model
# Optimized for PhysioNet Challenge server constraints

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Input, concatenate, BatchNormalization, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

from helper_code import *

# Aggressive memory management
import gc
tf.config.experimental.enable_memory_growth = True

# Ultra-conservative constants for memory efficiency
TARGET_SAMPLING_RATE = 400  # Match most PhysioNet data
TARGET_SIGNAL_LENGTH = 1000  # Very short - 2.5 seconds at 400Hz
MAX_SAMPLES = 1000  # Strict limit on training samples
BATCH_SIZE = 8  # Very small batches

def train_model(data_folder, model_folder, verbose):
    """
    Ultra memory-efficient training for Chagas disease detection
    """
    if verbose:
        print("Training ultra memory-efficient Chagas detection model...")
    
    os.makedirs(model_folder, exist_ok=True)
    
    try:
        # Try HDF5 approach first
        if all(os.path.exists(os.path.join(data_folder, f)) for f in ['exams.csv', 'samitrop_chagas_labels.csv', 'exams.hdf5']):
            return train_from_hdf5_minimal(data_folder, model_folder, verbose)
    except Exception as e:
        if verbose:
            print(f"HDF5 approach failed: {e}")
    
    # Fallback to WFDB with strict memory limits
    return train_from_wfdb_minimal(data_folder, model_folder, verbose)

def train_from_hdf5_minimal(data_folder, model_folder, verbose):
    """
    Minimal memory HDF5 training with better structure detection
    """
    import pandas as pd
    import h5py
    
    if verbose:
        print("Loading data with minimal memory usage...")
    
    # Load only essential data
    try:
        exams_df = pd.read_csv(os.path.join(data_folder, 'exams.csv'), nrows=MAX_SAMPLES)
        labels_df = pd.read_csv(os.path.join(data_folder, 'samitrop_chagas_labels.csv'), nrows=MAX_SAMPLES)
        
        if verbose:
            print(f"Exam columns: {list(exams_df.columns)}")
            print(f"Label columns: {list(labels_df.columns)}")
        
        # Try different merge strategies
        data_df = None
        
        # Strategy 1: Direct exam_id merge
        if 'exam_id' in exams_df.columns and 'exam_id' in labels_df.columns:
            data_df = pd.merge(exams_df, labels_df, on='exam_id', how='inner')
            if verbose:
                print(f"Merged on exam_id: {len(data_df)} samples")
        
        # Strategy 2: Try other column combinations
        if data_df is None or len(data_df) == 0:
            for exam_col in ['id', 'exam_id', 'record_id', 'index']:
                for label_col in ['exam_id', 'id', 'record_id', 'index']:
                    if exam_col in exams_df.columns and label_col in labels_df.columns:
                        data_df = pd.merge(exams_df, labels_df, 
                                         left_on=exam_col, right_on=label_col, 
                                         how='inner')
                        if len(data_df) > 0:
                            if verbose:
                                print(f"Merged on {exam_col}→{label_col}: {len(data_df)} samples")
                            break
                if data_df is not None and len(data_df) > 0:
                    break
        
        # Strategy 3: Index-based merge if columns don't work
        if data_df is None or len(data_df) == 0:
            min_len = min(len(exams_df), len(labels_df))
            data_df = pd.concat([
                exams_df.iloc[:min_len].reset_index(drop=True),
                labels_df.iloc[:min_len].reset_index(drop=True)
            ], axis=1)
            if verbose:
                print(f"Index-based merge: {len(data_df)} samples")
        
        if len(data_df) == 0:
            raise ValueError("No data after merging")
            
    except Exception as e:
        if verbose:
            print(f"CSV loading/merging failed: {e}")
        return create_minimal_dummy_model(model_folder, verbose)
    
    # Investigate HDF5 structure
    hdf5_path = os.path.join(data_folder, 'exams.hdf5')
    
    try:
        with h5py.File(hdf5_path, 'r') as hdf:
            if verbose:
                print(f"HDF5 structure investigation:")
                print(f"Root keys: {list(hdf.keys())}")
                
                # Explore the structure
                for key in list(hdf.keys())[:5]:  # Look at first 5 keys
                    item = hdf[key]
                    if hasattr(item, 'shape'):
                        print(f"  {key}: dataset with shape {item.shape}")
                    elif hasattr(item, 'keys'):
                        print(f"  {key}: group with keys {list(item.keys())[:10]}")
                        # Look deeper into groups
                        for subkey in list(item.keys())[:3]:
                            subitem = item[subkey]
                            if hasattr(subitem, 'shape'):
                                print(f"    {subkey}: shape {subitem.shape}")
                    else:
                        print(f"  {key}: unknown type {type(item)}")
            
            # Try to extract signals with multiple strategies
            signals = []
            metadata = []
            labels = []
            
            processed = 0
            
            # Strategy 1: Direct dataset access
            root_keys = list(hdf.keys())
            main_key = root_keys[0] if root_keys else None
            
            if main_key and hasattr(hdf[main_key], 'shape'):
                # Single large dataset
                if verbose:
                    print(f"Found main dataset '{main_key}' with shape: {hdf[main_key].shape}")
                
                dataset = hdf[main_key]
                max_samples = min(len(data_df), dataset.shape[0], MAX_SAMPLES)
                
                for idx in range(max_samples):
                    try:
                        row = data_df.iloc[idx] if idx < len(data_df) else data_df.iloc[idx % len(data_df)]
                        
                        # Extract metadata
                        age = float(row.get('age', 50.0)) if not pd.isna(row.get('age', 50.0)) else 50.0
                        sex = str(row.get('sex', 'U')).lower()
                        sex_encoding = 1.0 if sex.startswith('f') else 0.0
                        
                        # Extract label
                        chagas_label = None
                        for col in ['chagas', 'label', 'target', 'diagnosis']:
                            if col in row and not pd.isna(row[col]):
                                chagas_label = int(row[col])
                                break
                        
                        if chagas_label is None:
                            continue
                        
                        # Extract signal from dataset
                        try:
                            if len(dataset.shape) == 3:  # (samples, time, leads)
                                signal_data = dataset[idx]
                            elif len(dataset.shape) == 2:  # (samples, features) - might be flattened
                                signal_data = dataset[idx].reshape(-1, 12)  # Assume 12 leads
                            else:
                                continue
                            
                            processed_signal = process_signal_minimal(signal_data)
                            if processed_signal is None:
                                continue
                            
                            signals.append(processed_signal)
                            metadata.append(np.array([age / 100.0, sex_encoding]))
                            labels.append(chagas_label)
                            processed += 1
                            
                            if verbose and processed <= 5:
                                print(f"  Processed sample {processed}: age={age}, sex={sex}, chagas={chagas_label}")
                            
                        except Exception as e:
                            if verbose and processed < 3:
                                print(f"  Error extracting signal {idx}: {e}")
                            continue
                            
                    except Exception as e:
                        if verbose and processed < 3:
                            print(f"  Error processing sample {idx}: {e}")
                        continue
            
            # Strategy 2: Group-based access
            elif main_key and hasattr(hdf[main_key], 'keys'):
                if verbose:
                    print(f"Found group '{main_key}', exploring subkeys...")
                
                group = hdf[main_key]
                subkeys = list(group.keys())
                
                for idx, (_, row) in enumerate(data_df.iterrows()):
                    if processed >= MAX_SAMPLES:
                        break
                    
                    try:
                        # Extract metadata and label (same as above)
                        age = float(row.get('age', 50.0)) if not pd.isna(row.get('age', 50.0)) else 50.0
                        sex = str(row.get('sex', 'U')).lower()
                        sex_encoding = 1.0 if sex.startswith('f') else 0.0
                        
                        chagas_label = None
                        for col in ['chagas', 'label', 'target', 'diagnosis']:
                            if col in row and not pd.isna(row[col]):
                                chagas_label = int(row[col])
                                break
                        
                        if chagas_label is None:
                            continue
                        
                        # Try to find signal by various keys
                        signal_data = None
                        exam_id = row.get('exam_id', row.get('id', idx))
                        
                        # Try different key formats
                        potential_keys = [
                            str(exam_id),
                            f'{exam_id:05d}',
                            f'{exam_id:06d}',
                            f'exam_{exam_id}',
                            f'record_{exam_id}'
                        ]
                        
                        for key in potential_keys:
                            if key in subkeys:
                                try:
                                    signal_data = group[key][:]
                                    break
                                except:
                                    continue
                        
                        # Try accessing by index if key-based fails
                        if signal_data is None and idx < len(subkeys):
                            try:
                                signal_data = group[subkeys[idx]][:]
                            except:
                                pass
                        
                        if signal_data is None:
                            continue
                        
                        processed_signal = process_signal_minimal(signal_data)
                        if processed_signal is None:
                            continue
                        
                        signals.append(processed_signal)
                        metadata.append(np.array([age / 100.0, sex_encoding]))
                        labels.append(chagas_label)
                        processed += 1
                        
                        if verbose and processed <= 5:
                            print(f"  Processed sample {processed}: age={age}, sex={sex}, chagas={chagas_label}")
                            
                    except Exception as e:
                        if verbose and processed < 3:
                            print(f"  Error processing sample {idx}: {e}")
                        continue
            
            if verbose:
                print(f"Successfully extracted {processed} samples from HDF5")
                
    except Exception as e:
        if verbose:
            print(f"HDF5 processing failed: {e}")
        processed = 0
        signals = []
        
    if len(signals) < 10:
        if verbose:
            print(f"Insufficient data ({len(signals)} samples), creating minimal model")
        return create_minimal_dummy_model(model_folder, verbose)
    
    # Train minimal model
    return train_minimal_model(signals, metadata, labels, model_folder, verbose)

def train_from_wfdb_minimal(data_folder, model_folder, verbose):
    """
    Minimal WFDB training with strict memory limits
    """
    try:
        records = find_records(data_folder)[:MAX_SAMPLES]  # Limit records
        
        signals = []
        metadata = []
        labels = []
        
        for i, record_name in enumerate(records):
            if len(signals) >= MAX_SAMPLES:
                break
                
            try:
                record_path = os.path.join(data_folder, record_name)
                
                # Try to load label
                try:
                    label = load_label(record_path)
                except:
                    continue
                
                # Load minimal features
                age, sex, _, signal_mean, signal_std = extract_features(record_path)
                if age is None or sex is None:
                    continue
                
                # Create minimal signal representation
                signal_data = create_minimal_signal(signal_mean, signal_std)
                
                signals.append(signal_data)
                metadata.append(np.array([age[0] / 100.0, 1.0 if sex[0] > 0.5 else 0.0]))
                labels.append(int(label))
                
                if verbose and len(signals) % 50 == 0:
                    print(f"Processed {len(signals)} WFDB records")
                    
            except:
                continue
        
        if len(signals) < 5:
            return create_minimal_dummy_model(model_folder, verbose)
        
        return train_minimal_model(signals, metadata, labels, model_folder, verbose)
        
    except:
        return create_minimal_dummy_model(model_folder, verbose)

def process_signal_minimal(signal_data):
    """
    Ultra-minimal signal processing to save memory
    """
    try:
        # Convert to numpy if needed
        if hasattr(signal_data, 'shape'):
            signal = np.array(signal_data)
        else:
            signal = np.array(signal_data)
        
        # Handle shape
        if len(signal.shape) == 1:
            signal = signal.reshape(-1, 1)
        elif signal.shape[0] < signal.shape[1] and signal.shape[1] > 20:
            signal = signal.T
        
        # Take only first 3 leads to save memory (I, II, III)
        if signal.shape[1] > 3:
            signal = signal[:, :3]
        elif signal.shape[1] < 3:
            # Pad with zeros
            padding = np.zeros((signal.shape[0], 3 - signal.shape[1]))
            signal = np.hstack([signal, padding])
        
        # Downsample aggressively
        if signal.shape[0] > TARGET_SIGNAL_LENGTH:
            step = signal.shape[0] // TARGET_SIGNAL_LENGTH
            signal = signal[::step][:TARGET_SIGNAL_LENGTH]
        elif signal.shape[0] < TARGET_SIGNAL_LENGTH:
            # Pad
            padding = np.zeros((TARGET_SIGNAL_LENGTH - signal.shape[0], 3))
            signal = np.vstack([signal, padding])
        
        # Simple normalization
        signal = (signal - np.mean(signal, axis=0, keepdims=True)) / (np.std(signal, axis=0, keepdims=True) + 1e-8)
        signal = np.clip(signal, -3, 3)
        
        return signal.astype(np.float32)
        
    except:
        return np.zeros((TARGET_SIGNAL_LENGTH, 3), dtype=np.float32)

def create_minimal_signal(signal_mean, signal_std):
    """
    Create minimal synthetic signal
    """
    try:
        signal = np.zeros((TARGET_SIGNAL_LENGTH, 3), dtype=np.float32)
        
        # Use only first 3 leads
        for i in range(3):
            if i < len(signal_mean):
                mean = signal_mean[i] if not np.isnan(signal_mean[i]) else 0.0
                std = signal_std[i] if not np.isnan(signal_std[i]) else 1.0
                
                # Simple sinusoidal pattern
                t = np.linspace(0, 2.5, TARGET_SIGNAL_LENGTH)  # 2.5 seconds
                pattern = mean + std * np.sin(2 * np.pi * 1.2 * t)  # 72 bpm
                signal[:, i] = pattern
        
        return signal.astype(np.float32)
        
    except:
        return np.zeros((TARGET_SIGNAL_LENGTH, 3), dtype=np.float32)

def train_minimal_model(signals, metadata, labels, model_folder, verbose):
    """
    Train ultra-minimal model
    """
    if verbose:
        print(f"Training on {len(signals)} samples with minimal memory usage")
    
    # Convert to arrays
    signals = np.array(signals, dtype=np.float32)
    metadata = np.array(metadata, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    
    if verbose:
        print(f"Signal shape: {signals.shape}")
        print(f"Chagas positive: {np.sum(labels)} ({np.mean(labels)*100:.1f}%)")
    
    # Scale metadata
    scaler = StandardScaler()
    metadata_scaled = scaler.fit_transform(metadata)
    
    # Build minimal model
    model = build_minimal_model(signals.shape[1:], metadata.shape[1])
    
    if verbose:
        print("Minimal model architecture:")
        model.summary()
    
    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Train with minimal resources
    if len(signals) >= 20:
        X_train, X_val, X_meta_train, X_meta_val, y_train, y_val = train_test_split(
            signals, metadata_scaled, labels, test_size=0.2, random_state=42
        )
        
        # Minimal callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        ]
        
        model.fit(
            [X_train, X_meta_train], y_train,
            validation_data=([X_val, X_meta_val], y_val),
            epochs=20,  # Fewer epochs
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=1 if verbose else 0
        )
    else:
        model.fit(
            [signals, metadata_scaled], labels,
            epochs=10,
            batch_size=min(BATCH_SIZE, len(signals)),
            verbose=1 if verbose else 0
        )
    
    # Save minimal model
    save_minimal_model(model_folder, model, scaler, verbose)
    
    if verbose:
        print("Minimal Chagas model training completed")

def build_minimal_model(signal_shape, metadata_features_count):
    """
    Build ultra-minimal model architecture
    """
    # Signal branch - very small
    signal_input = Input(shape=signal_shape, name='signal_input')
    
    x = Conv1D(8, kernel_size=5, activation='relu')(signal_input)  # Very few filters
    x = MaxPooling1D(pool_size=4)(x)
    
    x = Conv1D(16, kernel_size=3, activation='relu')(x)
    x = GlobalAveragePooling1D()(x)  # More memory efficient than flatten
    
    x = Dense(16, activation='relu')(x)  # Small dense layer
    signal_features = Dropout(0.3)(x)
    
    # Metadata branch - minimal
    metadata_input = Input(shape=(metadata_features_count,), name='metadata_input')
    metadata_features = Dense(8, activation='relu')(metadata_input)
    
    # Combine
    combined = concatenate([signal_features, metadata_features])
    combined = Dense(8, activation='relu')(combined)
    output = Dense(1, activation='sigmoid')(combined)
    
    model = Model(inputs=[signal_input, metadata_input], outputs=output)
    return model

def create_minimal_dummy_model(model_folder, verbose):
    """
    Create minimal dummy model when no data available
    """
    if verbose:
        print("Creating minimal dummy model...")
    
    os.makedirs(model_folder, exist_ok=True)
    
    # Build minimal dummy model
    model = build_minimal_model((TARGET_SIGNAL_LENGTH, 3), 2)
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Minimal dummy training
    dummy_signals = np.random.randn(20, TARGET_SIGNAL_LENGTH, 3).astype(np.float32)
    dummy_metadata = np.random.randn(20, 2).astype(np.float32)
    dummy_labels = np.random.randint(0, 2, 20)
    
    model.fit([dummy_signals, dummy_metadata], dummy_labels, epochs=1, verbose=0)
    
    # Dummy scaler
    scaler = StandardScaler()
    scaler.fit(dummy_metadata)
    
    save_minimal_model(model_folder, model, scaler, verbose)
    
    if verbose:
        print("Minimal dummy model created")

def save_minimal_model(model_folder, model, scaler, verbose):
    """
    Save model with minimal overhead
    """
    # Save model
    model.save(os.path.join(model_folder, 'model.keras'))
    
    # Save scaler parameters
    np.save(os.path.join(model_folder, 'scaler_mean.npy'), scaler.mean_)
    np.save(os.path.join(model_folder, 'scaler_scale.npy'), scaler.scale_)
    
    # Save dimensions
    np.save(os.path.join(model_folder, 'signal_dims.npy'), np.array([TARGET_SIGNAL_LENGTH, 3]))
    
    if verbose:
        print("Minimal model saved")

def load_model(model_folder, verbose=False):
    """
    Load minimal model
    """
    if verbose:
        print(f"Loading model from {model_folder}")
    
    model = tf.keras.models.load_model(os.path.join(model_folder, 'model.keras'))
    
    scaler = StandardScaler()
    scaler.mean_ = np.load(os.path.join(model_folder, 'scaler_mean.npy'))
    scaler.scale_ = np.load(os.path.join(model_folder, 'scaler_scale.npy'))
    
    signal_dims = np.load(os.path.join(model_folder, 'signal_dims.npy'))
    
    return {
        'model': model,
        'scaler': scaler,
        'signal_dims': signal_dims
    }

def run_model(record, model_data, verbose=False):
    """
    Run minimal model on record
    """
    try:
        model = model_data['model']
        scaler = model_data['scaler']
        
        # Load record with minimal processing
        try:
            age, sex, _, signal_mean, signal_std = extract_features(record)
            
            if age is None or sex is None:
                age = np.array([50.0])
                sex = np.array([0, 0, 1])
            
            # Create minimal signal
            signal_data = create_minimal_signal(signal_mean, signal_std)
            
            # Prepare metadata
            metadata = np.array([age[0] / 100.0, 1.0 if sex[0] > 0.5 else 0.0]).reshape(1, -1)
            metadata_scaled = scaler.transform(metadata)
            
            # Prepare signal
            signal_input = signal_data.reshape(1, TARGET_SIGNAL_LENGTH, 3)
            
            # Predict
            probability = float(model.predict([signal_input, metadata_scaled], verbose=0)[0][0])
            binary_prediction = 1 if probability >= 0.5 else 0
            
            return binary_prediction, probability
            
        except Exception as e:
            if verbose:
                print(f"Error processing record: {e}")
            return 0, 0.5
        
    except Exception as e:
        if verbose:
            print(f"Error in run_model: {e}")
        return 0, 0.0

def extract_features(record):
    """
    Extract basic features (reusing from helper_code)
    """
    try:
        header = load_header(record)
    except Exception:
        return None, None, None, None, None

    # Extract age
    try:
        age = get_age(header)
        if age is None:
            age = 50.0
        age = np.array([float(age)])
    except Exception:
        age = np.array([50.0])

    # Extract sex (simplified)
    try:
        sex = get_sex(header)
        sex_encoding = np.zeros(3, dtype=float)
        if sex is not None and sex.casefold().startswith('f'):
            sex_encoding[0] = 1
        elif sex is not None and sex.casefold().startswith('m'):
            sex_encoding[1] = 1
        else:
            sex_encoding[2] = 1
    except Exception:
        sex_encoding = np.array([0, 0, 1])

    # Extract source
    try:
        source = get_source(header)
        if source is None:
            source = 'Unknown'
    except Exception:
        source = 'Unknown'

    # Extract basic signal statistics
    try:
        signal, fields = load_signals(record)
        
        # Simplified processing
        if signal.shape[1] > 12:
            signal = signal[:, :12]
        elif signal.shape[1] < 12:
            padding = np.zeros((signal.shape[0], 12 - signal.shape[1]))
            signal = np.hstack([signal, padding])
        
        signal_mean = np.nanmean(signal, axis=0)
        signal_std = np.nanstd(signal, axis=0)
        
        signal_mean = np.nan_to_num(signal_mean, nan=0.0)
        signal_std = np.nan_to_num(signal_std, nan=1.0)

        return age, sex_encoding, source, signal_mean, signal_std
        
    except Exception:
        return age, sex_encoding, source, np.zeros(12), np.ones(12)
