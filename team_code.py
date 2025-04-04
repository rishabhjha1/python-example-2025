#!/usr/bin/env python

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, BatchNormalization, Input, Concatenate, GlobalAveragePooling1D, SpatialDropout1D, add
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import wfdb
from scipy.signal import resample, butter, filtfilt
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

# Use memory growth to avoid GPU memory issues
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
except:
    pass

def butter_bandpass(lowcut, highcut, fs, order=5):
    """Design a butterworth bandpass filter"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(data, lowcut=0.5, highcut=40.0, fs=500, order=5):
    """Apply a bandpass filter to remove noise"""
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data, axis=0)
    return y

def load_and_preprocess_data(data_directory, target_length=5000, num_leads=12, sample_size=None, augment=False, verbose=False):
    """
    Load and preprocess ECG data
    - Enhanced preprocessing with bandpass filtering
    - Optionally generates augmented samples
    """
    record_files = glob.glob(os.path.join(data_directory, "**/*.hea"), recursive=True)
    
    if sample_size and sample_size < len(record_files):
        record_files = np.random.choice(record_files, sample_size, replace=False)
    
    if verbose:
        print(f"Loading {len(record_files)} records from {data_directory}")
    
    # Lists to hold our processed data and labels
    X = []
    y = []
    
    for header_file in tqdm(record_files, disable=not verbose):
        record_path = os.path.splitext(header_file)[0]
        record_name = os.path.basename(record_path)
        
        try:
            # Read the record and header
            record = wfdb.rdrecord(record_path)
            signal = record.p_signal
            
            # Check if Chagas annotation exists in comments
            chagas_present = 0
            if hasattr(record, 'comments') and record.comments:
                for comment in record.comments:
                    if 'chagas' in comment.lower() or 'chagasic' in comment.lower():
                        chagas_present = 1
                        break
            
            # Apply bandpass filter to remove noise (0.5-40 Hz)
            signal = apply_bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=record.fs)
            
            # Resample to target length
            if signal.shape[0] != target_length:
                signal = resample(signal, target_length)
            
            # Handle lead count mismatch
            if signal.shape[1] != num_leads:
                if signal.shape[1] > num_leads:
                    signal = signal[:, :num_leads]
                else:
                    pad_width = ((0, 0), (0, num_leads - signal.shape[1]))
                    signal = np.pad(signal, pad_width, 'constant')
            
            # Add the original sample
            X.append(signal)
            y.append(chagas_present)
            
            # Data augmentation (if enabled and this is a positive sample)
            if augment and chagas_present == 1:
                # Time shifting (shift right by 200 samples)
                shift_right = np.roll(signal, 200, axis=0)
                X.append(shift_right)
                y.append(chagas_present)
                
                # Time shifting (shift left by 200 samples)
                shift_left = np.roll(signal, -200, axis=0)
                X.append(shift_left)
                y.append(chagas_present)
                
                # Add small Gaussian noise
                noise = signal + np.random.normal(0, 0.01, signal.shape)
                X.append(noise)
                y.append(chagas_present)
                
                # Amplitude scaling (0.9x)
                scaled_down = signal * 0.9
                X.append(scaled_down)
                y.append(chagas_present)
                
                # Amplitude scaling (1.1x)
                scaled_up = signal * 1.1 
                X.append(scaled_up)
                y.append(chagas_present)
                
        except Exception as e:
            if verbose:
                print(f"Error processing {record_name}: {e}")
    
    if not X:
        if verbose:
            print("No valid records found!")
        return None, None
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    if verbose:
        print(f"Data loaded: {X.shape}, Labels: {y.shape}, Positive cases: {np.sum(y)}")
        print(f"Class distribution: {np.bincount(y)}")
    
    return X, y

def build_resnet_block(x, filters, kernel_size=5, strides=1):
    """Build a residual block for our model"""
    shortcut = x
    
    # First convolution
    x = Conv1D(filters, kernel_size, padding='same', strides=strides)(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    
    # Second convolution
    x = Conv1D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    
    # If dimensions don't match, adjust the shortcut
    if strides != 1 or shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters, 1, padding='same', strides=strides)(shortcut)
        shortcut = BatchNormalization()(shortcut)
    
    # Add shortcut to output
    x = add([x, shortcut])
    x = tf.keras.layers.Activation('relu')(x)
    
    return x

def create_model(input_shape, dropout_rate=0.5):
    """
    Create an improved model architecture with residual connections
    """
    inputs = Input(shape=input_shape)
    
    # Initial convolution
    x = Conv1D(64, kernel_size=7, strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = MaxPooling1D(pool_size=3, strides=2, padding='same')(x)
    
    # Residual blocks
    x = build_resnet_block(x, 64)
    x = build_resnet_block(x, 64)
    x = MaxPooling1D(pool_size=2)(x)
    
    x = build_resnet_block(x, 128, strides=2)
    x = build_resnet_block(x, 128)
    x = MaxPooling1D(pool_size=2)(x)
    
    x = build_resnet_block(x, 256, strides=2)
    x = build_resnet_block(x, 256)
    
    # Global pooling
    x = GlobalAveragePooling1D()(x)
    
    # Dense layers
    x = Dense(128, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(dropout_rate/2)(x)
    
    # Output layer
    outputs = Dense(1, activation='sigmoid')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

def train_model(data_directory, model_directory, verbose=False):
    """
    Train an improved model with better preprocessing and save it
    """
    if verbose:
        print(f"Training improved model...")
    
    # Ensure model directory exists
    os.makedirs(model_directory, exist_ok=True)
    
    # Set parameters
    target_length = 5000  # Increased from 2500 for better resolution
    num_leads = 12
    batch_size = 32
    epochs = 100  # Will use early stopping
    
    # Load and preprocess data
    X, y = load_and_preprocess_data(
        data_directory, 
        target_length=target_length,
        num_leads=num_leads,
        augment=True,  # Enable data augmentation
        verbose=verbose
    )
    
    if X is None or len(X) == 0:
        if verbose:
            print("No data loaded, cannot train model")
        return
    
    # Split data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Apply robust scaling (less sensitive to outliers)
    scaler = RobustScaler()
    
    # Reshape for scaling
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)
    
    # Fit scaler on training data only
    scaler.fit(X_train_flat)
    
    # Transform both sets
    X_train_scaled = scaler.transform(X_train_flat).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val_flat).reshape(X_val.shape)
    
    # Save scaler parameters
    np.save(os.path.join(model_directory, 'scaler_mean.npy'), scaler.center_)
    np.save(os.path.join(model_directory, 'scaler_scale.npy'), scaler.scale_)
    
    # Store signal dimensions for later use
    np.save(os.path.join(model_directory, 'signal_dims.npy'), np.array([target_length, num_leads]))
    
    # Create model
    model = create_model(input_shape=(target_length, num_leads), dropout_rate=0.5)
    
    # Calculate class weights to handle imbalance
    class_counts = np.bincount(y_train)
    class_weights = {0: 1.0, 1: class_counts[0]/class_counts[1] if len(class_counts) > 1 and class_counts[1] > 0 else 1.0}
    
    if verbose:
        print(f"Using class weights: {class_weights}")
    
    # Compile model with weighted binary cross-entropy
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    # Set up callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_auc',
            patience=15,
            mode='max',
            restore_best_weights=True,
            verbose=verbose
        ),
        ReduceLROnPlateau(
            monitor='val_auc',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            mode='max',
            verbose=verbose
        ),
        ModelCheckpoint(
            filepath=os.path.join(model_directory, 'best_model.h5'),
            monitor='val_auc',
            save_best_only=True,
            mode='max',
            verbose=verbose
        )
    ]
    
    # Train model
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1 if verbose else 0
    )
    
    # Save model
    model.save(os.path.join(model_directory, 'model.h5'), save_format='h5')
    
    # Plot training history if verbose
    if verbose:
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['auc'], label='Train AUC')
        plt.plot(history.history['val_auc'], label='Val AUC')
        plt.title('AUC')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(model_directory, 'training_history.png'))
        
        # Get final performance metrics
        val_loss, val_acc, val_auc = model.evaluate(
            X_val_scaled, y_val, 
            verbose=0
        )
        
        print(f"Final validation results - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, AUC: {val_auc:.4f}")
    
    return model

def load_model(model_directory, verbose=False):
    """
    Load model and metadata with improved error handling
    """
    if verbose:
        print(f"Loading model from {model_directory}")
    
    # Try loading best model first, fall back to model.h5
    if os.path.exists(os.path.join(model_directory, 'best_model.h5')):
        model_path = os.path.join(model_directory, 'best_model.h5')
    else:
        model_path = os.path.join(model_directory, 'model.h5')
    
    # Load model with reduced memory usage
    model = tf.keras.models.load_model(
        model_path,
        compile=False  # Don't compile model immediately to save memory
    )
    
    # Compile with metrics
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )
    
    # Load the scaler - use RobustScaler if available
    if os.path.exists(os.path.join(model_directory, 'scaler_center_.npy')):
        # This is a RobustScaler
        scaler = RobustScaler()
        scaler.center_ = np.load(os.path.join(model_directory, 'scaler_center_.npy'))
        scaler.scale_ = np.load(os.path.join(model_directory, 'scaler_scale_.npy'))
    else:
        # Fall back to StandardScaler format
        scaler = RobustScaler() if os.path.exists(os.path.join(model_directory, 'scaler_center_.npy')) else StandardScaler()
        scaler.center_ = np.load(os.path.join(model_directory, 'scaler_mean.npy'))
        scaler.scale_ = np.load(os.path.join(model_directory, 'scaler_scale.npy'))
    
    # Get expected signal dimensions
    if os.path.exists(os.path.join(model_directory, 'signal_dims.npy')):
        signal_dims = np.load(os.path.join(model_directory, 'signal_dims.npy'))
    else:
        signal_dims = np.array([5000, 12])  # Default to improved dimensions
    
    return {
        'model': model, 
        'scaler': scaler,
        'signal_dims': signal_dims
    }

def run_model(record, model_data, verbose=False):
    """
    Run model on a record with improved preprocessing
    """
    try:
        if verbose:
            print(f"Processing record: {record}")
        
        # Extract model, scaler and expected dimensions
        model = model_data['model']
        scaler = model_data['scaler']
        signal_dims = model_data.get('signal_dims', np.array([5000, 12]))
        target_length, num_leads = int(signal_dims[0]), int(signal_dims[1])
        
        # Try to load the signal
        try:
            signal, meta = wfdb.rdsamp(record)
            fs = meta['fs'] if 'fs' in meta else 500  # Default sampling rate if not available
        except Exception as e:
            if verbose:
                print(f"Error loading signal, using zeros: {e}")
            signal = np.zeros((target_length, num_leads))
            fs = 500  # Default sampling rate
        
        # Apply bandpass filter to clean the signal
        try:
            signal = apply_bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=fs)
        except Exception as e:
            if verbose:
                print(f"Error applying filter: {e}")
        
        # Resample to target length
        if signal.shape[0] != target_length:
            try:
                signal = resample(signal, target_length)
            except Exception as e:
                if verbose:
                    print(f"Error resampling: {e}")
                signal = np.zeros((target_length, signal.shape[1]))
        
        # Ensure we have exactly the expected number of leads
        if signal.shape[1] != num_leads:
            if signal.shape[1] > num_leads:
                signal = signal[:, :num_leads]
            else:
                # Pad with zeros
                pad_width = ((0, 0), (0, num_leads - signal.shape[1]))
                signal = np.pad(signal, pad_width, 'constant')
        
        # Reshape for scaling
        signal_flat = signal.reshape(1, -1)
        
        # Apply scaling
        signal_scaled = scaler.transform(signal_flat).reshape(1, target_length, num_leads)
        
        # Make prediction
        probability = float(model.predict(signal_scaled, verbose=0, batch_size=1)[0][0])
        binary_prediction = 1 if probability >= 0.5 else 0
        
        return binary_prediction, probability
        
    except Exception as e:
        if verbose:
            print(f"Error in run_model: {e}")
        # Default prediction in case of error
        return 0, 0.0

# Optional function for model evaluation
def evaluate_model(model_data, test_directory, verbose=False):
    """
    Evaluate model performance on a test set
    """
    if verbose:
        print("Evaluating model performance...")
    
    # Find all records in the test directory
    record_files = glob.glob(os.path.join(test_directory, "**/*.hea"), recursive=True)
    
    if not record_files:
        if verbose:
            print("No test records found")
        return None
    
    predictions = []
    actual_labels = []
    probabilities = []
    
    for header_file in tqdm(record_files, disable=not verbose):
        record_path = os.path.splitext(header_file)[0]
        record_name = os.path.basename(record_path)
        
        # Run model on this record
        prediction, probability = run_model(record_path, model_data, verbose=False)
        predictions.append(prediction)
        probabilities.append(probability)
        
        # Try to extract the actual label
        try:
            record = wfdb.rdrecord(record_path)
            chagas_present = 0
            if hasattr(record, 'comments') and record.comments:
                for comment in record.comments:
                    if 'chagas' in comment.lower() or 'chagasic' in comment.lower():
                        chagas_present = 1
                        break
            actual_labels.append(chagas_present)
        except:
            # If we can't get the label, don't add to actual_labels
            pass
    
    # If we have labels, compute metrics
    if actual_labels:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
        
        accuracy = accuracy_score(actual_labels, predictions)
        precision = precision_score(actual_labels, predictions, zero_division=0)
        recall = recall_score(actual_labels, predictions, zero_division=0)
        f1 = f1_score(actual_labels, predictions, zero_division=0)
        
        try:
            auc = roc_auc_score(actual_labels, probabilities)
        except:
            auc = 0
            
        cm = confusion_matrix(actual_labels, predictions)
        
        if verbose:
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print(f"AUC: {auc:.4f}")
            print(f"Confusion Matrix:")
            print(cm)
            
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'confusion_matrix': cm
        }
    
    return None