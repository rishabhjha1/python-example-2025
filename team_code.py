
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, BatchNormalization
from sklearn.preprocessing import StandardScaler
import wfdb
from scipy.signal import resample, butter, filtfilt
import glob

# Use memory growth to avoid GPU memory issues
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
except:
    pass

def load_challenge_data(data_directory, max_samples=1000):
    """
    Load actual challenge data for training with memory efficiency.
    """
    # Find all header files
    header_files = glob.glob(os.path.join(data_directory, '*.hea'))
    
    if len(header_files) == 0:
        print("No header files found in data directory")
        return None, None
    
    # Limit samples to prevent memory issues
    header_files = header_files[:max_samples]
    
    signals = []
    labels = []
    
    print(f"Loading {len(header_files)} records...")
    
    for i, header_file in enumerate(header_files):
        if i % 100 == 0:
            print(f"Processed {i}/{len(header_files)} records")
            
        try:
            record_name = header_file.replace('.hea', '')
            
            # Load signal
            signal, fields = wfdb.rdsamp(record_name)
            
            # Load header to get labels
            header = wfdb.rdheader(record_name)
            
            # Extract label from comments (this may need adjustment based on actual format)
            label = 0  # Default
            if hasattr(header, 'comments') and header.comments:
                for comment in header.comments:
                    if 'diagnosis' in comment.lower() or 'dx' in comment.lower():
                        # Extract binary label - adjust this based on actual label format
                        if any(term in comment.lower() for term in ['abnormal', 'pathological', 'disease']):
                            label = 1
                        break
            
            signals.append(signal)
            labels.append(label)
            
        except Exception as e:
            print(f"Error loading {header_file}: {e}")
            continue
    
    return signals, labels

def preprocess_signal(signal, target_length=2500, target_leads=12):
    """
    Improved signal preprocessing with filtering and normalization.
    """
    # Handle missing or invalid signals
    if signal is None or signal.size == 0:
        return np.zeros((target_length, target_leads))
    
    # Ensure signal is 2D
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)
    
    # Resample to target length
    if signal.shape[0] != target_length:
        signal = resample(signal, target_length, axis=0)
    
    # Handle lead count
    if signal.shape[1] > target_leads:
        # Take first 12 leads
        signal = signal[:, :target_leads]
    elif signal.shape[1] < target_leads:
        # Pad with zeros or repeat leads
        if signal.shape[1] > 0:
            # Repeat existing leads cyclically
            repeats = target_leads // signal.shape[1]
            remainder = target_leads % signal.shape[1]
            padded = np.tile(signal, (1, repeats))
            if remainder > 0:
                padded = np.concatenate([padded, signal[:, :remainder]], axis=1)
            signal = padded
        else:
            signal = np.zeros((target_length, target_leads))
    
    # Apply basic filtering to remove noise
    try:
        # Simple bandpass filter (0.5-40 Hz for ECG)
        fs = 500  # Assumed sampling frequency
        nyquist = fs / 2
        low = 0.5 / nyquist
        high = 40.0 / nyquist
        
        if low < 1.0 and high < 1.0:
            b, a = butter(3, [low, high], btype='band')
            for i in range(signal.shape[1]):
                signal[:, i] = filtfilt(b, a, signal[:, i])
    except:
        pass  # Skip filtering if it fails
    
    # Remove baseline drift (simple high-pass)
    try:
        for i in range(signal.shape[1]):
            signal[:, i] = signal[:, i] - np.mean(signal[:, i])
    except:
        pass
    
    # Clip extreme values
    signal = np.clip(signal, -10, 10)
    
    return signal

def train_model(data_directory, model_directory, verbose=False):
    """
    Train an improved model with actual data.
    """
    if verbose:
        print(f"Training improved model...")
    
    # Ensure model directory exists
    os.makedirs(model_directory, exist_ok=True)
    
    # Load training data
    signals, labels = load_challenge_data(data_directory, max_samples=1000)
    
    if signals is None or len(signals) == 0:
        if verbose:
            print("No training data found, creating dummy model...")
        # Create dummy model as fallback
        create_dummy_model(model_directory)
        return
    
    # Preprocess signals
    processed_signals = []
    valid_labels = []
    
    for i, (signal, label) in enumerate(zip(signals, labels)):
        try:
            processed_signal = preprocess_signal(signal)
            processed_signals.append(processed_signal)
            valid_labels.append(label)
        except Exception as e:
            if verbose:
                print(f"Error preprocessing signal {i}: {e}")
            continue
    
    if len(processed_signals) == 0:
        create_dummy_model(model_directory)
        return
    
    # Convert to numpy arrays
    X = np.array(processed_signals)
    y = np.array(valid_labels)
    
    if verbose:
        print(f"Training on {len(X)} samples")
        print(f"Signal shape: {X.shape}")
        print(f"Label distribution: {np.bincount(y)}")
    
    # Normalize features
    scaler = StandardScaler()
    X_reshaped = X.reshape(X.shape[0], -1)
    X_scaled = scaler.fit_transform(X_reshaped)
    X_scaled = X_scaled.reshape(X.shape)
    
    # Build improved model with better architecture
    model = Sequential([
        # First conv block
        Conv1D(32, kernel_size=7, activation='relu', input_shape=(2500, 12)),
        BatchNormalization(),
        MaxPooling1D(pool_size=3),
        
        # Second conv block
        Conv1D(64, kernel_size=5, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=3),
        
        # Third conv block
        Conv1D(64, kernel_size=3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        
        # Dense layers
        Flatten(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    
    # Compile with better optimizer settings
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    # Train with early stopping and class balancing
    from sklearn.utils.class_weight import compute_class_weight
    
    try:
        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
    except:
        class_weight_dict = None
    
    # Simple train/validation split
    split_idx = int(0.8 * len(X_scaled))
    X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,  # Reduced for memory efficiency
        batch_size=16,  # Small batch size
        class_weight=class_weight_dict,
        verbose=1 if verbose else 0
    )
    
    # Save model
    model.save(os.path.join(model_directory, 'model.h5'), include_optimizer=False)
    
    # Save scaler
    np.save(os.path.join(model_directory, 'scaler_mean.npy'), scaler.mean_)
    np.save(os.path.join(model_directory, 'scaler_scale.npy'), scaler.scale_)
    
    # Save signal dimensions
    np.save(os.path.join(model_directory, 'signal_dims.npy'), np.array([2500, 12]))
    
    if verbose:
        print(f"Model training completed. Final validation accuracy: {history.history['val_accuracy'][-1]:.3f}")
    
    return

def create_dummy_model(model_directory):
    """Create a simple dummy model as fallback."""
    model = Sequential([
        Conv1D(16, kernel_size=5, activation='relu', input_shape=(2500, 12)),
        MaxPooling1D(pool_size=4),
        Conv1D(32, kernel_size=5, activation='relu'),
        MaxPooling1D(pool_size=4),
        Flatten(),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.save(os.path.join(model_directory, 'model.h5'), include_optimizer=False)
    
    # Create dummy scaler
    scaler_mean = np.zeros(30000)
    scaler_scale = np.ones(30000)
    np.save(os.path.join(model_directory, 'scaler_mean.npy'), scaler_mean)
    np.save(os.path.join(model_directory, 'scaler_scale.npy'), scaler_scale)
    np.save(os.path.join(model_directory, 'signal_dims.npy'), np.array([2500, 12]))

def load_model(model_directory, verbose=False):
    """
    Load the trained model and preprocessing components.
    """
    if verbose:
        print(f"Loading model from {model_directory}")
    
    # Load model
    model_path = os.path.join(model_directory, 'model.h5')
    model = tf.keras.models.load_model(model_path, compile=False)
    
    # Recompile
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Load scaler
    scaler = StandardScaler()
    scaler.mean_ = np.load(os.path.join(model_directory, 'scaler_mean.npy'))
    scaler.scale_ = np.load(os.path.join(model_directory, 'scaler_scale.npy'))
    
    # Load signal dimensions
    signal_dims = np.load(os.path.join(model_directory, 'signal_dims.npy'))
    
    return {
        'model': model, 
        'scaler': scaler,
        'signal_dims': signal_dims
    }

def run_model(record, model_data, verbose=False):
    """
    Run model on a record with improved preprocessing.
    """
    try:
        if verbose:
            print(f"Processing record: {record}")
        
        # Extract model components
        model = model_data['model']
        scaler = model_data['scaler']
        signal_dims = model_data.get('signal_dims', np.array([2500, 12]))
        target_length, num_leads = int(signal_dims[0]), int(signal_dims[1])
        
        # Load signal
        try:
            signal, _ = wfdb.rdsamp(record)
        except Exception as e:
            if verbose:
                print(f"Error loading signal: {e}")
            signal = np.zeros((target_length, num_leads))
        
        # Preprocess signal with improved function
        processed_signal = preprocess_signal(signal, target_length, num_leads)
        
        # Normalize using saved scaler
        signal_flat = processed_signal.reshape(1, -1)
        signal_scaled = scaler.transform(signal_flat)
        final_signal = signal_scaled.reshape(1, target_length, num_leads)
        
        # Make prediction
        with tf.device('/cpu:0'):
            probability = float(model.predict(final_signal, verbose=0, batch_size=1)[0][0])
        
        # Use a more balanced threshold
        binary_prediction = 1 if probability >= 0.4 else 0
        
        return binary_prediction, probability
        
    except Exception as e:
        if verbose:
            print(f"Error in run_model: {e}")
        return 0, 0.0
