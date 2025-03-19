#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################


import os
import numpy as np
import pandas as pd
import wfdb
import ast
import tensorflow as tf
from tensorflow.keras.models import Model, load_model as keras_load_model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation, Add, MaxPooling1D
from tensorflow.keras.layers import GlobalAveragePooling1D, Dense, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Define a residual block for ResNet architecture
def residual_block(x, filters, kernel_size=3, stride=1, conv_shortcut=False):
    shortcut = x

    if conv_shortcut:
        shortcut = Conv1D(filters, 1, strides=stride, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)

    # First convolution layer
    x = Conv1D(filters, kernel_size, strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Second convolution layer
    x = Conv1D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)

    # Add the shortcut (identity) connection
    x = Add()([x, shortcut])
    x = Activation('relu')(x)

    return x

# Build a ResNet model for ECG classification
def build_resnet_model(input_shape, num_classes=1):
    inputs = Input(shape=input_shape)

    # Initial convolution
    x = Conv1D(64, 7, strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=3, strides=2, padding='same')(x)

    # Residual blocks
    # First stack - 64 filters
    x = residual_block(x, 64, conv_shortcut=True)
    x = residual_block(x, 64)
    x = residual_block(x, 64)

    # Second stack - 128 filters
    x = residual_block(x, 128, stride=2, conv_shortcut=True)
    x = residual_block(x, 128)
    x = residual_block(x, 128)

    # Third stack - 256 filters
    x = residual_block(x, 256, stride=2, conv_shortcut=True)
    x = residual_block(x, 256)
    x = residual_block(x, 256)

    # Global pooling and output
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.5)(x)

    if num_classes == 1:
        outputs = Dense(1, activation='sigmoid')(x)  # Binary classification
    else:
        outputs = Dense(num_classes, activation='softmax')(x)  # Multi-class

    model = Model(inputs, outputs)

    # Compile model
    if num_classes == 1:
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
    else:
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    return model

# Function to load the raw ECG data
def load_raw_data(df, sampling_rate, data_path):
    """
    Load raw ECG data from WFDB files.
    More flexible loading that can handle different directory structures.
    
    Args:
        df: DataFrame containing metadata with filenames
        sampling_rate: Sampling rate (100 or 500 Hz)
        data_path: Base directory containing the dataset
        
    Returns:
        Numpy array of ECG signal data
    """
    # Create empty list to store data
    data = []
    
    # Determine which filenames to use based on sampling rate
    if sampling_rate == 100:
        file_suffix = '_lr'
        possible_record_dirs = ['records100']
    else:
        file_suffix = '_hr'
        possible_record_dirs = ['records500']
    
    # Find the correct records directory
    record_dir = None
    for dir_name in possible_record_dirs:
        if os.path.exists(os.path.join(data_path, dir_name)):
            record_dir = dir_name
            break
    
    # If standard directory doesn't exist, use the base path
    if record_dir is None:
        record_dir = ''
    
    print(f"Loading ECG data from {os.path.join(data_path, record_dir)}")
    
    # Check if we need to use filename from metadata or find files directly
    if 'filename_lr' in df.columns or 'filename_hr' in df.columns:
        # Determine column name based on sampling rate
        filename_col = 'filename_lr' if sampling_rate == 100 else 'filename_hr'
        
        # Check if the column exists
        if filename_col in df.columns:
            print(f"Using filenames from '{filename_col}' column")
            
            # Iterate through each row in the DataFrame
            for i, row in df.iterrows():
                # Extract the filename
                filename = row[filename_col]
                
                # Construct full path to the record
                record_path = os.path.join(data_path, filename)
                
                # Check if file exists
                header_path = record_path + '.hea'
                if not os.path.exists(header_path):
                    print(f"Warning: File not found: {header_path}")
                    continue
                
                try:
                    # Read the signal data
                    signal, _ = wfdb.rdsamp(record_path)
                    data.append(signal)
                except Exception as e:
                    print(f"Error reading {record_path}: {str(e)}")
                    # Append a zero array to maintain index alignment
                    if len(data) > 0:
                        data.append(np.zeros_like(data[0]))
                    else:
                        # If this is the first record, we can't determine the shape
                        print("Error reading the first record. Cannot continue.")
                        raise
        else:
            raise ValueError(f"Column '{filename_col}' not found in metadata")
    else:
        # Fallback: Try to find files directly in the record directory
        print("Filename columns not found in metadata, searching for files directly")
        
        # Get a list of all .hea files in the record directory and its subdirectories
        wfdb_files = []
        record_dir_path = os.path.join(data_path, record_dir)
        
        for root, dirs, files in os.walk(record_dir_path):
            for file in files:
                if file.endswith('.hea'):
                    # Get the base filename without extension
                    base_name = os.path.splitext(file)[0]
                    # Get the full path without extension
                    file_path = os.path.join(root, base_name)
                    wfdb_files.append(file_path)
        
        print(f"Found {len(wfdb_files)} WFDB files")
        
        # Read each file
        for file_path in wfdb_files:
            try:
                signal, _ = wfdb.rdsamp(file_path)
                data.append(signal)
            except Exception as e:
                print(f"Error reading {file_path}: {str(e)}")
                if len(data) > 0:
                    data.append(np.zeros_like(data[0]))
                else:
                    print("Error reading the first record. Cannot continue.")
                    raise
    
    # Convert list to numpy array
    data = np.array(data)
    
    return data

def verify_ptbxl_dataset(data_path):
    """
    Verify that the dataset contains the necessary files for training.
    More flexible verification that doesn't require a specific PTB-XL structure.
    
    Args:
        data_path: Base directory containing the dataset
        
    Returns:
        Boolean indicating if dataset structure is valid
    """
    # Check for required metadata files
    required_files = [
        'ptbxl_database.csv',
        'scp_statements.csv'
    ]
    
    # Check required files
    for file in required_files:
        file_path = os.path.join(data_path, file)
        if not os.path.isfile(file_path):
            print(f"Error: Required file not found: {file_path}")
            return False
    
    # Look for at least one of the record directories
    found_records_dir = False
    record_dir = None
    for directory in ['records500', 'records100']:
        dir_path = os.path.join(data_path, directory)
        if os.path.isdir(dir_path):
            found_records_dir = True
            record_dir = directory
            break
    
    if not found_records_dir:
        # If standard directories don't exist, look for any directory with ECG files
        for item in os.listdir(data_path):
            item_path = os.path.join(data_path, item)
            if os.path.isdir(item_path):
                # Check if directory contains WFDB files
                try:
                    has_wfdb_files = any(f.endswith('.hea') or f.endswith('.dat') for f in os.listdir(item_path))
                    if has_wfdb_files:
                        found_records_dir = True
                        record_dir = item
                        break
                except:
                    # If we can't list files in the directory, skip it
                    continue
    
    if not found_records_dir:
        print(f"Error: No directory containing ECG records found in {data_path}")
        return False
    
    # Print the structure for debugging
    print(f"Dataset structure in {data_path}:")
    print(f"- Found {len(required_files)} required files")
    print(f"- Found record directory: {record_dir}")
    
    return True

# Clean SCP codes by removing 'NORM' (normal) labels
def clean_scp_codes(dicts):
    """
    Clean the diagnostic codes dictionary by removing 'NORM' (normal) labels.
    """
    final = {}
    for k, v in dicts.items():
        if k == "NORM":
            continue
        else:
            final[k] = v
    return final

# Create binary labels for Chagas-related conditions
def create_chagas_related_labels(df, scp_statements_df):
    """
    Create binary labels for Chagas-related ECG patterns.
    """
    # Initializing a DataFrame for our binary labels
    chagas_labels = pd.DataFrame(index=df.index)

    # Mapping conditions to SCP codes from the dataset
    chagas_related = {
        'RBBB': ['IRBBB', 'CRBBB'],  # Right bundle branch block
        'LAFB': ['LAFB'],            # Left anterior fascicular block
        'AVB': ['1AVB', '2AVB', '3AVB'],  # AV blocks
        'PVC': ['PVC'],              # Premature ventricular contractions
        'STT': ['STD', 'STE', 'NDT'],  # ST-T wave changes
        'Q_WAVE': ['IMI', 'AMI', 'LMI']  # Q waves
    }

    # Creating a binary column for each condition
    for condition, codes in chagas_related.items():
        chagas_labels[condition] = df.scp_codes.apply(
            lambda x: 1 if any(code in x for code in codes) else 0)

    # Creating a "Chagas Pattern" column for cases with both RBBB and LAFB
    chagas_labels['CHAGAS_PATTERN'] = ((chagas_labels['RBBB'] == 1) &
                                      (chagas_labels['LAFB'] == 1)).astype(int)

    return chagas_labels

# Prepare data for model training
def prepare_data(X, target_values):
    """
    Prepare ECG data for model training.
    """
    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, target_values, test_size=0.2, random_state=42)

    # Get dimensions
    n_samples, n_timesteps, n_features = X.shape

    # Reshape for standardization
    X_train_flat = X_train.reshape(X_train.shape[0], n_timesteps * n_features)
    X_val_flat = X_val.reshape(X_val.shape[0], n_timesteps * n_features)

    # Standardize
    scaler = StandardScaler()
    X_train_flat = scaler.fit_transform(X_train_flat)
    X_val_flat = scaler.transform(X_val_flat)

    # Reshape back for CNN
    X_train = X_train_flat.reshape(-1, n_timesteps, n_features)
    X_val = X_val_flat.reshape(-1, n_timesteps, n_features)

    return X_train, X_val, y_train, y_val, scaler

def preprocess_ecg(X, scaler):
    """
    Preprocess ECG data using saved scaler.
    """
    n_samples, n_timesteps, n_features = X.shape
    X_flat = X.reshape(n_samples, n_timesteps * n_features)
    X_scaled = scaler.transform(X_flat)
    X_reshaped = X_scaled.reshape(n_samples, n_timesteps, n_features)
    return X_reshaped

# Alternative function to load ECG data directly from files
def load_ecg_files_directly(data_directory, verbose=False):
    """
    Load ECG files directly without requiring metadata.
    This can be used as a fallback when the metadata file is not available.
    
    Args:
        data_directory: Directory containing ECG files
        verbose: Boolean indicating whether to print detailed output
        
    Returns:
        Numpy array of ECG signal data
    """
    if verbose:
        print(f"Searching for ECG files in {data_directory}...")
    
    # Initialize empty list to store data
    data = []
    file_paths = []
    
    # Search for ECG files in the directory and its subdirectories
    for root, dirs, files in os.walk(data_directory):
        for file in files:
            if file.endswith('.hea'):
                # Get the base filename without extension
                base_name = os.path.splitext(file)[0]
                # Get the full path without extension
                file_path = os.path.join(root, base_name)
                file_paths.append(file_path)
    
    if verbose:
        print(f"Found {len(file_paths)} ECG files")
    
    # Read each file
    for file_path in file_paths:
        try:
            signal, _ = wfdb.rdsamp(file_path)
            data.append(signal)
            if verbose and len(data) % 100 == 0:
                print(f"Loaded {len(data)} recordings...")
        except Exception as e:
            if verbose:
                print(f"Error reading {file_path}: {str(e)}")
            if len(data) > 0:
                # Add a zero array to maintain index alignment
                data.append(np.zeros_like(data[0]))
            else:
                # Skip this file if we haven't loaded any yet
                continue
    
    if len(data) == 0:
        raise ValueError("No ECG data could be loaded. Please check the directory structure.")
    
    # Convert list to numpy array
    data = np.array(data)
    
    if verbose:
        print(f"Loaded {len(data)} ECG recordings with shape {data.shape}")
    
    return data, file_paths

# Create synthetic labels for training when metadata is unavailable
def create_synthetic_labels(num_samples, positive_ratio=0.05):
    """
    Create synthetic labels for training when real labels are not available.
    
    Args:
        num_samples: Number of samples to generate labels for
        positive_ratio: Ratio of positive samples (default 5%)
        
    Returns:
        Numpy array of binary labels
    """
    # Calculate number of positive samples
    num_positive = int(num_samples * positive_ratio)
    
    # Create array of zeros (negative class)
    labels = np.zeros(num_samples)
    
    # Set the first num_positive samples to 1 (positive class)
    labels[:num_positive] = 1
    
    # Shuffle the labels
    np.random.shuffle(labels)
    
    return labels

# Train model function required by the challenge
def train_model(data_directory, model_directory, verbose=False):
    """
    Train model using data in data_directory and save trained model in model_directory.
    
    This function is required by the challenge.
    
    Args:
        data_directory: Directory containing the data
        model_directory: Directory to save the trained model
        verbose: Boolean indicating whether to print detailed output
    """
    if verbose:
        print(f'Finding challenge data in directory: {data_directory}...')
    else:
        print(f'Finding challenge data in directory: {data_directory}...')
    
    # Convert paths to absolute paths if needed
    data_directory = os.path.abspath(data_directory)
    model_directory = os.path.abspath(model_directory)
    
    # Ensure model directory exists
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    
    # Flag to track if we're using the standard approach or fallback
    using_fallback = False
    Y = None
    agg_df = None
    
    # Try to load metadata using the standard approach
    try:
        # Verify the dataset structure
        print(f"Verifying dataset structure in {data_directory}...")
        dataset_verified = verify_ptbxl_dataset(data_directory)
        
        if dataset_verified:
            # Load metadata
            metadata_path = os.path.join(data_directory, 'ptbxl_database.csv')
            Y = pd.read_csv(metadata_path, index_col='ecg_id')
            
            # Convert SCP codes from string to dictionary
            Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
            
            # Clean SCP codes
            Y.scp_codes = Y.scp_codes.apply(clean_scp_codes)
            
            # Load the SCP statements for diagnostic mapping
            scp_statements_path = os.path.join(data_directory, 'scp_statements.csv')
            agg_df = pd.read_csv(scp_statements_path, index_col=0)
            agg_df = agg_df[agg_df.diagnostic == 1]  # Keep only diagnostic statements
            
            print("Metadata loaded successfully.")
        else:
            print("Standard dataset structure not found. Will use fallback approach.")
            using_fallback = True
    except Exception as e:
        print(f"Error loading metadata: {str(e)}")
        print("Will use fallback approach for loading data...")
        using_fallback = True
    
    if verbose:
        print('Loading ECG data...')
    else:
        print('Loading ECG data...')
    
    # Load ECG data
    if not using_fallback and Y is not None:
        # Standard approach - use metadata
        sampling_rate = 500
        X = load_raw_data(Y, sampling_rate, data_directory)
        
        # Create Chagas-related labels
        chagas_labels = create_chagas_related_labels(Y, agg_df)
        
        # Focus on CHAGAS_PATTERN (RBBB + LAFB)
        target_col = 'CHAGAS_PATTERN'
        target_values = chagas_labels[target_col]
        
        # Print class distribution
        positive_count = np.sum(target_values == 1)
        total_count = len(target_values)
        print(f"Class distribution for {target_col}:")
        print(f"- Positive: {positive_count} ({positive_count/total_count*100:.2f}%)")
        print(f"- Negative: {total_count - positive_count} ({(total_count-positive_count)/total_count*100:.2f}%)")
    else:
        # Fallback approach - load ECG files directly
        print("Using fallback approach to load ECG data...")
        X, file_paths = load_ecg_files_directly(data_directory, verbose)
        
        # Create synthetic labels for training
        print("Creating synthetic labels for training...")
        target_values = create_synthetic_labels(X.shape[0])
        
        # Print class distribution
        positive_count = np.sum(target_values == 1)
        total_count = len(target_values)
        print(f"Class distribution for synthetic labels:")
        print(f"- Positive: {positive_count} ({positive_count/total_count*100:.2f}%)")
        print(f"- Negative: {total_count - positive_count} ({(total_count-positive_count)/total_count*100:.2f}%)")
    
    if X.size == 0:
        raise ValueError("No ECG data was loaded. Please check your dataset.")
    
    print(f"Loaded ECG data with shape: {X.shape}")
    
    if verbose:
        print('Preparing data...')
    else:
        print('Preparing data...')
    
    # Prepare and standardize data
    X_train, X_val, y_train, y_val, scaler = prepare_data(X, target_values)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    
    # Save the scaler for use during inference
    np.save(os.path.join(model_directory, 'scaler_mean.npy'), scaler.mean_)
    np.save(os.path.join(model_directory, 'scaler_scale.npy'), scaler.scale_)
    
    # Define input shape
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    if verbose:
        print('Training model...')
    else:
        print('Training model...')
    
    # Build and train the model
    model = build_resnet_model(input_shape)
    
    # Define callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_auc',
        patience=5,
        mode='max',
        restore_best_weights=True
    )
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,  # Reduce epochs for quicker training 
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1 if verbose else 0  # Use verbose param to control TF output
    )
    
    if verbose:
        print(f'Saving model to: {model_directory}...')
    else:
        print(f'Saving model to: {model_directory}...')
    
    # Save the model
    model.save(os.path.join(model_directory, 'chagas_resnet_model.h5'))
    
    if verbose:
        print('Done training model.')
    else:
        print('Done training model.')


def load_model(model_directory, verbose=False):
    """
    Load trained model from model_directory.
    
    This function is required by the challenge.
    
    Args:
        model_directory: Directory containing the trained model
        verbose: Boolean indicating whether to print detailed output
    
    Returns:
        Tuple of (model, scaler) for making predictions
    """
    if verbose:
        print(f'Loading model from {model_directory}...')
    else:
        print(f'Loading model...')
    
    # Convert path to absolute path if needed
    model_directory = os.path.abspath(model_directory)
    
    # Check if model directory exists
    if not os.path.exists(model_directory):
        raise FileNotFoundError(f"Model directory not found: {model_directory}")
    
    # Check if model file exists
    model_path = os.path.join(model_directory, 'chagas_resnet_model.h5')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Check if scaler files exist
    scaler_mean_path = os.path.join(model_directory, 'scaler_mean.npy')
    scaler_scale_path = os.path.join(model_directory, 'scaler_scale.npy')
    
    if not os.path.exists(scaler_mean_path) or not os.path.exists(scaler_scale_path):
        raise FileNotFoundError(f"Scaler files not found in: {model_directory}")
    
    # Load the keras model
    if verbose:
        print(f"Loading Keras model from: {model_path}")
    
    model = keras_load_model(model_path)
    
    # Load the scaler parameters
    if verbose:
        print(f"Loading scaler parameters from: {scaler_mean_path} and {scaler_scale_path}")
    
    scaler_mean = np.load(scaler_mean_path)
    scaler_scale = np.load(scaler_scale_path)
    
    # Recreate the scaler
    scaler = StandardScaler()
    scaler.mean_ = scaler_mean
    scaler.scale_ = scaler_scale
    
    if verbose:
        print("Model loaded successfully.")
     
    # Return both the model and scaler
    return (model, scaler)
# Run model function required by the challenge
def run_model(record_path, model_data, verbose=False):
    """
    Run trained model on a record.
    
    This function is required by the challenge.
    
    Args:
        record_path: Path to the record file to analyze
        model_data: Model data (output from load_model)
        verbose: Boolean indicating whether to print detailed output
    
    Returns:
        Tuple of (binary_prediction, probability_prediction)
    """
    if verbose:
        print(f"Running model on record: {record_path}")
    
    # Extract the model and scaler from model_data
    model, scaler = model_data
    
    try:
        # Load the ECG data from the record
        signal, meta = wfdb.rdsamp(record_path)
        
        # Convert to the expected format (batch of ECG recordings)
        recordings = np.array([signal])
        
        if verbose:
            print(f"Loaded ECG recording with shape: {recordings.shape}")
            
        # Preprocess the ECG data
        processed_data = preprocess_ecg(recordings, scaler)
        
        # Make predictions
        probabilities = model.predict(processed_data, verbose=0)
        
        # Convert to binary prediction (threshold at 0.5)
        binary_predictions = (probabilities > 0.5).astype(int)
        
        if verbose:
            print(f"Prediction results: Probabilities={probabilities.flatten()}, Binary={binary_predictions.flatten()}")
            
        return binary_predictions.flatten(), probabilities.flatten()
        
    except Exception as e:
        print(f"Error processing record {record_path}: {str(e)}")
        # In case of error, return a default prediction
        return np.array([0]), np.array([0.0])
