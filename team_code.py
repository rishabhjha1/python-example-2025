#!/usr/bin/env python

# Required packages:
# pip install numpy pandas wfdb tensorflow scikit-learn joblib

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
import joblib
import glob
from scipy.signal import resample

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

# Find records in a directory (adapted for challenge structure)
def find_records(data_directory):
    # Look for header files in the data directory
    records = []
    
    # First look for .hea files directly in the data directory
    for root, _, files in os.walk(data_directory):
        for file in files:
            if file.endswith('.hea'):
                # Get the record name without the .hea extension
                record_name = os.path.splitext(file)[0]
                records.append(os.path.join(root, record_name))
    
    return records

# Function to load header from a record
def load_header(record):
    try:
        return wfdb.rdheader(record)
    except Exception as e:
        print(f"Error loading header for record {record}: {e}")
        return None

# Function to extract age from header
def get_age(header):
    if header and hasattr(header, 'comments'):
        for comment in header.comments:
            if comment.startswith('Age:'):
                try:
                    age = float(comment.split(':')[1].strip())
                    return age
                except:
                    return float('nan')
    return float('nan')

# Function to extract sex from header
def get_sex(header):
    if header and hasattr(header, 'comments'):
        for comment in header.comments:
            if comment.split(':')[0].strip() == 'Sex':
                sex = comment.split(':')[1].strip()
                return sex
    return 'Unknown'

# Function to load signals from a record
def load_signals(record):
    try:
        signals, fields = wfdb.rdsamp(record)
        return signals, fields
    except Exception as e:
        print(f"Error loading signals for record {record}: {e}")
        return np.array([]), {}

# Function to check if a record has a Chagas label
def load_label(record):
    header = load_header(record)
    if header and hasattr(header, 'comments'):
        for comment in header.comments:
            if comment.startswith('Chagas:'):
                label_str = comment.split(':')[1].strip().lower()
                return label_str == 'true' or label_str == '1'
    return False

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
def create_chagas_related_labels(df, scp_statements_df=None):
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

    # Check if scp_codes column exists
    if 'scp_codes' in df.columns:
        # Creating a binary column for each condition
        for condition, codes in chagas_related.items():
            chagas_labels[condition] = df.scp_codes.apply(
                lambda x: 1 if any(code in x for code in codes) else 0)

        # Creating a "Chagas Pattern" column for cases with both RBBB and LAFB
        chagas_labels['CHAGAS_PATTERN'] = ((chagas_labels['RBBB'] == 1) &
                                         (chagas_labels['LAFB'] == 1)).astype(int)
    else:
        # If SCP codes are not available, try to look for Chagas directly
        chagas_labels['CHAGAS_PATTERN'] = 0  # Default
        
        # If the dataframe contains a Chagas column, use it
        if 'chagas' in df.columns:
            chagas_labels['CHAGAS_PATTERN'] = df['chagas'].astype(int)

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

# Function to load ECG data from WFDB files
def load_ecg_data(records):
    """
    Load ECG data from a list of WFDB records and resample to consistent length.
    
    Args:
        records: List of record paths
        
    Returns:
        Numpy array of ECG signal data with consistent dimensions
    """
    # Create empty list to store data
    data = []
    valid_records = []
    target_length = 5000  # Target length for resampling (adjust as needed)
    
    print(f"Total records to load: {len(records)}")
    
    # Get sample dimensions from first record to understand data shape
    if records:
        try:
            signal, meta = wfdb.rdsamp(records[0])
            print(f"Sample signal shape: {signal.shape}")
            print(f"Sample metadata: {meta}")
        except Exception as e:
            print(f"Error examining first record: {e}")
    
    # Load each record
    for i, record_path in enumerate(records):
        if i % 100 == 0:
            print(f"Loading record {i+1}/{len(records)}")
            
        try:
            # Read the signal data
            signal, _ = wfdb.rdsamp(record_path)
            
            # Check if we need to resample
            if signal.shape[0] != target_length:
                # Resample to target length
                resampled_signal = resample(signal, target_length)
                data.append(resampled_signal)
            else:
                data.append(signal)
                
            valid_records.append(record_path)
        except Exception as e:
            print(f"Error reading {record_path}: {str(e)}")
            # Skip this record
            continue
    
    print(f"Successfully loaded {len(data)} records")
    
    # Convert list to numpy array
    try:
        data_array = np.array(data)
        print(f"Data array shape: {data_array.shape}")
        return data_array, valid_records
    except ValueError as e:
        print(f"Error creating numpy array: {e}")
        print("Checking signal shapes...")
        shapes = [signal.shape for signal in data[:5]]  # Print first 5 shapes
        print(f"Sample shapes: {shapes}")
        raise

# Function to load the raw ECG data for PTB-XL
def load_raw_data_ptbxl(df, sampling_rate, data_path):
    """
    Load raw ECG data from WFDB files in the PTB-XL dataset.
    
    Args:
        df: DataFrame containing metadata with filenames
        sampling_rate: Sampling rate (100 or 500 Hz)
        data_path: Base directory containing the PTB-XL dataset
        
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

# Detect dataset type
def detect_dataset_type(data_directory):
    """
    Detect if the dataset is PTB-XL or SaMi-Trop based on directory structure.
    
    Args:
        data_directory: Directory containing the dataset
        
    Returns:
        String indicating the dataset type ("ptbxl", "samitrop", or "unknown")
    """
    # Check for PTB-XL specific files
    ptbxl_meta_path = os.path.join(data_directory, 'ptbxl_database.csv')
    ptbxl_statements_path = os.path.join(data_directory, 'scp_statements.csv')
    
    # Check for PTB-XL record directories
    ptbxl_records100_path = os.path.join(data_directory, 'records100')
    ptbxl_records500_path = os.path.join(data_directory, 'records500')
    
    # Check for SaMi-Trop structure - typically just WFDB files after preprocessing
    samitrop_wfdb_files = False
    for root, _, files in os.walk(data_directory):
        if any(f.endswith('.hea') for f in files):
            samitrop_wfdb_files = True
            break
    
    # Determine dataset type
    if os.path.exists(ptbxl_meta_path) and os.path.exists(ptbxl_statements_path):
        return "ptbxl"
    elif samitrop_wfdb_files and not (os.path.exists(ptbxl_records100_path) or os.path.exists(ptbxl_records500_path)):
        return "samitrop"
    else:
        return "unknown"

# Helper function for extracting metadata from records
def extract_metadata_from_records(records):
    """
    Extract metadata from record headers and create a DataFrame.
    
    Args:
        records: List of record paths
        
    Returns:
        DataFrame with metadata and record paths
    """
    metadata = {
        'record_path': [],
        'age': [],
        'sex': [],
        'chagas': []
    }
    
    for record in records:
        metadata['record_path'].append(record)
        
        # Load header
        header = load_header(record)
        
        # Extract age
        metadata['age'].append(get_age(header))
        
        # Extract sex
        metadata['sex'].append(get_sex(header))
        
        # Extract Chagas label
        metadata['chagas'].append(load_label(record))
    
    # Convert to DataFrame
    return pd.DataFrame(metadata)

def debug_data_directory(data_directory):
    """
    Print detailed information about the data directory structure for debugging.
    """
    print(f"\n*** DEBUG: Examining data directory: {data_directory} ***")
    
    # Check if directory exists
    if not os.path.exists(data_directory):
        print(f"ERROR: Data directory does not exist: {data_directory}")
        return
    
    # List all files in the directory
    print("\nFiles directly in the data directory:")
    for item in os.listdir(data_directory):
        item_path = os.path.join(data_directory, item)
        if os.path.isfile(item_path):
            print(f"  File: {item} ({os.path.getsize(item_path)} bytes)")
        elif os.path.isdir(item_path):
            print(f"  Directory: {item}")
            
    # Go through subdirectories and check for WFDB files
    print("\nSearching for WFDB files (.hea, .dat):")
    wfdb_files_found = False
    
    for root, dirs, files in os.walk(data_directory):
        hea_files = [f for f in files if f.endswith('.hea')]
        if hea_files:
            wfdb_files_found = True
            rel_path = os.path.relpath(root, data_directory)
            print(f"  Found {len(hea_files)} .hea files in: {rel_path if rel_path != '.' else 'root directory'}")
            if len(hea_files) > 0:
                print(f"    Sample files: {', '.join(hea_files[:3])}")
                
    if not wfdb_files_found:
        print("  No WFDB files found in any subdirectory!")
    
    # Check if this might be a PhysioNet challenge dataset
    print("\nChecking for common PhysioNet challenge files:")
    common_files = ['ptbxl_database.csv', 'scp_statements.csv', 'exams.csv', 'samitrop_chagas_labels.csv']
    for file in common_files:
        file_path = os.path.join(data_directory, file)
        if os.path.exists(file_path):
            print(f"  Found: {file}")
    
    print("\n*** End of debug information ***\n")

# Train model function required by the challenge
def train_model(data_directory, model_directory, verbose=False):
    """
    Train model using data in data_directory and save trained model in model_directory.
    
    This function is required by the challenge.
    
    Args:
        data_directory: Directory containing the training data
        model_directory: Directory to save the trained model
        verbose: Boolean indicating whether to print detailed output
    """

    print(f'Finding challenge data in directory: {data_directory}...')
    
    # Debug: Check if directory exists
    print(f"Directory exists: {os.path.exists(data_directory)}")
    if os.path.exists(data_directory):
        print("Contents of directory:")
        for item in os.listdir(data_directory):
            print(f"  - {item}")
    
    # DEBUG: Print detailed info about the data directory
    debug_data_directory(data_directory)
    
    # Convert paths to absolute paths if needed
    data_directory = os.path.abspath(data_directory)
    model_directory = os.path.abspath(model_directory)
    
    # Ensure model directory exists
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
        
    # Detect dataset type
    dataset_type = detect_dataset_type(data_directory)
    print(f"Detected dataset type: {dataset_type}")
    
    if dataset_type == "ptbxl":
        # PTB-XL dataset handling
        print("Using PTB-XL dataset handling approach")
        
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
        
        if verbose:
            print('Loading ECG data...')
        
        # Load raw ECG data (using 500Hz sampling rate)
        sampling_rate = 500
        X = load_raw_data_ptbxl(Y, sampling_rate, data_directory)
        
        if X.size == 0:
            raise ValueError("No ECG data was loaded. Please check your dataset.")
        
        print(f"Loaded ECG data with shape: {X.shape}")
        
        # Create Chagas-related labels
        chagas_labels = create_chagas_related_labels(Y, agg_df)
        
        # Focus on CHAGAS_PATTERN (RBBB + LAFB)
        target_col = 'CHAGAS_PATTERN'
        target_values = chagas_labels[target_col]
        
    else:
        # Default to SaMi-Trop or universal approach
        print("Using universal dataset handling approach")
        
        # Find records in the data directory
        records = find_records(data_directory)
        num_records = len(records)
        
        if num_records == 0:
            raise FileNotFoundError('No data were provided.')
        
        if verbose:
            print(f'Found {num_records} records.')
        
        # Load ECG data
        if verbose:
            print('Loading ECG data...')
        
        X, valid_records = load_ecg_data(records)
        
        if X.size == 0:
            raise ValueError("No ECG data was loaded. Please check your dataset.")
        
        if verbose:
            print(f"Loaded ECG data with shape: {X.shape}")
        
        # Extract metadata from records
        if verbose:
            print('Extracting metadata...')
        
        metadata_df = extract_metadata_from_records(valid_records)
        
        # Create Chagas-related labels
        if verbose:
            print('Creating labels...')
        
        chagas_labels = create_chagas_related_labels(metadata_df)
        
        # Focus on CHAGAS_PATTERN
        target_col = 'CHAGAS_PATTERN'
        target_values = chagas_labels[target_col]
    
    # Print class distribution
    positive_count = np.sum(target_values == 1)
    total_count = len(target_values)
    
    if verbose:
        print(f"Class distribution for {target_col}:")
        print(f"- Positive: {positive_count} ({positive_count/total_count*100:.2f}%)")
        print(f"- Negative: {total_count - positive_count} ({(total_count-positive_count)/total_count*100:.2f}%)")
    
    if verbose:
        print('Preparing data...')
    
    # Prepare and standardize data
    X_train, X_val, y_train, y_val, scaler = prepare_data(X, target_values)
    
    if verbose:
        print(f"Training data shape: {X_train.shape}")
        print(f"Validation data shape: {X_val.shape}")
    
    # Save the scaler for use during inference
    np.save(os.path.join(model_directory, 'scaler_mean.npy'), scaler.mean_)
    np.save(os.path.join(model_directory, 'scaler_scale.npy'), scaler.scale_)
    
    # Define input shape
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    if verbose:
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
    
    # Save the Keras model separately using its native format
    model.save(os.path.join(model_directory, 'chagas_resnet_model.h5'))

    # Save a small marker file with model info
    with open(os.path.join(model_directory, 'model_info.txt'), 'w') as f:
        f.write(f"Model trained on {len(X_train)} samples\n")
        f.write(f"Input shape: {input_shape}\n")
        f.write(f"Positive class proportion: {positive_count/total_count*100:.2f}%\n")
        f.write(f"Dataset type: {dataset_type}\n")

    if verbose:
        print('Done training model.')

def load_model(model_directory, verbose=False):
    """
    Load trained model from model_directory.
    
    This function is required by the challenge.
    
    Args:
        model_directory: Directory containing the trained model
        verbose: Boolean indicating whether to print detailed output
    
    Returns:
        Model object for making predictions
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
    
    # Load the Keras model
    keras_model_path = os.path.join(model_directory, 'chagas_resnet_model.h5')
    if not os.path.exists(keras_model_path):
        raise FileNotFoundError(f"Model file not found: {keras_model_path}")
    
    if verbose:
        print(f"Loading Keras model from: {keras_model_path}")
    
    model = keras_load_model(keras_model_path)
    
    # Load the scaler parameters
    scaler_mean_path = os.path.join(model_directory, 'scaler_mean.npy')
    scaler_scale_path = os.path.join(model_directory, 'scaler_scale.npy')
    
    if not os.path.exists(scaler_mean_path) or not os.path.exists(scaler_scale_path):
        raise FileNotFoundError(f"Scaler files not found in: {model_directory}")
    
    if verbose:
        print(f"Loading scaler parameters")
    
    scaler_mean = np.load(scaler_mean_path)
    scaler_scale = np.load(scaler_scale_path)
    
    # Recreate the scaler
    scaler = StandardScaler()
    scaler.mean_ = scaler_mean
    scaler.scale_ = scaler_scale
    
    if verbose:
        print("Model loaded successfully.")
    
    # Return components in a dictionary
    return {'model': model, 'scaler': scaler}

# Run model function required by the challenge
def run_model(record, model_data, verbose=False):
    """
    Run trained model on a record.
    
    This function is required by the challenge.
    
    Args:
        record: Path to the record file to analyze
        model_data: Model data (output from load_model)
        verbose: Boolean indicating whether to print detailed output
    
    Returns:
        Tuple of (binary_prediction, probability_prediction)
    """
    if verbose:
        print(f"Running model on record: {record}")
    
    # Extract the model and scaler from model_data
    model = model_data.get('model')
    scaler = model_data.get('scaler')
    
    try:
        # Load the ECG data from the record
        signal, meta = wfdb.rdsamp(record)
        
        # Convert to the expected format (batch of ECG recordings)
        recordings = np.array([signal])
        
        if verbose:
            print(f"Loaded ECG recording with shape: {recordings.shape}")
        
        # Check if we need to resample to match the training data length
        target_length = 5000  # Same as in load_ecg_data function
        if recordings.shape[1] != target_length:
            if verbose:
                print(f"Resampling from {recordings.shape[1]} to {target_length} points")
            
            # Resample to target length
            resampled = np.zeros((recordings.shape[0], target_length, recordings.shape[2]))
            for i in range(recordings.shape[0]):
                resampled[i] = resample(recordings[i], target_length)
            
            recordings = resampled
        
        # Preprocess the ECG data
        processed_data = preprocess_ecg(recordings, scaler)
        
        # Make predictions
        probabilities = model.predict(processed_data, verbose=0)
        
        # Convert to binary prediction (threshold at 0.5)
        binary_predictions = (probabilities > 0.5).astype(int)
        
        if verbose:
            print(f"Prediction results: Probabilities={probabilities.flatten()}, Binary={binary_predictions.flatten()}")
        
        return binary_predictions.flatten()[0], probabilities.flatten()[0]
        
    except Exception as e:
        print(f"Error processing record {record}: {str(e)}")
        # In case of error, return a default prediction
        return 0, 0.0
