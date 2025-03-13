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
    """
    if sampling_rate == 100:
        data = [wfdb.rdsamp(os.path.join(data_path, f)) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(os.path.join(data_path, f)) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data

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

# Train model function required by the challenge
def train_model(data_directory, model_directory):
    """
    Train model using data in data_directory and save trained model in model_directory.
    
    This function is required by the challenge.
    """
    print('Finding challenge data...')
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
    
    print('Loading ECG data...')
    # Load raw ECG data (using 100Hz sampling rate)
    sampling_rate = 100
    X = load_raw_data(Y, sampling_rate, data_directory)
    
    # Create Chagas-related labels
    chagas_labels = create_chagas_related_labels(Y, agg_df)
    
    # Focus on CHAGAS_PATTERN (RBBB + LAFB)
    target_col = 'CHAGAS_PATTERN'
    target_values = chagas_labels[target_col]
    
    print('Preparing data...')
    # Prepare and standardize data
    X_train, X_val, y_train, y_val, scaler = prepare_data(X, target_values)
    
    # Save the scaler for use during inference
    np.save(os.path.join(model_directory, 'scaler_mean.npy'), scaler.mean_)
    np.save(os.path.join(model_directory, 'scaler_scale.npy'), scaler.scale_)
    
    # Define input shape
    input_shape = (X_train.shape[1], X_train.shape[2])
    
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
        epochs=10,  # Reduce epochs for quicker training 
        batch_size=32,
        callbacks=[early_stopping]
    )
    
    print('Saving model...')
    # Save the model
    model.save(os.path.join(model_directory, 'chagas_resnet_model.h5'))
    
    print('Done training model.')

# Load model function required by the challenge
def load_model(model_directory):
    """
    Load trained model from model_directory.
    
    This function is required by the challenge.
    """
    print('Loading model...')
    
    # Load the keras model
    model_path = os.path.join(model_directory, 'chagas_resnet_model.h5')
    model = keras_load_model(model_path)
    
    # Load the scaler parameters
    scaler_mean = np.load(os.path.join(model_directory, 'scaler_mean.npy'))
    scaler_scale = np.load(os.path.join(model_directory, 'scaler_scale.npy'))
    
    # Recreate the scaler
    scaler = StandardScaler()
    scaler.mean_ = scaler_mean
    scaler.scale_ = scaler_scale
    
    # Return both the model and scaler
    return model, scaler

# Run model function required by the challenge
def run_model(model, data, recordings):
    """
    Run trained model on data.
    
    This function is required by the challenge.
    """
    # Unpack the model and scaler
    keras_model, scaler = model
    
    # Preprocess the ECG data
    X_processed = preprocess_ecg(recordings, scaler)
    
    # Make predictions
    y_pred_prob = keras_model.predict(X_processed)
    
    # Convert to binary prediction (threshold at 0.5)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    # Return both binary prediction and probability
    return y_pred.flatten(), y_pred_prob.flatten()
