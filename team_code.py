import joblib
import numpy as np
import os
import sys
import pandas as pd
import h5py
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

from helper_code import *

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments for the functions.
#
################################################################################

def train_model(data_folder, model_folder, verbose):
    if verbose:
        print('Loading PhysioNet Challenge data...')

    # Load data from HDF5 and CSV files
    try:
        # Load exam metadata
        exams_csv_path = os.path.join(data_folder, 'exams.csv')
        labels_csv_path = os.path.join(data_folder, 'samitrop_chagas_labels.csv')
        hdf5_path = os.path.join(data_folder, 'exams.hdf5')
        
        if verbose:
            print(f'Loading exam metadata from {exams_csv_path}')
        exams_df = pd.read_csv(exams_csv_path)
        
        if verbose:
            print(f'Loading labels from {labels_csv_path}')
        labels_df = pd.read_csv(labels_csv_path)
        
        if verbose:
            print(f'Loading signal data from {hdf5_path}')
            print(f'Found {len(exams_df)} exams')
            print(f'Found {len(labels_df)} labels')
            
    except Exception as e:
        if verbose:
            print(f'Error loading CSV files: {e}')
        # Fallback to original approach
        return train_model_fallback(data_folder, model_folder, verbose)

    # Merge exams with labels
    try:
        # Merge on exam_id or appropriate key
        if 'exam_id' in exams_df.columns and 'exam_id' in labels_df.columns:
            data_df = pd.merge(exams_df, labels_df, on='exam_id', how='inner')
        elif 'id' in exams_df.columns and 'exam_id' in labels_df.columns:
            data_df = pd.merge(exams_df, labels_df, left_on='id', right_on='exam_id', how='inner')
        else:
            # Try to merge on index
            data_df = pd.merge(exams_df, labels_df, left_index=True, right_index=True, how='inner')
            
        if verbose:
            print(f'Merged data: {len(data_df)} samples')
            print(f'Columns: {list(data_df.columns)}')
            
    except Exception as e:
        if verbose:
            print(f'Error merging data: {e}')
        return train_model_fallback(data_folder, model_folder, verbose)

    # Extract features and labels
    features = []
    labels = []
    processed_count = 0
    error_count = 0
    
    # Load HDF5 file
    try:
        with h5py.File(hdf5_path, 'r') as hdf:
            if verbose:
                print(f'HDF5 keys: {list(hdf.keys())}')
            
            # Process each sample
            for idx, row in data_df.iterrows():
                try:
                    if verbose and processed_count < 5:
                        print(f'Processing sample {processed_count + 1}...')
                    
                    # Extract metadata features
                    age = row.get('age', 50.0)  # Default age if missing
                    if pd.isna(age):
                        age = 50.0
                    
                    sex = row.get('sex', 'U')  # Unknown if missing
                    if pd.isna(sex):
                        sex = 'U'
                    
                    # One-hot encode sex
                    sex_features = np.zeros(3)
                    if str(sex).lower().startswith('f'):
                        sex_features[0] = 1  # Female
                    elif str(sex).lower().startswith('m'):
                        sex_features[1] = 1  # Male  
                    else:
                        sex_features[2] = 1  # Unknown/Other
                    
                    # Extract label (assuming it's a binary classification)
                    # Common label column names in PhysioNet challenges
                    label_cols = ['label', 'target', 'diagnosis', 'chagas', 'is_positive']
                    label = None
                    for col in label_cols:
                        if col in row:
                            label = row[col]
                            break
                    
                    if label is None:
                        # Try numeric columns that might be labels
                        numeric_cols = data_df.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) > 0:
                            label = row[numeric_cols[-1]]  # Use last numeric column as label
                    
                    if pd.isna(label):
                        continue
                    
                    # Extract ECG signal features
                    # Try different possible keys for signal data
                    signal_keys = ['ecg', 'signal', 'data', 'tracings', f'{idx}', f'exam_{idx}']
                    signal_data = None
                    
                    for key in signal_keys:
                        if key in hdf:
                            try:
                                signal_data = hdf[key][:]
                                break
                            except:
                                continue
                    
                    if signal_data is None:
                        # Try accessing by index if signals are stored as indexed groups
                        try:
                            if str(idx) in hdf:
                                signal_data = hdf[str(idx)][:]
                            elif f'exam_{idx}' in hdf:
                                signal_data = hdf[f'exam_{idx}'][:]
                            else:
                                # Try the first available dataset
                                for key in hdf.keys():
                                    try:
                                        temp_data = hdf[key][:]
                                        if len(temp_data.shape) >= 2:  # Should be 2D for ECG
                                            signal_data = temp_data
                                            break
                                    except:
                                        continue
                        except:
                            pass
                    
                    if signal_data is None:
                        if verbose and error_count < 5:
                            print(f'  No signal data found for sample {idx}')
                        error_count += 1
                        continue
                    
                    # Process signal data
                    if len(signal_data.shape) == 1:
                        # Single lead, expand to 12 leads
                        signal_data = np.tile(signal_data.reshape(1, -1), (12, 1))
                    elif signal_data.shape[0] > 12:
                        # More than 12 leads, take first 12
                        signal_data = signal_data[:12, :]
                    elif signal_data.shape[0] < 12:
                        # Less than 12 leads, pad with zeros
                        padding = np.zeros((12 - signal_data.shape[0], signal_data.shape[1]))
                        signal_data = np.vstack([signal_data, padding])
                    
                    # Extract signal features (same as original)
                    signal_mean = np.zeros(12)
                    signal_std = np.zeros(12)
                    
                    for i in range(12):
                        channel_data = signal_data[i, :]
                        if len(channel_data) > 0:
                            signal_mean[i] = np.nanmean(channel_data)
                            signal_std[i] = np.nanstd(channel_data)
                    
                    # Handle NaN values
                    signal_mean = np.nan_to_num(signal_mean, nan=0.0)
                    signal_std = np.nan_to_num(signal_std, nan=0.0)
                    
                    # Combine all features
                    combined_features = np.concatenate([
                        [age],           # Age
                        sex_features,    # Sex (3 values)
                        signal_mean,     # Signal means (12 values)
                        signal_std       # Signal stds (12 values)
                    ])
                    
                    # Add to dataset
                    features.append(combined_features)
                    labels.append(int(label))
                    processed_count += 1
                    
                    if verbose and processed_count <= 5:
                        print(f'  Processed sample {processed_count}: age={age}, sex={sex}, label={label}')
                    
                except Exception as e:
                    error_count += 1
                    if verbose and error_count <= 5:
                        print(f'  Error processing sample {idx}: {e}')
                    continue
                    
                # Limit processing for large datasets
                if processed_count >= 10000:  # Limit to first 10k samples
                    break
                    
    except Exception as e:
        if verbose:
            print(f'Error accessing HDF5 file: {e}')
        return train_model_fallback(data_folder, model_folder, verbose)

    if verbose:
        print(f'Successfully processed {processed_count} records')
        print(f'Errors in {error_count} records')

    # Check if we have any data
    if len(features) == 0:
        if verbose:
            print('No data processed from HDF5, trying fallback method...')
        return train_model_fallback(data_folder, model_folder, verbose)

    # Convert to numpy arrays
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=bool)

    if verbose:
        print(f'Final dataset: {len(features)} samples, {features.shape[1]} features')
        print(f'Positive samples: {np.sum(labels)} ({np.mean(labels)*100:.1f}%)')

    # Handle NaN values
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Train the model
    if verbose:
        print('Training the model...')

    # Improved Random Forest parameters
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )

    model.fit(features_scaled, labels)

    # Optional validation
    if len(features) >= 10:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                features_scaled, labels, test_size=0.2, random_state=42, 
                stratify=labels if len(np.unique(labels)) > 1 else None
            )
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
            if verbose and len(np.unique(y_test)) > 1:
                auc_score = roc_auc_score(y_test, y_prob)
                print(f'Validation AUC: {auc_score:.4f}')
        except Exception as e:
            if verbose:
                print(f'Validation error: {e}')
            model.fit(features_scaled, labels)

    # Save model
    os.makedirs(model_folder, exist_ok=True)
    save_model(model_folder, model, scaler)

    if verbose:
        print('Training completed successfully!')

def train_model_fallback(data_folder, model_folder, verbose):
    """Fallback to original WFDB-based approach"""
    if verbose:
        print('Using fallback WFDB approach...')
    
    # Original code logic here
    records = find_records(data_folder)
    num_records = len(records)

    if num_records == 0:
        raise FileNotFoundError('No data were provided.')

    features = []
    labels = []
    processed_count = 0
    error_count = 0
    
    for i in range(num_records):
        if verbose:
            width = len(str(num_records))
            print(f'- {i+1:>{width}}/{num_records}: {records[i]}...')

        record = os.path.join(data_folder, records[i])
        
        try:
            age, sex, source, signal_mean, signal_std = extract_features(record)
            label = load_label(record)

            if source != 'CODE-15%' or (i % 10) == 0:
                if (age is not None and sex is not None and 
                    signal_mean is not None and signal_std is not None):
                    combined_features = np.concatenate((age, sex, signal_mean, signal_std))
                    if not np.any(np.isnan(combined_features)) and len(combined_features) > 0:
                        features.append(combined_features)
                        labels.append(int(label))
                        processed_count += 1
            
        except Exception as e:
            error_count += 1
            if verbose:
                print(f'  Error: {e}')
            continue

    if len(features) == 0:
        raise ValueError(f"No valid records processed. Processed {processed_count} out of {num_records} records. Errors: {error_count}")

    # Rest of training logic...
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=bool)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    model = RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_split=10,
        min_samples_leaf=5, random_state=42, class_weight='balanced', n_jobs=-1
    )
    model.fit(features_scaled, labels)
    
    os.makedirs(model_folder, exist_ok=True)
    save_model(model_folder, model, scaler)

def load_model(model_folder, verbose):
    model_filename = os.path.join(model_folder, 'model.sav')
    model_data = joblib.load(model_filename)
    return model_data

def run_model(record, model_data, verbose):
    model = model_data['model']
    scaler = model_data['scaler']

    # Try HDF5 approach first, then fallback to WFDB
    try:
        # If record is a path to HDF5 data, handle accordingly
        # For now, use the original approach
        age, sex, source, signal_mean, signal_std = extract_features(record)
        
        if any(x is None for x in [age, sex, signal_mean, signal_std]):
            return False, 0.0
        
        features = np.concatenate((age, sex, signal_mean, signal_std)).reshape(1, -1)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        features_scaled = scaler.transform(features)

        binary_output = model.predict(features_scaled)[0]
        probability_output = model.predict_proba(features_scaled)[0][1]
        return bool(binary_output), float(probability_output)
        
    except Exception as e:
        if verbose:
            print(f'Prediction error: {e}')
        return False, 0.0

################################################################################
#
# Helper functions
#
################################################################################

def extract_features(record):
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

    # Extract sex
    try:
        sex = get_sex(header)
        sex_one_hot_encoding = np.zeros(3, dtype=float)
        if sex is not None and sex.casefold().startswith('f'):
            sex_one_hot_encoding[0] = 1
        elif sex is not None and sex.casefold().startswith('m'):
            sex_one_hot_encoding[1] = 1
        else:
            sex_one_hot_encoding[2] = 1
    except Exception:
        sex_one_hot_encoding = np.array([0, 0, 1])

    # Extract source
    try:
        source = get_source(header)
        if source is None:
            source = 'Unknown'
    except Exception:
        source = 'Unknown'

    try:
        # Load signals
        signal, fields = load_signals(record)
        channels = fields['sig_name'] if 'sig_name' in fields else []
        reference_channels = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        
        signal = reorder_signal(signal, channels, reference_channels)
        num_channels = 12

        signal_mean = np.zeros(num_channels)
        signal_std = np.zeros(num_channels)

        for i in range(num_channels):
            if i < signal.shape[1]:
                channel_data = signal[:, i]
                num_finite_samples = np.sum(np.isfinite(channel_data))
                
                if num_finite_samples > 0:
                    signal_mean[i] = np.nanmean(channel_data)
                else:
                    signal_mean[i] = 0.0
                    
                if num_finite_samples > 1:
                    signal_std[i] = np.nanstd(channel_data)
                else:
                    signal_std[i] = 0.0
            else:
                signal_mean[i] = 0.0
                signal_std[i] = 0.0

        signal_mean = np.nan_to_num(signal_mean, nan=0.0)
        signal_std = np.nan_to_num(signal_std, nan=0.0)

        return age, sex_one_hot_encoding, source, signal_mean, signal_std
        
    except Exception:
        return age, sex_one_hot_encoding, source, np.zeros(12), np.zeros(12)

def save_model(model_folder, model, scaler):
    d = {'model': model, 'scaler': scaler}
    filename = os.path.join(model_folder, 'model.sav')
    joblib.dump(d, filename, protocol=0)
