import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import logging
from imblearn.over_sampling import SMOTE

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(file_path):
    """Load the stroke dataset"""
    logger.info(f"Loading data from {file_path}")
    return pd.read_csv(file_path)

def check_data_quality(data):
    """Verify data quality and completeness"""
    # Check for null values
    null_counts = data.isnull().sum()
    
    # Check for expected columns
    expected_columns = ['gender', 'age', 'hypertension', 'heart_disease', 
                       'ever_married', 'work_type', 'Residence_type',
                       'avg_glucose_level', 'bmi', 'smoking_status', 'stroke']
    
    missing_columns = set(expected_columns) - set(data.columns)
    
    if null_counts.sum() > 0 or missing_columns:
        return False, {"null_counts": null_counts.to_dict(), 
                      "missing_columns": list(missing_columns)}
    return True, {}

def handle_outliers(data, columns):
    """Handle outliers using IQR method"""
    data_clean = data.copy()
    
    for column in columns:
        Q1 = data_clean[column].quantile(0.25)
        Q3 = data_clean[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        data_clean = data_clean[(data_clean[column] >= lower_bound) & 
                               (data_clean[column] <= upper_bound)]
    
    logger.info(f"Removed {len(data) - len(data_clean)} rows as outliers")
    return data_clean

def preprocess_data(data):
    """Apply preprocessing to raw data"""
    # Handle outliers for numerical columns
    numerical_columns = ['age', 'avg_glucose_level', 'bmi']
    data_clean = handle_outliers(data, numerical_columns)
    
    # One-hot encoding for categorical columns
    categorical_columns = ['gender', 'ever_married', 'work_type', 
                          'Residence_type', 'smoking_status']
    
    data_encoded = pd.get_dummies(data_clean, columns=categorical_columns)
    
    # Convert to appropriate types
    data_encoded = data_encoded.astype({col: 'int64' for col in 
                                       data_encoded.select_dtypes('bool').columns})
    
    logger.info(f"Data preprocessed. Final shape: {data_encoded.shape}")
    return data_encoded

def split_data(data, test_size=0.2, val_size=0.1, random_state=42):
    """Split data into training, validation, and testing sets"""
    # First split: separate test set
    X = data.drop('stroke', axis=1) 
    y = data['stroke']
    
    # Apply SMOTE to the training data
    smote = SMOTE(random_state=42)
    X_smote, y_smote = smote.fit_resample(X, y)

    df_smote = pd.DataFrame(X_smote, columns=X.columns)
    df_smote['stroke'] = y_smote

    X = df_smote.drop('stroke', axis=1) # Features
    y = df_smote['stroke']
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    # Second split: create validation set from remaining data
    if val_size > 0:
        val_size_adjusted = val_size / (1 - test_size)  # Adjust validation size
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, 
            random_state=random_state, stratify=y_temp)
        
        logger.info(f"Data split into train ({len(X_train)}), "
                  f"validation ({len(X_val)}), and test ({len(X_test)}) sets")
        
        return (X_train, X_val, X_test, y_train, y_val, y_test)
    else:
        logger.info(f"Data split into train ({len(X_temp)}) and test ({len(X_test)}) sets")
        return (X_temp, X_test, y_temp, y_test)

def save_splits(X_train, X_test, y_train, y_test, X_val=None, y_val=None, output_dir="data/processed"):
    """Save the data splits to disk"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create DataFrames with features and target
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)
    
    # Save to CSV
    train_data.to_csv(f"{output_dir}/train_data.csv", index=False)
    test_data.to_csv(f"{output_dir}/test_data.csv", index=False)
    
    # Also save validation set if provided
    if X_val is not None and y_val is not None:
        val_data = pd.concat([X_val, y_val], axis=1)
        val_data.to_csv(f"{output_dir}/val_data.csv", index=False)
    
    # Save a reference copy of the data for drift detection
    X_train.describe().to_csv(f"{output_dir}/reference_data_stats.csv")
    X_train.head(1000).to_csv(f"{output_dir}/reference_data.csv", index=False)
    
    logger.info(f"Data splits saved to {output_dir}")