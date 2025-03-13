import pandas as pd
import numpy as np
import joblib
import json
import os
import logging
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path):
    """Load model configuration"""
    with open(config_path, 'r') as f:
        return json.load(f)

def apply_smote(X_train, y_train, random_state=42):
    """Apply SMOTE to handle class imbalance"""
    logger.info("Applying SMOTE to balance the training data")
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    logger.info(f"Original shape: {X_train.shape}, Resampled shape: {X_resampled.shape}")
    return X_resampled, y_resampled

def train_model(X_train, y_train, model_params=None, grid_search=True, cv=5, 
               random_state=42, n_jobs=-1):
    """Train the Random Forest model with optional GridSearch"""
    
    # Apply SMOTE to balance the dataset
    X_resampled, y_resampled = apply_smote(X_train, y_train, random_state)
    
    # Initialize model
    rf_model = RandomForestClassifier(random_state=random_state)
    
    if grid_search:
        # Define hyperparameter grid for tuning
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'bootstrap': [True, False]
        } if model_params is None else model_params
        
        logger.info("Starting GridSearchCV for hyperparameter tuning")
        grid_search = GridSearchCV(
            estimator=rf_model,
            param_grid=param_grid,
            cv=cv,
            n_jobs=n_jobs,
            verbose=1,
            scoring='f1'
        )
        
        grid_search.fit(X_resampled, y_resampled)
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best cross-validation score: {best_score:.4f}")
        
        return best_model, best_params, best_score
    
    # Train with provided parameters if grid search disabled
    if model_params:
        rf_model.set_params(**model_params)
    
    logger.info("Training model without GridSearchCV")
    rf_model.fit(X_resampled, y_resampled)
    
    return rf_model, rf_model.get_params(), None

def save_model(model, model_dir='models', model_name=None):
    """Save the trained model and metadata"""
    os.makedirs(model_dir, exist_ok=True)
    
    # Create timestamp for versioning
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = model_name or f"stroke_prediction_model_{timestamp}.joblib"
    model_path = os.path.join(model_dir, model_name)
    
    # Save the model
    joblib.dump(model, model_path)
    
    # Create a symlink to the latest model
    latest_path = os.path.join(model_dir, "latest_model.joblib")
    if os.path.exists(latest_path):
        os.remove(latest_path)
    
    # Create symlink (Windows doesn't support symlinks easily, so use copy as fallback)
    try:
        os.symlink(model_path, latest_path)
    except:
        import shutil
        shutil.copy2(model_path, latest_path)
    
    logger.info(f"Model saved to {model_path}")
    logger.info(f"Latest model link updated: {latest_path}")
    
    return model_path