import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    confusion_matrix, 
    roc_curve, 
    roc_auc_score
)
import logging

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    logger.info("Evaluating model performance")
    
    # Generate predictions
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # ROC and AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    roc_auc = roc_auc_score(y_test, y_probs)
    
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision}")
    logger.info(f"Recall: {recall}")
    logger.info(f"F1 Score: {f1}")
    logger.info(f"ROC AUC: {roc_auc:.4f}")
    
    # Compile results
    results = {
        'accuracy': float(accuracy),
        'precision': precision.tolist(),
        'recall': recall.tolist(),
        'f1': f1.tolist(),
        'support': support.tolist(),
        'confusion_matrix': conf_matrix.tolist(),
        'roc_auc': float(roc_auc),
        'roc_curve': {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': thresholds.tolist()
        }
    }
    
    return results

def save_evaluation_results(results, output_path):
    """Save evaluation results to a file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"Evaluation results saved to {output_path}")
    
    return output_path

def plot_evaluation_results(results, output_dir):
    """Create and save visualizations of model performance"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create confusion matrix plot
    plt.figure(figsize=(8, 6))
    conf_matrix = np.array(results['confusion_matrix'])
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
               xticklabels=['No Stroke', 'Stroke'],
               yticklabels=['No Stroke', 'Stroke'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    cm_path = f"{output_dir}/confusion_matrix.png"
    plt.savefig(cm_path)
    
    # Create ROC curve plot
    plt.figure(figsize=(8, 6))
    plt.plot(results['roc_curve']['fpr'], results['roc_curve']['tpr'], 
             color='blue', label=f"ROC Curve (area = {results['roc_auc']:.2f})")
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    roc_path = f"{output_dir}/roc_curve.png"
    plt.savefig(roc_path)
    
    logger.info(f"Evaluation plots saved to {output_dir}")
    
    return {
        "confusion_matrix_path": cm_path,
        "roc_curve_path": roc_path
    }