import numpy as np
from sklearn.metrics import auc, precision_recall_curve

def precision_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: float) -> float:
    """
    Calculate precision @ top k percent of predicted scores.
    
    Args:
        y_true (np.ndarray): True labels.
        y_scores (np.ndarray): Predicted scores or probablities.
        k (float): Top k percent (5.0 for top 5%)
        
    Returns:
        float: Precision @ top k
    """
    
    if k <= 0 or k > 100:
        raise ValueError("k must be between 0 and 100")
    
    num_top_k = int(len(y_scores) * k / 100)
    
    if num_top_k == 0:
        return 0.0
    
    # Get the indices of the item with the highest scores
    top_k_indices = np.argsort(y_scores)[-num_top_k:]
    
    # Count how many of these top k items are actually positive (fraud)
    n_relevant_in_top_k = np.sum(y_true[top_k_indices])
    
    # Precision = (True Positive in Top K) / K
    return n_relevant_in_top_k / num_top_k

def pr_auc_score(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Calculate PR AUC (are under the precision-recall curve) score.
    
    Args:
        y_true (np.ndarray): True labels.
        y_scores (np.ndarray): Predicted scores or probablities.
        
    Returns:
        float: PR AUC score
    """
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    return auc(recall, precision)
