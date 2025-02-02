import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    auc, roc_curve, precision_recall_curve, confusion_matrix, 
    det_curve, DetCurveDisplay, class_likelihood_ratios
)
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
def get_labels(model, dataset) -> (list, list):
    """
    Given a trained model and a dataset, returns the true labels and model-predicted probabilities.

    Parameters:
    model (tf.keras.Model): Trained TensorFlow model.
    dataset (tf.data.Dataset or tuple): Either a tf.data.Dataset or (X, y) tuple.

    Returns:
    tuple: (true_labels, predicted_probs), where both are lists.
    """
    true_labels = []
    predicted_probs = []

    # Check dataset format
    if isinstance(dataset, tuple):
        X, y = dataset  # Assume (features, labels) tuple
        true_labels = list(y)
        predicted_probs = list(model.predict(X, verbose=0).flatten())
    else:
        for batch in dataset:  # tf.data.Dataset
            X_batch, y_batch = batch
            true_labels.extend(y_batch.numpy())
            predicted_probs.extend(model.predict(X_batch, verbose=0).flatten())

    return true_labels, predicted_probs


def eval_model(y_true:list[float], y_pred_prob:list[float], y_pred_label:list[float],*, save_fig:bool = False, save_fig_path:str = None) -> None:
    """
    Evaluate the model's performance using various classification metrics and visualization.

    Parameters:
    - y_true (array-like): True class labels.
    - y_pred_prob (array-like): Predicted probabilities for the positive class.
    - y_pred_label (array-like): Predicted class labels.
    - save_fig (bool): If set to True then figure will be saved to specified filepath.
    - save_fig_path (str): Filepath where to save the figure.

    Displays metrics such as accuracy, precision, recall, F1-score, and likelihood ratios,
    and plots the ROC curve, Precision-Recall curve, confusion matrix heatmap, and prediction histogram.
    """
    accuracy = accuracy_score(y_true, y_pred_label)
    precision = precision_score(y_true, y_pred_label)
    recall = recall_score(y_true, y_pred_label)
    f1 = f1_score(y_true, y_pred_label)
    conf_matrix = confusion_matrix(y_true, y_pred_label)
    pos_LR, neg_LR = class_likelihood_ratios(y_true, y_pred_label)

    figure, ax = plt.subplots(2, 2, figsize=(8, 6), dpi=300)
    figure.suptitle("Model Evaluation Metrics", fontsize=16)

    # --- Metrics Table (0,0) ---
    ax[0,0].axis('off')
    metrics = [
        ['Positive LR', f"{pos_LR:.3f}"],
        ['Negative LR', f"{neg_LR:.3f}"],
        ['Accuracy', f"{accuracy:.3f}"],
        ['Precision', f"{precision:.3f}"],
        ['Recall', f"{recall:.3f}"],
        ['F1-Score', f"{f1:.3f}"]
    ]
    
    table = ax[0,0].table(
        cellText=metrics,
        colLabels=['Metric', 'Value'],
        cellLoc='center',
        loc='center',
        colColours=['#f0f0f0', '#f0f0f0']
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)

    # --- ROC Curve (0, 1) ---
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    ax[0, 1].plot(fpr, tpr, color='blue', lw=2, label=f"AUC = {roc_auc:.3f}")
    ax[0, 1].plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
    ax[0, 1].set_xlabel("False Positive Rate")
    ax[0, 1].set_ylabel("True Positive Rate")
    ax[0, 1].set_title('ROC Curve')
    ax[0, 1].legend(loc="lower right")
    ax[0, 1].grid(alpha=0.3)


    # --- Confusion Matrix (2,1) ---
    group_counts = ["{0:0.0f}".format(value) for value in conf_matrix.flatten()]
    group_percentage = np.array([np.round((row/total)*100, 2) for row, total in zip(conf_matrix, np.sum(conf_matrix, axis=1))]).flatten()
    labels = np.asarray([f"{counts}\n{percent}%" for counts, percent in zip(group_counts, group_percentage)]).reshape((2, 2))
    sns.heatmap(conf_matrix, annot=labels, fmt='', ax=ax[1, 0])
    ax[1, 0].set_xlabel("Predicted Labels")
    ax[1, 0].set_ylabel("True Labels")
    ax[1, 0].set_title("Confusion Matrix Heatmap")

    # --- Predicted Probabilities Histogram(2,2) ---
    ax[1, 1].hist(y_pred_prob, histtype='step', linewidth=2, label="Model's Prediction")
    ax[1, 1].hist(y_true, histtype='step', color='green', linewidth=2, label="Actual Labels", alpha=0.4)
    ax[1, 1].set_xlabel('Predicted Probabilities')
    ax[1, 1].set_ylabel('Frequency')
    ax[1, 1].set_title('Predicted Probabilities Histogram')
    ax[1, 1].legend()
    
    plt.tight_layout()
    if save_fig & (save_fig_path != None):
        plt.savefig(save_fig_path, bbox_inches = 'tight', pad_inches = 0.05)
        print(f"Figure saved to {save_fig_path} successfully :)")
    
    plt.show()
    
    return None