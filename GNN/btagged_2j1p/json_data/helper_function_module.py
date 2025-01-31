from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    auc, roc_curve, precision_recall_curve, confusion_matrix, 
    det_curve, DetCurveDisplay, class_likelihood_ratios
)
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import random
import torch_geometric

def check_pyg_cuda_status():
    """
    Display the CUDA and PyTorch Geometric (PyG) configuration, including version and GPU status.
    Verifies the installation of key PyG dependencies (torch_scatter, torch_sparse, etc.) and runs a practical
    test with a GCNConv layer to ensure CUDA-based computations are functional.
    """
    print("System CUDA Status:")
    print(f"- PyTorch version: {torch.__version__}")
    print(f"- CUDA Version: {torch.version.cuda}")
    print(f"- CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"- CUDA version: {torch.version.cuda}")
        print(f"- CUDA device: {torch.cuda.get_device_name(0)}")
    
    print("\nPyTorch Geometric Status:")
    print(f"- PyG version: {torch_geometric.__version__}")
    
    try:
        import torch_scatter
        print("- torch_scatter: Installed")
    except ImportError:
        print("- torch_scatter: Not installed")
    
    try:
        import torch_sparse
        print("- torch_sparse: Installed")
    except ImportError:
        print("- torch_sparse: Not installed")
    
    try:
        import torch_cluster
        print("- torch_cluster: Installed")
    except ImportError:
        print("- torch_cluster: Not installed")
    
    try:
        import torch_spline_conv
        print("- torch_spline_conv: Installed")
    except ImportError:
        print("- torch_spline_conv: Not installed")

    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    x = torch.randn(2, 10)
    if torch.cuda.is_available():
        edge_index = edge_index.cuda()
        x = x.cuda()
        try:
            conv = torch_geometric.nn.GCNConv(10, 10).cuda()
            out = conv(x, edge_index)
            print("\nPractical Test: Successfully ran GCNConv on GPU")
        except Exception as e:
            print(f"\nPractical Test Failed: {str(e)}")

def set_seed(seed):
    """
    Set random seed for reproducibility across Python's random module, NumPy, and PyTorch.

    Parameters:
    - seed (int): The seed value to ensure deterministic behavior.
    """
    random.seed(seed)
    np.random.seed(seed) 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def init_setup(batch_size=70, use_cuda=True, random_seed=1234):
    """
    Initialize the computational environment by setting up CUDA, batch size, and seed for reproducibility.

    Parameters:
    - batch_size (int): The batch size for training.
    - use_cuda (bool): Flag to enable or disable CUDA usage.
    - random_seed (int): Seed for reproducibility.
    """
    BATCH_SIZE = batch_size
    device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')
    
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name()}")
        print(f"GPU Memory Usage: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
    check_pyg_cuda_status()
    set_seed(random_seed)

def eval_model(y_true, y_pred_prob, y_pred_label):
    """
    Evaluate the model's performance using various classification metrics and visualization.

    Parameters:
    - y_true (array-like): True class labels.
    - y_pred_prob (array-like): Predicted probabilities for the positive class.
    - y_pred_label (array-like): Predicted class labels.

    Displays metrics such as accuracy, precision, recall, F1-score, and likelihood ratios,
    and plots the ROC curve, Precision-Recall curve, confusion matrix heatmap, and prediction histogram.
    """
    accuracy = accuracy_score(y_true, y_pred_label)
    precision = precision_score(y_true, y_pred_label)
    recall = recall_score(y_true, y_pred_label)
    f1 = f1_score(y_true, y_pred_label)
    conf_matrix = confusion_matrix(y_true, y_pred_label)
    pos_LR, neg_LR = class_likelihood_ratios(y_true, y_pred_label)
    print(f"LR+: {pos_LR:.3f}")
    print(f"LR-: {neg_LR:.3f}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-Score: {f1:.3f}")

    figure, ax = plt.subplots(2, 2, figsize=(8, 6), dpi=300)
    figure.suptitle("Model Evaluation Metrics", fontsize=16)
    
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    ax[0, 0].plot(fpr, tpr, color='blue', lw=2, label=f"AUC = {roc_auc:.3f}")
    ax[0, 0].plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
    ax[0, 0].set_xlabel("False Positive Rate")
    ax[0, 0].set_ylabel("True Positive Rate")
    ax[0, 0].set_title('ROC Curve')
    ax[0, 0].legend(loc="lower right")
    ax[0, 0].grid(alpha=0.3)

    precision_values, recall_values, thresholds = precision_recall_curve(y_true, y_pred_prob)
    pr_auc = auc(recall_values, precision_values)
    ax[0, 1].plot(recall_values, precision_values, label=f"AUC = {pr_auc:.3f}")
    ax[0, 1].plot([0, 1], [1, 0], color='red', lw=2, linestyle='--')
    ax[0, 1].set_xlabel("Recall")
    ax[0, 1].set_ylabel("Precision")
    ax[0, 1].set_title('Precision-Recall Curve')
    ax[0, 1].legend(loc="lower left")
    ax[0, 1].grid(alpha=0.3)

    group_counts = ["{0:0.0f}".format(value) for value in conf_matrix.flatten()]
    group_percentage = np.array([np.round((row/total)*100, 2) for row, total in zip(conf_matrix, np.sum(conf_matrix, axis=1))]).flatten()
    labels = np.asarray([f"{counts}\n{percent}%" for counts, percent in zip(group_counts, group_percentage)]).reshape((2, 2))
    sns.heatmap(conf_matrix, annot=labels, fmt='', ax=ax[1, 0])
    ax[1, 0].set_xlabel("Predicted Labels")
    ax[1, 0].set_ylabel("True Labels")
    ax[1, 0].set_title("Confusion Matrix Heatmap")

    ax[1, 1].hist(y_pred_prob, histtype='step', linewidth=2, label="Model's Prediction")
    ax[1, 1].hist(y_true, histtype='step', color='green', linewidth=2, label="Actual Labels", alpha=0.4)
    ax[1, 1].set_xlabel('Predicted Probabilities')
    ax[1, 1].set_ylabel('Frequency')
    ax[1, 1].legend()
    
    plt.tight_layout()
    plt.show()

def Euclidean_distance(node1: list[float, float], node2: list[float, float]) -> float:
    """
    Compute the Euclidean distance between two nodes in a graph using (Eta, Phi) coordinates.

    Parameters:
    - node1 (list): A list containing Eta and Phi of the first node.
    - node2 (list): A list containing Eta and Phi of the second node.

    Returns:
    - float: The Euclidean distance considering periodic boundaries in Phi.
    """
    diff = node1 - node2
    if np.abs(diff[1]) > np.pi:
        diff[1] = 2 * np.pi - np.abs(diff[1])
    euclidean_distance = np.sqrt(np.sum(diff**2))
    return euclidean_distance

def invariant_mass(x: pd.Series):
    """
    Calculate the invariant mass of a system comprising four jets and one photon.

    Parameters:
    - x (pd.Series): A pandas Series containing energies and momenta (Px, Py, Pz) of
      jets and the isolated photon.

    Returns:
    - float: The invariant mass of the event.
    """
    total_energy = x.loc[['isophoton_E', 'jet1_E', 'jet2_E', 'jet3_E', 'jet4_E']].sum()
    total_px = x.loc[['isophoton_Px', 'jet1_Px', 'jet2_Px', 'jet3_Px', 'jet4_Px']].sum()
    total_py = x.loc[['isophoton_Py', 'jet1_Py', 'jet2_Py', 'jet3_Py', 'jet4_Py']].sum()
    total_pz = x.loc[['isophoton_Pz', 'jet1_Pz', 'jet2_Pz', 'jet3_Pz', 'jet4_Pz']].sum()

    inv_mass = np.sqrt(total_energy**2 - total_px**2 - total_py**2 - total_pz**2)
    return inv_mass
