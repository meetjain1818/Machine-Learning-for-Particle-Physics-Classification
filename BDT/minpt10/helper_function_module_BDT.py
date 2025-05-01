from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    auc, roc_curve, precision_recall_curve, confusion_matrix, 
    det_curve, DetCurveDisplay, class_likelihood_ratios
)
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import os
from tqdm import tqdm
import json

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    auc, roc_curve, precision_recall_curve, confusion_matrix, 
    det_curve, DetCurveDisplay, class_likelihood_ratios
)
import seaborn as sns

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

# --- Function to Load JSON ---
def load_json_data(filepath):
    """Loads data from a JSON file."""
    if not os.path.exists(filepath):
        print(f"*** Error: JSON file not found at {filepath} :(")
        return None
    try:
        print(f"Loading event data from {filepath}...")
        with open(filepath, 'r') as f:
            data = json.load(f)
        if not isinstance(data, list):
            print(f"*** Error: Expected a list of events in JSON file, found {type(data)} :(")
            return None
        print(f"--- Successfully loaded {len(data)} events :)")
        return data
    except Exception as e:
        print(f"*** An unexpected error occurred during JSON loading: {e} :(")
        return None




def calculate_delta_phi(phi1:float, phi2:float) -> float:
    """Calculates delta phi correctly handling periodicity."""
    dphi = phi1 - phi2
    dphi = np.arctan2(np.sin(dphi), np.cos(dphi))
    return dphi

def calculate_delta_r(eta1:float, phi1:float, eta2:float, phi2:float) -> float:
    """Calculates delta R between two objects."""
    deta = eta1 - eta2
    dphi = calculate_delta_phi(phi1, phi2)
    return np.sqrt(deta**2 + dphi**2)

def calculate_invariant_mass(particles:list[dict, ...]) -> float:
    """
    Calculates the invariant mass of a system of particles.

    Args:
        particles (list): A list of dictionaries, where each dictionary
                          represents a particle and must contain keys
                          'E', 'Px', 'Py', 'Pz' with numeric values.

    Returns:
        float: The invariant mass, or np.nan if input is invalid or
               mass squared is negative. Returns 0.0 if particles list is empty.
    """
    if not particles:
        print(f"*** Warning: Empty particles list encountered. Returning Nan :(")
        return np.nan

    tot_E, tot_Px, tot_Py, tot_Pz = 0.0, 0.0, 0.0, 0.0
    required_keys = {'E', 'Px', 'Py', 'Pz'}

    for p in particles:
        if not required_keys.issubset(p.keys()):
            print(f"*** Warning: Particle missing required keys {required_keys - set(p.keys())}. Cannot calculate invariant mass :(")
            return np.nan
        try:
            # Ensure values are numeric and not None before summing
            p_E = float(p['E']) if p['E'] is not None else 0.0
            p_Px = float(p['Px']) if p['Px'] is not None else 0.0
            p_Py = float(p['Py']) if p['Py'] is not None else 0.0
            p_Pz = float(p['Pz']) if p['Pz'] is not None else 0.0

            tot_E += p_E
            tot_Px += p_Px
            tot_Py += p_Py
            tot_Pz += p_Pz
        except (TypeError, ValueError) as e:
             print(f"*** Warning: Non-numeric value encountered in particle {p}. Error: {e}. \nCannot calculate invariant mass :(")
             return np.nan


    mass_squared = tot_E**2 - (tot_Px**2 + tot_Py**2 + tot_Pz**2)

    if mass_squared < 0:
        print(f"*** Warning: Negative mass squared ({mass_squared}) encountered. Returning Nan :(")
        return np.nan
    else:
        return np.sqrt(mass_squared)


# --- Function to Prepare Data ---
def prepare_data_for_model(event_data:list[dict, ...], required_jets:int=2, required_photons:int=1, event_label:int = None) -> tuple[np.array, np.array, list[int]]:
    """
    Extracts features and labels from event dictionaries for model training.

    Args:
        event_data (list): List of event dictionaries from JSON.
        required_jets (int): Minimum number of jets required per event.
        required_photons (int): Minimum number of photons required per event.
        event_label (int): Binary label, 0 for Background, 1 for Signla

    Returns:
        tuple: (X, y, valid_event_indices)
               X (np.ndarray): Feature matrix (n_valid_events, n_features).
               y (np.ndarray): Target labels (n_valid_events,).
               valid_event_indices (list): Indices of events from original list
                                           that were used.
            Returns (None, None, []) if input is invalid or no valid events found.
    """
    if not isinstance(event_data, list) or not event_data:
        print("*** Error: Input event_data is invalid or empty :(")
        return None, None, []

    features_list = []
    labels_list = []
    valid_indices = []
    feature_names = ['isophoton_pT', 'deltaR_jet12', 'invMass_2j1p', 'inv_mass_2j']

    print(f"Processing {len(event_data)} events to extract features...")
    for idx, event in enumerate(tqdm(event_data, desc="Extracting Features")):
        jets = event.get('jets', [])
        photons = event.get('photons', [])

        # --- Check minimum requirements ---
        if len(jets) < required_jets or len(photons) < required_photons:
            continue # Skip event if not enough jets or photons

        if event_label is None or event_label == -1: # Check if label is valid
             print(f"*** Skipping event {event.get('eventno', 'N/A')} due to invalid event label :(")
             continue

        # --- Sort by pT (descending) ---
        # Use try-except for robustness against missing 'pT' or non-numeric values
        try:
            jets_sorted = sorted(jets, key=lambda j: float(j.get('pT', -np.inf)), reverse=True)
            photons_sorted = sorted(photons, key=lambda p: float(p.get('pT', -np.inf)), reverse=True)
        except (TypeError, ValueError):
            print(f"*** Warning: Skipping event {event.get('eventno', 'N/A')} due to non-numeric pT value :(")
            continue

        # --- Get leading objects ---
        jet1 = jets_sorted[0]
        jet2 = jets_sorted[1]
        photon1 = photons_sorted[0]

        # --- Calculate Features ---
        try:
            # 1. IsoPhoton pT
            iso_photon_pt = float(photon1.get('pT', np.nan)) # Default to NaN if missing

            # 2. Delta R Jet1-Jet2
            eta1 = float(jet1.get('Eta', np.nan))
            phi1 = float(jet1.get('Phi', np.nan))
            eta2 = float(jet2.get('Eta', np.nan))
            phi2 = float(jet2.get('Phi', np.nan))
            # Check if coordinates are valid before calculating delta R
            if np.isnan(eta1) or np.isnan(phi1) or np.isnan(eta2) or np.isnan(phi2):
                delta_r_12 = np.nan
            else:
                delta_r_12 = calculate_delta_r(eta1, phi1, eta2, phi2)

            # 3. Invariant Mass (2 jets, 1 photon)
            # The function handles missing E, Px, Py, Pz keys internally
            inv_mass_2j1p = calculate_invariant_mass([jet1, jet2, photon1])
            inv_mass_2j = calculate_invariant_mass([jet1, jet2])

            # --- Store results ---
            current_features = [iso_photon_pt, delta_r_12, inv_mass_2j1p, inv_mass_2j]

            # Optional: Check if any feature calculation resulted in NaN
            if np.isnan(current_features).any():
                 print(f"*** Skipping event {event.get('eventno', 'N/A')} due to NaN in calculated features :(")
                 continue

            features_list.append(current_features)
            labels_list.append(int(event_label))
            valid_indices.append(event.get('eventno', idx))

        except (KeyError, TypeError, ValueError) as e:
            print(f"*** Warning: Error processing event {event.get('eventno', 'N/A')}: {e}. Skipping :(")
            continue

    if not features_list:
        print("*** Error: No valid events found after feature extraction :(")
        return None, None, []

    X_df = pd.DataFrame(features_list, columns=feature_names, index=valid_indices)
    X_df['event_label'] = labels_list

    # --- Handle potential NaNs/Infs using Imputer (operates on and returns numpy array) ---
    X_np = X_df.values # Get numpy array for imputation
    imputed = False
    if np.isnan(X_np).any():
        print("\nWarning: NaN values found in feature matrix. Applying Imputer (median).")
        imputer = SimpleImputer(strategy='median')
        X_np = imputer.fit_transform(X_np)
        imputed = True

    if np.isinf(X_np).any():
        print("\nWarning: Infinite values found in feature matrix. Replacing with NaN and imputing.")
        X_np[np.isinf(X_np)] = np.nan
        imputer = SimpleImputer(strategy='median') # Re-apply if Inf was present
        X_np = imputer.fit_transform(X_np)
        imputed = True

    # If imputation happened, update the DataFrame
    if imputed:
        X_df = pd.DataFrame(X_np, columns=feature_names, index=valid_indices)
        print("DataFrame updated after imputation.")


    return X_df