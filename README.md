# Classifying Signal from Background in High Energy Collisions using Machine Learning Approaches

**Master Thesis Project**
*Authors: Meet Jain (2023PHS7210), Kashish Jangra (2023PHS7178)*
*Supervisor: Prof. Abhishek Murlidhar Iyer*
*Department of Physics, Indian Institute of Technology Delhi*
*May 2025*

---

## Abstract

This project explores the application of modern machine learning techniques for classifying rare signal events from the Beyond Standard Model process $pp \to Z \to a\gamma$ (with $a \to b\bar{b}$, where $a$ is an Axion-Like Particle) against overwhelming Standard Model backgrounds. Using Monte Carlo simulations generated via MadGraph5_aMC@NLO, Pythia 8, and Delphes 3, we compared the performance of Boosted Decision Trees (BDTs), Convolutional Neural Networks (CNNs), and Graph Neural Networks (GNNs). Initial studies showed BDTs achieved strong performance (AUC ≈ 0.87) with engineered kinematic features, while CNNs performed poorly (AUC ≈ 0.57). GNNs, particularly the NNConv architecture leveraging $\Delta R$ as edge features, showed significant promise, outperforming simpler GCNConv models, achieving a baseline AUC ≈ 0.89 using a 3-node graph ($\gamma, b, b$) with b-tag labels. Sensitivity studies varying ALP mass ($m_a = 15, 45$ GeV), jet $p_T$ threshold (10, 20 GeV), and b-jet selection criteria revealed high sensitivity to these parameters. Lowering the $p_T$ threshold to 10 GeV was crucial for accessing the $m_a = 15$ GeV signal phase space. The highest classification performance (AUC ≈ 0.96) was achieved by both BDT and NNConv GNN models for the $m_a=15$ GeV, $p_{T,min}=10$ GeV scenario, using only the first two b-tagged jets. The results highlight the power of GNNs (NNConv) and the critical importance of aligning analysis choices with the signal hypothesis.

---

## Table of Contents

*   [Motivation](#motivation)
*   [Project Goal](#project-goal)
*   [Methodology Overview](#methodology-overview)
    *   [Data Simulation](#data-simulation)
    *   [Data Preprocessing](#data-preprocessing)
    *   [Machine Learning Models](#machine-learning-models)
*   [Repository Structure](#repository-structure)
*   [Dependencies](#dependencies)
*   [Installation](#installation)
*   [Usage Workflow](#usage-workflow)
*   [Dataset Information](#dataset-information)
*   [Results Summary](#results-summary)
*   [License](#license)
*   [Citation](#citation)
*   [Contact](#contact)

---

## Motivation

Searching for new physics beyond the Standard Model (BSM) is a primary goal of high-energy physics. Collider experiments like the LHC produce vast amounts of data, but potential BSM signal events are extremely rare compared to background events from known Standard Model processes. This project tackles the challenge of distinguishing a theoretically motivated rare signal ($Z \to a\gamma, a \to b\bar{b}$) from its dominant backgrounds, facing issues like:
*   **Extreme Class Imbalance:** Signal events are many orders of magnitude rarer than background.
*   **Complex Event Topology:** Events involve multiple particles with intricate kinematic and spatial correlations.
*   **Limitations of Traditional Methods:** Simple cut-based analyses may lack sensitivity, while standard ML on tabular data might miss relational information, and image-based CNNs can struggle with sparse HEP data.
*   **Graph Neural Networks (GNNs):** Offer a promising approach by naturally modeling events as graphs, representing particles as nodes and their relationships (like angular separation) as edges.

## Project Goal

The primary objective is to investigate and compare the effectiveness of various machine learning techniques (BDTs, CNNs, GNNs) for classifying signal vs. background events in the $pp \to Z \to a\gamma, a \to b\bar{b}$ channel, using simulated data. This includes exploring different data representations and evaluating model sensitivity to key physics parameters like ALP mass ($m_a$) and kinematic cuts ($p_{T,min}$).

## Methodology Overview

The project follows a systematic approach:

### Data Simulation
*   Signal ($pp \to Z \to a\gamma, a \to b\bar{b}$) and background (e.g., $Z+\gamma \to q\bar{q}+\gamma$) events were generated using standard HEP tools:
    *   **MadGraph5_aMC@NLO:** For matrix element calculation.
    *   **Pythia 8:** For parton showering and hadronization.
    *   **Delphes 3:** For fast detector simulation and reconstruction of physics objects (photons, jets, b-tagging).
*   Simulations were performed for different scenarios: $m_a \in \{15, 45\}$ GeV and minimum jet $p_T \in \{10, 20\}$ GeV.

### Data Preprocessing
*   **Event Selection:** Basic cuts on photon $p_T$, jet $p_T$, $|\eta|$, and number of b-tagged jets.
*   **Object Definition:** Identifying the isolated photon ($\gamma$), leading jet (Jet1), and sub-leading jet (Jet2). Specific b-jet selection criteria ("Only first 2 b-tagged" vs. "Any two b-tagged") were explored.
*   **Feature Engineering:** Calculating high-level variables like invariant masses ($m_{\gamma JJ}, m_{JJ}$) and angular separation ($\Delta R_{JJ}$).
*   **Data Standardization:** Transforming $\eta-\phi$ coordinates to center the photon at the origin and rotate the leading jet to a fixed angle, removing trivial rotations/translations.

### Machine Learning Models
1.  **BDTs (XGBoost, GradientBoost, RandomForest):** Trained on engineered tabular features ($p_{T,\gamma}, m_{\gamma JJ}, m_{JJ}, \Delta R_{JJ}$). Hyperparameters optimized via GridSearchCV. Feature importance was analyzed.
2.  **CNNs (Custom 3-Layer, AlexNet-like):** Trained on 2D image representations of the standardized $\eta-\phi$ plane, with pixel intensity proportional to $p_T$.
3.  **GNNs (GCNConv, NNConv):** Trained on graph representations of events:
    *   **Nodes:** Photon, selected b-jets.
    *   **Node Features:** $\eta, \phi$, b-tag label (and initially $p_T, E$).
    *   **Edge Features:** $\Delta R_{ij}$.
    *   **Graph Features:** Global kinematics like $p_{T,\gamma}, m_{\gamma JJ}, m_{JJ}$.
    *   NNConv dynamically uses edge features, GCNConv typically does not.

## Dependencies

The project primarily uses Python 3. Key libraries include:
*   `pandas`, `numpy` for data manipulation
*   `uproot`, `awkward-array` for handling ROOT files (if applicable)
*   `scikit-learn` for preprocessing, BDTs (RF, GB), evaluation metrics
*   `xgboost` for XGBoost BDT
*   `matplotlib`, `seaborn` for plotting
*   `tensorflow`/`keras` or `pytorch` for CNN implementation
*   `pytorch`
*   `pytorch-geometric` (PyG) for GNN implementation

See `requirements.txt` for a full list of dependencies and versions.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/meetjain1818/Machine-Learning-for-Particle-Physics-Classification.git
    cd Machine-Learning-for-Particle-Physics-Classification
    ```
2.  **Set up the Python environment (Recommended: using Conda):**
    ```bash
    conda env create -f environment.yml
    conda activate hep_ml_env # Or your chosen environment name
    ```
    **Alternatively (using pip):**
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

## Dataset Information

*   **Source:** Monte Carlo simulation using MadGraph5_aMC@NLO + Pythia 8 + Delphes 3.
*   **Signal Process:** $pp \to Z \to a\gamma$, with $a \to b\bar{b}$.
*   **Background Process:** Dominated by SM processes mimicking the final state (e.g., $Z+\gamma \to q\bar{q}+\gamma$).
*   **Parameters Studied:**
    *   $m_a$: 15 GeV, 45 GeV
    *   Jet $p_{T,min}$: 10 GeV, 20 GeV
*   **B-Jet Selection:**
    *   "Only first 2": Leading and sub-leading jets if both b-tagged.
    *   "Any two": Requiring at least two b-tagged jets in the event.

## Results Summary

*   BDTs provided a strong baseline (AUC ≈ 0.87) using engineered features.
*   CNNs applied to sparse $\eta-\phi$ images performed poorly (AUC ≈ 0.57).
*   GNNs (NNConv) effectively utilized graph structure and edge features ($\Delta R$), outperforming GCNConv and achieving AUC ≈ 0.89 with refined features (3-node graph, b-tag labels).
*   Sensitivity studies showed performance strongly depends on $m_a$ and $p_{T,min}$. Crucially, $p_{T,min}=20$ GeV kinematically suppresses the $m_a=15$ GeV signal.
*   Optimal performance (AUC ≈ 0.96) was achieved by both BDT and NNConv for $m_a=15$ GeV, $p_{T,min}=10$ GeV, using the "Only first 2 b-tagged jets" selection.
*   The GNN showed potentially greater robustness than BDTs under less ideal b-jet selection criteria.

## License

MIT License

Example:
```markdown
This project is licensed under the MIT License - see the LICENSE file for details.
```

## Citation

If you use this code or findings in your research, please cite the accompanying Master's Thesis:

```bibtex
@mastersthesis{JangraJainThesis2025,
  author       = {Jangra, Kashish and Jain, Meet},
  title        = {Classifying Signal from Background in High Energy Collisions using Machine Learning Approaches},
  school       = {Indian Institute of Technology Delhi},
  year         = {2025},
  month        = {May},
  address      = {New Delhi, India},
  note         = {Supervised by Prof. Abhishek Murlidhar Iyer}
}
```

## Contact

*   Meet Jain - meetjain1818@gmail.com

Project Link: [https://github.com/meetjain1818/Machine-Learning-for-Particle-Physics-Classification.git](https://github.com/meetjain1818/Machine-Learning-for-Particle-Physics-Classification.git)