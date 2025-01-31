# Graph Neural Networks for Particle Collision Classification  
**Signal vs. Background Event Discrimination Using GNNs**  

---

## 📌 Overview  
This repository contains the implementation of a **Graph Neural Network (GNN)**-based framework for binary classification of particle collision events simulated using MadGraph, Pythia, and Delphes. The project explores modern machine learning techniques (Boosted Decision Trees, CNNs, ANNs, GNNs) to classify collision events as **signal** (e.g., rare physics processes) or **background** (Standard Model processes). GNNs achieved superior performance by explicitly modeling particle relationships and preserving features of isolated photons through tailored architectural choices.  

![GNN Message Passing](docs/gnn_schematic.png)  
*Figure: Message-passing mechanism in GNNs for particle collision data.*

---

## ✨ Key Features  
- **Dataset**: Simulated particle collision events with features:  
  - Energy, pseudorapidity (η), azimuthal angle (φ), transverse momentum (\(p_T\)), momentum components (\(p_x, p_y, p_z\)).  
  - Signal/Background labels for binary classification.  
- **Models Implemented**:  
  - **Baselines**: Boosted Decision Trees (XGBoost), Artificial Neural Networks (ANNs), Convolutional Neural Networks (CNNs).  
  - **GNN Variants**: Graph Convolutional Networks (GCN), Graph Attention Networks (GAT), NNConv (edge-aware convolution).  
- **Innovations**:  
  - Graph construction with isolated photons (disconnected nodes) to prevent feature dilution.  
  - Global max pooling for graph-level embeddings to preserve critical isolated features.  
  - Dynamic edge feature handling via `NNConv` (PyTorch Geometric).  

---

## 🛠️ Installation  
1. **Clone the repository**:  
   ```bash  
   git clone https://github.com/<your-username>/particle-collision-gnn.git  
   cd particle-collision-gnn  
   ```  

2. **Install dependencies**:  
   ```bash  
   conda create -n gnn python=3.8  
   conda activate gnn  
   pip install -r requirements.txt  
   ```  
   **Key Libraries**:  
   - PyTorch Geometric  
   - PyTorch  
   - XGBoost  
   - scikit-learn  
   - pandas, numpy
   - TensorFLow 

3. **Download Dataset**:  
   - Follow instructions in `data/README.md` to download/preprocess the Delphes-simulated dataset.  

---

## 🚀 Usage  

### 1. Data Preprocessing  
```python  
python scripts/preprocess.py --data_dir ./data/raw --output_dir ./data/processed  
```  
- Converts raw Delphes ROOT files into graph-structured data (`.pt` files).  
- Isolated photons are disconnected by default.  

### 2. Training Models  
**Example: Train a GNN with NNConv layers**  
```python  
python train.py --model nnconv --hidden_channels 64 --lr 0.001 --epochs 100  
```  
**Supported Models**: `--model [gcn, gat, nnconv, xgboost, cnn]`  

### 3. Evaluation  
```python  
python evaluate.py --checkpoint runs/nnconv/best_model.pt --test_data data/processed/test.pt  
```  
- Reports metrics: AUC-ROC, F1-score, accuracy, confusion matrices.  

---

## 📊 Results  
| Model       | AUC-ROC | F1-Score | Accuracy |  
|-------------|---------|----------|----------|  
| XGBoost     | 0.82    | 0.78     | 0.81     |  
| CNN         | 0.85    | 0.79     | 0.83     |  
| GCN         | 0.88    | 0.83     | 0.86     |  
| **GAT**     | **0.92**| **0.89** | **0.90** |  

**Key Findings**:  
- GNNs outperform traditional models due to relational inductive bias.  
- Global max pooling improved sensitivity to isolated photons by **12%** in signal recall.  

---

## 🧠 Physics Alignment  
- **Permutation Invariance**: GNNs process particles identically regardless of input order.  
- **Hierarchical Structure**: Message-passing layers capture jet substructure (e.g., subjets from boosted \(W\) bosons).  
- **Detector Geometry**: Spatial edges (\(\Delta R < 0.4\)) mimic calorimeter clustering.  

---

## 📚 References  
1. ParticleNet: [arXiv:1902.08570](https://arxiv.org/abs/1902.08570)  
2. PyTorch Geometric: [Documentation](https://pytorch-geometric.readthedocs.io)  
3. Delphes Simulation Framework: [GitHub](https://github.com/delphes/delphes)  

---

## 🙏 Acknowledgments  
- Supervised by **Prof. Abhishek M. Iyer** at Indian Institute of Technology Delhi.  
- Dataset generated using MadGraph-Pythia-Delphes workflow.  

--- 

📧 **Contact**: [meetjain1818@gmail.com]  
📜 **License**: MIT  
```  

---

### Suggested Repository Structure:  
```  
.  
├── data/  
│   ├── raw/              # Raw ROOT files  
│   └── processed/        # Preprocessed graphs (.pt files)  
├── models/               # Model definitions (GCN, GAT, NNConv, etc.)  
├── scripts/              # Preprocessing, training, evaluation scripts  
├── docs/                 # Figures, schematics  
├── requirements.txt  
└── README.md  
```  

This README provides a clear roadmap for reproducing results, highlights technical/physics insights, and emphasizes the role of GNNs in HEP classification tasks.