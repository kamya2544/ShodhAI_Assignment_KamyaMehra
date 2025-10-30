# Loan Approval Optimization — DL + Offline RL

The end-to-end pipeline to optimize loan approvals using:

Task 1: Exploratory Data Analysis (EDA) & Preprocessing
Task 2: Deep Learning default-risk model (PyTorch MLP)
Task 3: Offline Reinforcement Learning (CQL via d3rlpy) for profit-maximizing approvals
Task 4: Analysis & comparison (AUC/F1 vs Estimated Policy Value), disagreement cases, and future steps

We have reproduced all results locally (recommended Python 3.11)

## 1) Environment Setup
### Using Conda (Windows) 
Use Python 3.11. 
### From the repo root
conda create -n loan-rl python=3.11 -y
conda activate loan-rl
### Install dependencies
python -m pip install -U pip
pip install -r requirements.txt
### Register this env as a Jupyter kernel
python -m ipykernel install --user --name loan-rl --display-name "Python 3.11 (loan-rl)"

### Launch Jupyter and select the kernel:
jupyter notebook
In the UI: Kernel → Change Kernel → "Python 3.11 (loan-rl)"

## 2) Data Files
Place the following in data/:
Preferred: loan_clean_subset.csv (produced by Task 1).
Contains preprocessed features and a default column (0=Fully Paid, 1=Defaulted).
Optional raw: accepted_2007_to_2018Q4.csv.gz.
Ensure loan_amnt and int_rate columns are present in the clean file (used for RL rewards).

## 3) Procedure
### Step 1 — Task 1: EDA & Preprocessing
Open notebooks/Task1_EDA_Preprocess.ipynb and Run All:
Loads raw CSV (if present) → cleans & engineers features → saves data/loan_clean_subset.csv
Or validate the provided clean file.

Outputs:
data/loan_clean_subset.csv

### Step 2 — Task 2: Deep Learning (DL) Model
Open notebooks/Task2_DL_Predictive_Model.ipynb and Run All:
Loads loan_clean_subset.csv
Builds preprocess (impute + OHE + scale)
Trains PyTorch MLP (or loads existing)
Evaluates ROC-AUC and F1 on test set

Outputs:
models/model_mlp_default_risk/pytorch_mlp.pt (DL weights)
Printed metrics (AUC, F1, classification report, confusion matrix)

### Step 3 — Task 3: Offline RL (CQL)
Open notebooks/Task3_Offline_RL_CQL.ipynb and Run All:
Builds bandit-style MDPDataset (duplicates each state with actions {deny, approve})
Trains CQL agent with d3rlpy (code handles version differences)
Computes Estimated Policy Value (EPV) on the held-out test set

Outputs:
models/offline_rl_cql/cql_discrete_model.d3 (if saving is supported by your d3rlpy build)
models/offline_rl_cql/preprocess.joblib (sklearn preprocessor)
Printed EPV, approval rate, approval breakdown

### Step 4 — Task 4: Analysis & Comparison
Open your notebooks/Task4_Analysis_Comparison.ipynb and Run All:
Re-evaluates DL (AUC, F1)
Computes RL EPV (and FQE if available)
Finds policy disagreements and shows examples
Prints a one-page final report text block you can paste into your submission

Outputs:
Final printed metrics (AUC, F1, EPV, approval rate)
Disagreement examples table
Final report text in the notebook output


## 4) Insights Available
DL metrics (test): ROC-AUC and F1 printed with a classification report + confusion matrix
RL metrics: Estimated Policy Value (EPV) per application, approval rate, counts of approved-paid and approved-defaulted
Disagreements table: rows where DL and RL differ, with indicative fields and RL reward
Final report: auto-generated text block summarizing results and next steps

## 5) Citations / Libraries
PyTorch — DL model
scikit-learn — preprocessing, metrics, split
d3rlpy — Offline RL (CQL), optional OPE (FQE)
gymnasium — env dependencies for RL wrappers


### Jupyter kernel support (optional but useful locally)
ipykernel>=6.29.0
