# Diabetes Prediction Using K-Nearest Neighbors (KNN)

## Overview

This project develops a machine-learning classification model to predict whether a subject is classified as having diabetes using **K-Nearest Neighbors (KNN)**.

The project demonstrates a complete, reproducible ML workflow:

- Exploratory data analysis
- Data-quality checks
- Feature/target separation
- Stratified train/test splitting
- Feature standardization
- KNN classification
- Hyperparameter tuning with cross-validation
- Evaluation using accuracy, precision, recall, F1-score, and confusion matrices
- Comparison of accuracy-based and F1-based model selection
- Reusable model-training code

> **Educational project:** This model is not a medical diagnostic tool and should not be used for clinical decision-making.

---

## Project Structure

```text
diabetes-prediction-knn/
│
├── data/
│   └── Dataset_Diabetes_Prediction.csv
│
├── notebooks/
│   └── diabetes_knn_analysis.ipynb
│
├── src/
│   └── train_model.py
│
├── models/
│   └── knn_diabetes_model.joblib
│
├── reports/
│   ├── figures/
│   └── model_comparison.csv
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Dataset

The notebook supplied for this project contains a dataset with 1,000 observations and these variables:

| Feature | Description |
|---|---|
| Pregnancies | Number of pregnancies |
| Glucose | Glucose measurement |
| BloodPressure | Blood pressure measurement |
| SkinThickness | Skin-thickness measurement |
| Insulin | Insulin measurement |
| BMI | Body mass index |
| DiabetesPedigreeFunction | Diabetes pedigree function |
| Age | Age |
| Diagnosis | Target: 0 = No Diabetes, 1 = Diabetes |

The observed target distribution in the supplied notebook was:

- No Diabetes: 69.4%
- Diabetes: 30.6%

The notebook also identified unusual observed values, including negative values in some numeric fields. These should be investigated against the dataset's source documentation before treating the data as clinically valid.

## Methodology

### 1. Train/test split

The data is split into:

- 80% training
- 20% testing

The split is stratified by `Diagnosis` so that class proportions are preserved.

### 2. Feature scaling

KNN relies on distances between observations. `StandardScaler` is therefore used so that variables measured on different scales do not dominate the distance calculation.

The scaler is placed inside a scikit-learn `Pipeline` to prevent preprocessing leakage during cross-validation.

### 3. Baseline KNN

The first model searches K values from 1 through 30 using 5-fold stratified cross-validation and selects K using **accuracy**.

### 4. F1-tuned KNN

A second search selects K and KNN weighting (`uniform` or `distance`) using **F1-score for the Diabetes class**.

This is included because accuracy can be misleading when the target classes are imbalanced.

## Evaluation

The project reports:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

For the Diabetes class, recall is particularly important to inspect because it describes how many actual positive cases were identified.

The project intentionally does not rely on accuracy as the only measure of model quality.

## Running the Project

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd diabetes-prediction-knn
```

### 2. Create an environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the dataset

Place the CSV here:

```text
data/Dataset_Diabetes_Prediction.csv
```

### 5. Run the training script

```bash
python src/train_model.py
```

The script prints model metrics and saves the trained F1-tuned model to:

```text
models/knn_diabetes_model.joblib
```

### 6. Run the notebook

```bash
jupyter notebook notebooks/diabetes_knn_analysis.ipynb
```

## Key Learning Points

This project demonstrates several important machine-learning concepts:

1. **KNN is distance-based**, so feature scaling matters.
2. **Cross-validation** provides a more robust basis for selecting K than evaluating many choices directly on the test set.
3. **Accuracy can be misleading** when one class is more common than another.
4. **Recall and F1-score** provide additional information about positive-class performance.
5. A model can have reasonable accuracy while failing to identify the minority class.
6. A reusable training script makes the notebook analysis easier to reproduce.

## Limitations

- The project uses the supplied dataset and does not establish clinical validity.
- The dataset's collection process and population are not documented in the notebook.
- Some observed numeric values require source-level investigation for plausibility.
- KNN performance may change with a different sample, preprocessing method, feature set, or validation strategy.
- The model should not be interpreted as a medical diagnosis.

## Future Improvements

Potential extensions include:

- Compare KNN with Logistic Regression, Random Forest, and Gradient Boosting.
- Evaluate ROC-AUC and PR-AUC.
- Perform feature-selection experiments.
- Investigate the unusual numeric values using the dataset's source documentation.
- Tune the probability/classification threshold when appropriate.
- Add cross-validated performance intervals.
- Add a small prediction interface for educational demonstration only.
