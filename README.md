# Support Vector Machine (SVM): Predicting Heart Disease

**Group 6 · Classification Algorithms Exploration (mini-project)**

Can a Support Vector Machine predict whether a patient has heart disease? This project explores SVM end to end: exploratory data analysis, preprocessing, model training, tuning, and a recall-focused model built for a medical setting.

## Dataset

- **Source:** [Heart Failure Prediction (Kaggle)]([https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction))
- **Size:** 918 patients, 11 features, 1 target (917 patients after cleaning)
- **Target:** `HeartDisease` (1 = heart disease, 0 = normal), about 55% / 45%

| Column | Description |
|---|---|
| `Age` | Age of the patient (years) |
| `Sex` | M / F |
| `ChestPainType` | TA, ATA, NAP, ASY |
| `RestingBP` | Resting blood pressure (mm Hg) |
| `Cholesterol` | Serum cholesterol (mg/dl) |
| `FastingBS` | Fasting blood sugar > 120 mg/dl (1 = yes) |
| `RestingECG` | Resting ECG result: Normal, ST, LVH |
| `MaxHR` | Maximum heart rate reached |
| `ExerciseAngina` | Chest pain during exercise (Y / N) |
| `Oldpeak` | ST depression during exercise |
| `ST_Slope` | Slope of the ST segment: Up, Flat, Down |
| `HeartDisease` | **Target** |

## Project Workflow

1. **EDA:** data quality, target balance, outliers, and four hypotheses (sex, age, exercise angina, cholesterol).
2. **Preprocessing:** fix hidden zeros (`Cholesterol = 0` replaced by the median, one `RestingBP = 0` row removed), one-hot encoding, stratified 80/20 split, `StandardScaler` fitted on the training set only.
3. **Models:** SVM without scaling, SVM with scaling, tuned SVM (`GridSearchCV`), Logistic Regression for comparison, and a recall-focused SVM.
4. **Recall-focused SVM:** in medicine, missing a sick patient is the most dangerous mistake. This model uses `class_weight` (a sick patient counts double) and picks the best model with the **F2 score**, which gives recall more weight than precision.

## Key EDA Findings

- Strongest signs of heart disease: `ST_Slope` (flat), exercise angina and `Oldpeak`.
- Men (63% vs 26%) and older patients (55.9 vs 50.6 years on average) are at higher risk.
- Patients with **no typical chest pain symptoms (ASY)** have the highest rate of heart disease (79%).
- Cholesterol has almost no effect (correlation 0.08).
- Outliers were kept: they are medically possible values.

## Results (test set)

| Model | Accuracy | Recall |
|---|---|---|
| SVM, no scaling | 0.734 | 0.804 |
| SVM, scaled | 0.859 | 0.892 |
| SVM, tuned (`rbf`, `C=1`, `gamma=scale`) | 0.859 | 0.892 |
| Logistic Regression | 0.875 | 0.892 |
| **SVM, recall-focused** (`poly`, `C=0.1`, `gamma=0.1`, `class_weight={0: 1, 1: 2}`) | 0.826 | **0.941** |

The recall-focused SVM finds 96 of the 102 sick patients in the test set and misses only 6, at the cost of more false alarms (26 instead of 15).

## Repository Structure

```
├── README.md
└── MiniProject_Group6_SVM_final.ipynb   # EDA, modelling and evaluation
```

## How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn kagglehub jupyter
jupyter notebook MiniProject_Group6_SVM_final.ipynb
```

The notebook downloads the dataset from Kaggle with `kagglehub`.

## Team (Group 6)

- Nouf Turki
- Nouf Alotaibi
- Mohammed
- Razan
