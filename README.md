# NutriClass: Food Classification Using Nutritional Data

## Project Overview

NutriClass is a multi-class machine learning project that classifies food into predefined food categories using nutritional and food-related attributes.

The project focuses on data preprocessing, feature engineering, multi-class classification, model comparison, model evaluation, feature importance, and interactive Streamlit deployment.

## Problem Statement

The objective is to build a robust classification system that predicts the most likely food category from a nutritional profile.

Target variable: `Food_Name`

Food categories:
Apple, Banana, Burger, Donut, Ice Cream, Pasta, Pizza, Salad, Steak, Sushi.

## Dataset

Original dataset:
- 31,700 records
- 16 columns

Final cleaned dataset:
- 31,260 records

Input features:
- Calories
- Protein
- Fat
- Carbs
- Sugar
- Fiber
- Sodium
- Cholesterol
- Glycemic_Index
- Water_Content
- Serving_Size
- Meal_Type
- Preparation_Method
- Is_Vegan
- Is_Gluten_Free

## Methodology

1. Data understanding and exploratory analysis
2. Missing-value handling
3. Duplicate removal
4. IQR-based outlier detection
5. Training-data-only IQR outlier capping
6. Numerical standardization using `StandardScaler`
7. Categorical encoding using `OneHotEncoder`
8. Multi-class model training and comparison
9. Feature-selection experiment using `SelectKBest`
10. Final model selection
11. Five-fold stratified cross-validation
12. Feature importance analysis
13. Streamlit deployment

## Models Evaluated

- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors
- Support Vector Machine
- XGBoost
- Gradient Boosting

## Final Model

**Gradient Boosting Classifier + IQR Outlier Capping + Full processed feature set**

### Holdout Performance

| Metric | Score |
|---|---:|
| Accuracy | 99.50% |
| Precision | 99.50% |
| Recall | 99.50% |
| Weighted F1 | 99.50% |
| Macro F1 | 99.52% |

Holdout test set:
- 6,252 records
- 6,221 correct predictions
- 31 misclassifications

### Cross-Validation

Five-fold stratified cross-validation:
- Mean Macro F1: 99.54%
- Standard deviation: 0.09 percentage points

## Feature Selection Experiment

The preprocessing stage produced 21 features. `SelectKBest` reduced this to 15 features.

Feature-selection model:
- Accuracy: 99.46%
- Macro F1: 99.54%

The full-feature Gradient Boosting model was retained as the final model because it maintained higher holdout accuracy.

## Key Feature Importance Findings

Top contributors in the final model:
1. Serving Size
2. Sodium
3. Cholesterol
4. Protein
5. Is Gluten Free

## Streamlit Application

The application provides:
1. **Quick Prediction** — consumer-friendly inputs
2. **Full Model Input** — all 15 model features for evaluator testing
3. **Food + Quantity Lookup** — select a food and quantity, view a representative profile, and classify it

The saved model is a complete preprocessing + Gradient Boosting pipeline, so a processed CSV is not required for prediction.

## Business Use Cases

- Smart dietary applications
- Health monitoring and diet planning
- Automated food logging
- Nutrition education
- Grocery and meal planning

## Limitations

- The model classifies only the 10 food categories represented in the training data.
- A food category absent from the training data cannot be reliably identified.
- Prediction confidence depends on how closely a new nutritional profile resembles the training distribution.
- The model supports classification and is not a medical diagnosis.

## Project Structure

```text
NutriClass/
├── README.md
├── .gitignore
├── requirements.txt
├── app.py
├── notebooks/
│   └── NutriClass_Final_Presentation.ipynb
├── models/
│   └── nutriclass_final_model.pkl
├── data/
│   └── synthetic_food_dataset_imbalanced.csv
├── reports/
│   └── NutriClass_Final_Project_Report.docx
└── screenshots/
```

## Run the Streamlit Application

Use Python 3.12.x.

```powershell
python -m streamlit run app.py
```

## Skills Demonstrated

Python, Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn, Streamlit, data preprocessing, feature engineering, multi-class classification, model evaluation, cross-validation, and model explainability.
