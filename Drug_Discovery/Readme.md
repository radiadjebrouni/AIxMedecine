# Drug Discovery: Targeting Beta-secretase 1 for Alzheimer's disease

## Overview
This project focuses on the use of Random Forest models for drug discovery, specifically targeting bioactivity classification and regression tasks related to Beta-secretase 1. The notebook encompasses various stages including data preprocessing, exploratory data analysis, descriptor calculation, and model building. The models are applied to predict bioactivity and pIC50 values of compounds using machine learning techniques.


![250px-Protein_BACE1_PDB_1fkn](https://github.com/user-attachments/assets/52ab5dbc-cbae-4f47-b436-b2c9f2fbb8f0)


## Table of Contents

1. [Retrieving and Preprocessing Bioactivity Data](#retrieving-and-preprocessing-bioactivity-data)
2. [Exploratory Data Analysis](#exploratory-data-analysis)
3. [Descriptor Calculation and Dataset Preparation](#descriptor-calculation-and-dataset-preparation)
4. [Model Building](#model-building)

## Retrieving and Preprocessing Bioactivity Data

This section describes how bioactivity data is collected and preprocessed. It involves handling missing values, encoding categorical variables, and splitting the data into training and test sets.

## Exploratory Data Analysis

In this section, we perform exploratory data analysis to understand the distribution of various molecular descriptors. Key statistical tests, such as the Mann-Whitney U test, are used to identify significant differences between active and inactive compounds.

### Statistical Analysis: Mann-Whitney U Test

The Mann-Whitney U test is a non-parametric test used to assess whether there is a significant difference between the distributions of two independent samples. The test results are saved as CSV files for further analysis.

## Descriptor Calculation and Dataset Preparation

Molecular descriptors are calculated using PaDEL-Descriptor to quantify the properties of molecules. The dataset is then prepared for model building by concatenating these descriptors with bioactivity labels and splitting into regression and classification datasets.

## Model Building

### Regression Model with Random Forest

A Random Forest Regressor is trained to predict pIC50 values, providing insights into the potency of compounds. Model performance is evaluated using R-squared metrics and visualized with regression plots.

### Classification Model with Random Forest

A Random Forest Classifier is trained to categorize compounds as "Active" or "Inactive." Class imbalance is addressed through class weighting, and model performance is assessed using classification metrics and confusion matrices.

### Hyperparameter Tuning

Hyperparameter tuning is performed using GridSearchCV to optimize the Random Forest model's performance. Cross-validation is used to ensure the model generalizes well.

## Models

The trained models can be saved and loaded using the following commands:

```python
import joblib

# Save the regression model
joblib.dump(model, 'random_forest_regression_model.pkl')

# Save the classification model
joblib.dump(model_class, 'random_forest_classification_model.pkl')
```

To load the models:
```python
import joblib

# Load the regression model
model = joblib.load('random_forest_regression_model.pkl')

# Load the classification model
model_class = joblib.load('random_forest_classification_model.pkl')
```

**Dependencies**
* pandas
* numpy
* scikit-learn
* seaborn
* matplotlib
* joblib
* padelpy (for descriptor calculation)

  **Ressources**
  
  https://github.com/dataprofessor/bioactivity-prediction-app/tree/main/PaDEL-Descriptor
  https://pubmed.ncbi.nlm.nih.gov/15126696/
  https://www.collaborativedrug.com/cdd-blog/why-using-pic50-instead-of-ic50-will-change-your-life
  https://peerj.com/articles/2322/
