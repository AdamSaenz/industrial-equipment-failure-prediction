# Industrial Equipment Failure Prediction

**Author:** Adam A. Saenz  
**Course:** DSC680 Applied Data Science  
**Technologies:** Python, Pandas, NumPy, Scikit-learn, Matplotlib, Jupyter Notebook

## Project Overview

This project evaluates machine learning techniques for predicting industrial equipment failures using operational sensor measurements. The objective is to support predictive maintenance decisions by identifying potential equipment failures before they occur.

Two supervised learning algorithms, logistic regression and random forest, were developed and evaluated to compare their ability to identify equipment failures while minimizing unnecessary maintenance alerts.

## Business Problem

Unexpected equipment failures can result in operational disruptions, maintenance expenses, and lost productivity.

Predictive maintenance offers organizations an opportunity to use equipment data to identify potential failures and prioritize inspections. However, inaccurate predictions can create unnecessary maintenance work or allow actual failures to go undetected.

This analysis examines the tradeoffs between detecting failures and limiting false alarms.

## Dataset

**Source:** UCI Machine Learning Repository, AI4I 2020 Predictive Maintenance Dataset

The dataset contains 10,000 synthetic equipment observations, including 339 recorded machine failures.

Predictor variables include:

- Product type
- Air temperature
- Process temperature
- Rotational speed
- Torque
- Tool wear

The target variable is machine failure.

Identifier columns and individual failure mode indicators were excluded from model training to reduce the risk of target leakage.

## Methodology

The analysis follows a structured machine learning workflow:

1. Import and inspect the dataset.
2. Examine data quality and class distribution.
3. Prepare operational variables for modeling.
4. Partition the data into training and testing subsets.
5. Train logistic regression and random forest classifiers.
6. Evaluate performance using cross validation and a held out test set.
7. Compare precision, recall, F1 score, average precision, and ROC AUC.
8. Interpret the operational implications of model performance.

## Model Results

The held out test dataset contains 2,000 observations, including 68 actual equipment failures.

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Precision | 0.142 | 0.796 |
| Recall | 0.824 | 0.574 |
| F1 Score | 0.242 | 0.667 |
| Average Precision | 0.382 | 0.759 |
| ROC AUC | 0.907 | 0.971 |
| Correctly Identified Failures | 56 | 39 |
| False Alarms | 339 | 10 |

Logistic regression identified more actual failures but generated substantially more false alarms.

Random forest achieved stronger precision and overall classification performance but missed more actual failures.

The appropriate model depends on the operational costs associated with missed failures and unnecessary maintenance interventions.

## Key Findings

- Machine learning can identify useful patterns associated with equipment failure in the synthetic dataset.
- Logistic regression provides higher failure detection recall.
- Random forest substantially reduces false positive maintenance alerts.
- Class imbalance makes accuracy alone an insufficient measure of model effectiveness.
- Model selection should account for maintenance costs, equipment criticality, and organizational risk tolerance.

## Limitations and Ethical Considerations

The dataset is synthetic and does not establish how either model would perform in a real industrial environment.

Potential risks include missed equipment failures, unnecessary maintenance activity, changes in operational conditions, and overreliance on automated predictions.

Before operational deployment, the models would require validation using real equipment data, appropriate threshold selection, performance monitoring, and human oversight.

## Reproducing the Analysis

1. Download the Jupyter Notebook and `ai4i2020.csv` from this repository.
2. Place both files in the same folder.
3. Install Python and the required libraries: Pandas, NumPy, Matplotlib, and Scikit-learn.
4. Open the notebook in Jupyter Notebook or JupyterLab.
5. Run the notebook cells sequentially.

The notebook contains the data preparation, modeling, evaluation, and visualization workflow.

## Applications

This project demonstrates analytical techniques relevant to predictive maintenance, operational risk management, equipment monitoring, and data informed maintenance planning.

## Reference

Matzka, S. (2020). *AI4I 2020 Predictive Maintenance Dataset*. UCI Machine Learning Repository. https://doi.org/10.24432/C5HS5C
