# Smoking Behaviour Prediction — UK Survey Data

A comprehensive machine learning analysis of smoking behaviour in the UK, using survey data on 1,691 respondents. Combines unsupervised learning (k-means clustering), multiple supervised models (Linear Regression, Logistic Regression, CART, Random Forest), and rigorous statistical diagnostics to identify the key demographic and socioeconomic predictors of smoking.

## Problem Statement

Smoking remains a critical public health concern in the UK. Understanding which demographic and socioeconomic factors predict smoking behaviour is essential for designing effective public health interventions. This project applies both supervised and unsupervised machine learning techniques to a UK smoking survey to uncover these patterns.

## Dataset

**Source:** [Smoking Dataset from UK](https://www.kaggle.com/datasets/utkarshx27/smoking-dataset-from-uk) (1,691 observations, 12 variables)

Features include gender, age, marital status, highest qualification, nationality, ethnicity, gross income, region, smoking status, cigarettes smoked on weekdays/weekends, and cigarette type.

## Methodology

### Exploratory Data Analysis
- Bar plots to visualise smoking distribution across all demographic variables
- Identified notable skewness in nationality (British/English dominance) and ethnicity (predominantly White respondents)

### Unsupervised Learning — K-Means Clustering
- Clustered respondents by age and smoking amount (weekday + weekend)
- Determined optimal cluster count (k=7) using silhouette scores
- Visualised distinct smoking behaviour groups by age and consumption level

### Supervised Learning — Model Comparison

| Model | Accuracy | Notes |
|---|---|---|
| Linear Regression (reduced) | 74.26% | Gender, age, marital status, qualification, income |
| Linear Regression (full) | 75.14% | Added nationality, ethnicity, region |
| CART Decision Tree | 72.49% | Marital status as primary split |
| Logistic Regression | 72.78% | Binary classification on smoke status |
| Random Forest (500 trees) | ~100% OOB | All variables including smoking amounts |

### Statistical Diagnostics
- **Breusch-Pagan test** — detected heteroscedasticity in initial model; resolved after removing influential observations via Cook's Distance
- **Shapiro-Wilk test** — assessed normality of residuals
- **Cook's Distance** — identified and removed 81 influential observations
- **VIF analysis** — flagged multicollinearity between marital status and age
- **Chi-square tests** — confirmed significant associations between smoking and age, marital status, qualification, income, and region
- **T-test** — smokers are significantly younger (mean 42.7 yrs) than non-smokers (mean 52.2 yrs)

## Key Findings

- **Age** is the strongest demographic predictor — younger individuals are significantly more likely to smoke
- **Marital status** is highly influential — married/widowed individuals smoke less; divorced/separated/single individuals smoke more
- **Education level** matters — individuals with A Levels or degrees are less likely to smoke
- **Gender and ethnicity** showed no statistically significant association with smoking behaviour
- **Income and region** have moderate but significant effects

## Tech Stack

- **R** — tidyverse, ggplot2, randomForest, rpart, caret, car, lmtest
- **Models** — K-Means Clustering, Linear Regression, Logistic Regression, CART, Random Forest
- **Diagnostics** — Breusch-Pagan, Shapiro-Wilk, Cook's Distance, VIF, Chi-square, T-test

## Project Structure

```
smoking-behaviour-prediction/
├── smoking_analysis.R    # Full analysis script
├── data/                 # Place dataset here
│   └── smoking.csv
└── README.md
```

## How to Run

```r
install.packages(c("tidyverse", "naniar", "cluster", "caTools",
                    "car", "lmtest", "randomForest", "rpart", "rpart.plot"))
```

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/utkarshx27/smoking-dataset-from-uk) and place it in `data/`
2. Run in RStudio:
```r
source("smoking_analysis.R")
```
