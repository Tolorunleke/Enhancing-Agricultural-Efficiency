# Regression with an Abalone Dataset

This repository contains the code and resources for the Kaggle competition **[Regression with an Abalone Dataset](https://kaggle.com/competitions/playground-series-s4e4)**, authored by Walter Reade and Ashley Chow (2024). The goal of this competition is to predict the age of abalone, a type of sea snail, using various physical measurements. The evaluation metric for this task is **Root Mean Squared Logarithmic Error (RMSLE)**.

## 📊 Project Overview

### 🔍 Data Loading and Exploration
- The Abalone dataset is loaded, containing multiple physical measurements of abalone, including:
  - **Length**: Longest shell measurement (in mm).
  - **Diameter**: Perpendicular to the length (in mm).
  - **Height**: Height with meat (in mm).
  - **Whole weight**: Weight of the whole abalone.
  - **Shucked weight**: Weight of the meat.
  - **Viscera weight**: Gut weight (after bleeding).
  - **Shell weight**: Weight after drying.
  - **Rings**: Count of growth rings (used to calculate the age).
- Initial exploratory data analysis (EDA) is conducted to understand the data distribution, identify potential outliers, and prepare the data for modeling.

### 🛠️ Data Preprocessing
- Data preprocessing includes:
  - Handling missing values and outliers.
  - Normalizing and standardizing numerical features to ensure all features contribute equally to the model.
  - Encoding any categorical variables if present.
  - Splitting the data into training and test sets using `train_test_split`.

### 🧠 Building Regression Models
- Multiple regression models are built to predict the age of abalone, including:
  - **K-Nearest Neighbors (KNN) Regression**
  - **Decision Tree Regression**
  - **Random Forest Regression**
  - **Gradient Boosting Regression**
  - **AdaBoost Regression**
- Each model is trained and optimized using techniques such as cross-validation and grid search to fine-tune hyperparameters.

### 🎯 Model Evaluation
- The primary evaluation metric is **Root Mean Squared Logarithmic Error (RMSLE)**.
- Additional metrics like **Mean Squared Error (MSE)** are also computed to understand the model performance better.
- Visualizations, including plots of feature importance, residuals, and error distributions, are used to interpret and validate the results.

### 🏆 Results and Insights
- The model performances are compared, and the best model is selected based on the lowest RMSLE score.
- Key findings, insights, and recommendations for improving the model are discussed.

## 🚀 Getting Started

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Tolorunleke/Regression-ML-Model-on-Abalones.git
## 🤝 Acknowledgments
- This project is part of a Kaggle competition, Regression with an Abalone Dataset, by Walter Reade and Ashley Chow (2024). https://kaggle.com/competitions/playground-series-s4e4
