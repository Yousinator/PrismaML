# PrismaML Development Guide

## Table of Contents

- [Introduction](#introduction)
- [Setting Up Your Development Environment](#setting-up-your-development-environment)
- [Feature Development Checklist](#feature-development-checklist)
  - [Data Analysis and Visualization (dataset_information/)](#data-analysis-and-visualization-dataset_information)
  - [Machine Learning (machine_learning/)](#machine-learning-machine_learning)
  - [Plotting (plotting/)](#plotting-plotting)
- [Testing](#testing)
- [Code Style and Contribution Guidelines](#code-style-and-contribution-guidelines)
- [Continuous Integration](#continuous-integration)
- [Documentation](#documentation)
- [Future Directions](#future-directions)
  - [DataPreprocessing](#datapreprocessing)
  - [FeatureEngineering](#featureengineering)
  - [ModelBuilder](#modelbuilder)
  - [AutomatedML](#automatedml)
- [License](#license)

## Introduction

Welcome to the PrismaML development guide. This document provides a structured approach for developers contributing to PrismaML. Here, you will find detailed instructions on setting up your development environment, a checklist of tasks categorized by project component, and guidelines for coding and contributions.

> [!Important]
> These are few of the features that are planned to be implemented, you can also suggest new features and work on them. Open an issue and discuss the feature you want to implement.

## Setting Up Your Development Environment

To contribute effectively, set up your development environment according to the specifications in the [Development Environment Setup Guide](https://github.com/Yousinator/PrismaML/docs/installation.md). Additionally, ensure you have the following tools configured:

- **Integrated Development Environment (IDE)**: We recommend using PyCharm or Visual Studio Code with Python extensions installed.
- **Code Linters**: Use Flake8 for Python to maintain code quality.
- **Version Control**: Contributions should be managed through Git. Familiarity with basic Git operations is essential.

## Feature Development Checklist

Development tasks are categorized by project modules to streamline the contribution process.

### Data Analysis and Visualization (dataset_information/)

- **DataFrame Summary Enhancements**:

  - Improve the `dataframe_summary` method to include more detailed statistics and visualizations.
  - Add support for custom metadata to enhance the summary output.

- **Categorical Data Analysis**:

  - Enhance the `categorical_summary` method to provide more detailed insights into categorical data.
  - Include visualizations like bar plots and pie charts for better representation.

- **Numerical Data Analysis**:
  - Improve the `numerical_summary` method to include advanced statistical measures.
  - Add correlation analysis and visualizations like heatmaps and pair plots.

### Machine Learning (machine_learning/)

- **Feature Selection Optimization**:

  - Refine the `select_best_features` method to support more feature selection techniques.
  - Optimize the process for large datasets to improve performance.

- **Model Evaluation Enhancements**:

  - Enhance the `evaluate_model` method to support a wider range of metrics and evaluation techniques.
  - Implement cross-validation and other resampling methods for more robust evaluation.

- **Visualization Improvements**:
  - Improve the `plot_accuracy_vs_features` and `plot_iteration_metrics` methods to provide more insightful and interactive visualizations.
  - Add support for saving plots in various formats.

### Plotting (plotting/)

- **Categorical Plot Enhancements**:

  - Enhance the `draw_categorical_plots` method to support more plot types and customization options.
  - Add interactive plot capabilities using libraries like Plotly.

- **Numerical Plot Enhancements**:

  - Improve the `draw_numerical_plots` method to include more plot types and better handling of large datasets.
  - Add customization options for histograms and KDE plots.

- **Algorithm Comparison Visualization**:
  - Enhance the `plot_algorithm_comparison` method to support more metrics and better visualization techniques.
  - Implement features for dynamic updates and comparisons.

## Testing

- Develop and run comprehensive tests for all new features or updates using pytest.
- Ensure all tests pass before submitting pull requests.

## Code Style and Contribution Guidelines

- Follow PEP8 standards for Python code.
- Document all changes comprehensively; use clear, descriptive commit messages.

## Continuous Integration

- Implement CI workflows with GitHub Actions to automate tests, lint checks, and builds upon new commits and pull requests.

## Documentation

- Keep all project documentation current to reflect feature changes or additions.
- Maintain high-quality inline and API documentation for ease of maintenance and future development.

## Future Directions

To further enhance PrismaML and expand its capabilities, we propose the development of the following new classes. These classes are designed to streamline data analysis, improve model building, and provide advanced visualization techniques.

1. DataPreprocessing

- **Purpose**: Automate and standardize data preprocessing tasks to prepare datasets for analysis and model building.
- **Features**:
  - Handle missing values with various strategies (mean, median, mode, interpolation).
  - Encode categorical variables using techniques like one-hot encoding and label encoding.
  - Normalize and standardize numerical features.
  - Detect and handle outliers.

```python
class DataPreprocessing:
  def __init__(self, df):
    self.df = df

  def handle_missing_values(self, strategy='mean'):
      # Implement missing value handling logic
      pass

  def encode_categorical(self, method='one-hot'):
      # Implement categorical encoding logic
      pass

  def normalize_features(self, method='z-score'):
      # Implement feature normalization logic
      pass

  def handle_outliers(self, method='iqr'):
      # Implement outlier handling logic
      pass
```

2. FeatureEngineering

- **Purpose**: Provide tools for creating new features and transforming existing features to enhance model performance.

- **Features**:
  - Generate polynomial features for numerical columns.
  - Create interaction terms between features.
  - Perform feature scaling and binning.
  - Automate feature selection using techniques like recursive feature elimination.

```python
class FeatureEngineering:
  def __init__(self, df):
    self.df = df

  def generate_polynomial_features(self, degree=2):
      # Implement polynomial feature generation
      pass

  def create_interaction_terms(self):
      # Implement interaction term creation
      pass

  def scale_features(self, method='min-max'):
      # Implement feature scaling logic
      pass

  def bin_features(self, bins=5):
      # Implement feature binning logic
      pass

  def select_features(self, method='rfe'):
      # Implement feature selection logic
      pass
```

3. ModelBuilder

- **Purpose**: Simplify the process of building, training, and evaluating machine learning models.

- **Features**:
  - Automate model selection and hyperparameter tuning.
  - Provide easy-to-use methods for training and evaluating models.
  - Support multiple model types (classification, regression, clustering).
  - Integrate with popular libraries like scikit-learn, XGBoost, and TensorFlow.

```python
class ModelBuilder:
  def __init__(self):
    pass

  def train_model(self, X, y, model_type='classification'):
      # Implement model training logic
      pass

  def evaluate_model(self, X, y, model):
      # Implement model evaluation logic
      pass

  def tune_hyperparameters(self, model, param_grid):
      # Implement hyperparameter tuning logic
      pass

  def save_model(self, model, filepath):
      # Implement model saving logic
      pass

  def load_model(self, filepath):
      # Implement model loading logic
      pass
```

4. AutomatedML

- **Purpose**: Automate the end-to-end machine learning workflow, from data preprocessing to model deployment.
- **Features**:
  - Perform automated feature engineering and selection.
  - Train and evaluate multiple models using AutoML techniques.
  - Optimize model hyperparameters automatically.
  - Deploy the best-performing model to a production environment.

```python
class AutomatedML:
  def __init__(self, df, target):
      self.df = df
      self.target = target

  def preprocess_data(self):
      # Implement automated data preprocessing
      pass

  def train_models(self):
      # Implement automated model training
      pass

  def evaluate_models(self):
      # Implement automated model evaluation
      pass

  def deploy_best_model(self, deployment_target):
      # Implement model deployment logic
      pass
```

## License

All contributions to this project must comply with the terms outlined in the LICENSE file. For more details, refer to the full license documentation.
