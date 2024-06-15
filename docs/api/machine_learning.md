# Detailed Documentation for `MachineLearning` Class

## Table of Contents

- [select_best_features()](#select_best_features)
- [plot_accuracy_vs_features()](#plot_accuracy_vs_features)
- [evaluate_model()](#evaluate_model)
- [plot_iteration_metrics()](#plot_iteration_metrics)
- [Notes](#notes)

## select_best_features(self, X, y, model, feature_selector)

**Functionality**:<br>
This method evaluates and selects the best features for the given machine learning model based on the performance metric (F1 score for classification or R2 score for regression).

**Usage**:<br>

- The method splits the dataset into training and testing sets.
- It iteratively applies the feature selection strategy to the dataset, fits the specified machine learning model, and evaluates its performance.
- It tracks performance metrics for each number of features used and identifies the optimal number of features that yield the best performance score.
- Additionally, it returns the names of the best features along with the fitted feature selector for the best model.

**Example**:<br>

```python
ml = MachineLearning()
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, f_regression
model = LinearRegression()
feature_selector = lambda k: SelectKBest(f_regression, k=k)
best_features, results, best_feature_names, best_selector = ml.select_best_features(X, y, model, feature_selector)
print("Best number of features:", best_features)
print("Best features:", best_feature_names)
```

**Detailed Steps**:<br>

1. **Splitting the Dataset**: The method splits the dataset into training and testing sets using train_test_split.
2. **Feature Selection Loop**:
   - Iterates over the number of features from 1 to the total number of features.
   - For each iteration, applies the feature selector to select the top k features.
   - Fits the model with the selected features and evaluates its performance.
3. **Performance Metrics**:
   - For classifiers, it computes accuracy, precision, recall, and F1 score.
   - For regressors, it computes MAE, MSE, RMSE, and R2 score.
4. **Identifying Best Features**: Tracks the best performance score and the corresponding number of features, feature names, and feature selector.

## plot_accuracy_vs_features(self, save_path=None)

**Functionality**:<br>
This method plots the model performance (F1 score for classification or R2 score for regression) against the number of features used.

**Usage**:

- The method should be run after the select_best_features method.
- It uses the results from select_best_features to create the plot.
- Automatically detects the model type (classification or regression) based on the metrics available in the results and plots the relevant metric.

**Example**:

```python
ml = MachineLearning()
ml.plot_accuracy_vs_features(save_path='/absolute/path/to/save/plot.png')
```

**Detailed Steps**:

1. **Check Results**: Ensures that the select_best_features method has been run by checking for the existence of the results attribute.
2. **Identify Metric**: Determines the appropriate performance metric to plot based on the available columns in the results DataFrame.
3. **Plotting**:
   - Sets up the plot with matplotlib.
   - Plots the performance metric against the number of features used.
   - Adds titles, labels, and grid lines for clarity.
4. **Save Plot**: If a save path is provided, saves the plot to the specified location.

## evaluate_model(self, X, y, test_size, iterations_number, model)

**Functionality**:
This method evaluates the given model over multiple train-test splits and computes average performance metrics, which are different for classifiers and regressors.

**Usage**:

- The method performs multiple iterations of train-test splits.
- It fits the model on the training data and evaluates it on the testing data.
- Computes the average performance metrics over all iterations.

**Example**:

```python
ml = MachineLearning()
from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor(n_neighbors=3) # or any classifier
iteration_metrics, average_metrics, final_model = ml.evaluate_model(X, y, 0.3, 30, model)
print(average_metrics)
```

**Detailed Steps**:

1. **Initialize Metrics**: Sets up dictionaries to store performance metrics for each iteration.
2. **Iteration Loop**:
   - Performs multiple iterations (specified by iterations_number).
   - In each iteration, splits the data, fits the model, and evaluates its performance.
3. **Performance Metrics**:
   - For classifiers, it computes accuracy, precision, recall, and F1 score.
   - For regressors, it computes MAE, MSE, RMSE, and R2 score.
4. **Calculate Averages**: Computes the average of each performance metric over all iterations.
5. **Return Results**: Returns a DataFrame of metrics for each iteration, a dictionary of average metrics, and the model fitted in the last iteration.

## plot_iteration_metrics(self)

**Functionality**: This method plots boxplots for the metrics recorded in each iteration by the evaluate_model method.

**Usage**:

- The method requires that evaluate_model has been run beforehand to generate iteration metrics.
- It creates boxplots for each performance metric across all iterations.

**Example**:

```python
ml = MachineLearning()
from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor(n*neighbors=3)
*, \_, iteration_metrics = ml.evaluate_model(X, y, 0.3, 30, model)
ml.plot_iteration_metrics()
```

**Detailed Steps**:

1. **Check Metrics**: Ensures that the evaluate_model method has been run by checking for the existence of the iteration_metrics attribute.
2. **Plotting**:
   - Sets up the plot with matplotlib.
   - Creates boxplots for each performance metric across all iterations.
   - Adds titles, labels, and grid lines for clarity.

## Notes

1. Ensure that your DataFrame is correctly loaded and contains the necessary columns before using the MachineLearning class.
2. The `select_best_features` and `evaluate_model` methods automatically adjust for the model type (classification or regression) and compute the relevant performance metrics.
3. The plotting methods provide flexible options for visualizing model performance and feature selection results.
