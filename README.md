<p align="center">
  <a href="https://github.com/Yousinator/PrismaML">
    <img src="https://github.com/ShaanCoding/ReadME-Generator/blob/main/images/logo.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">PrismaML</h3>

  <p align="center">
PrismaML is a Python package designed to streamline and simplify the machine learning model building and data pre-processing process.    <br/>
    <br/>
    <a href="https://github.com/Yousinator/PrismaML/blob/main/docs"><strong>Explore the docs »</strong></a>
    <br/>
    <br/>
    <a href="https://github.com/Yousinator/PrismaML/issues/new?template=bug_report.md">Report Bug</a>
    .
    <a href="https://github.com/Yousinator/PrismaML/issues/new?template=feature_request.md">Request Feature</a>
  </p>
</p>
<p align="center">
  <a href="">
<img src="https://img.shields.io/github/downloads/Yousinator/PrismaML/total"> <img src ="https://img.shields.io/github/contributors/Yousinator/PrismaML?color=dark-green"> <img src ="https://img.shields.io/github/forks/Yousinator/PrismaML?style=social"> <img src ="https://img.shields.io/github/stars/Yousinator/PrismaML?style=social"> <img src ="https://img.shields.io/github/license/Yousinator/PrismaML">
  </a>
</p>

## Table Of Contents

- [About the Project](#about-the-project)
- [Built With](#built-with)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [DatasetInformation](#class-datasetinformation)
    - [dataframe_summary()](#dataframe_summary)
    - [categorical_summary()](#categorical_summary)
    - [numerical_summary()](#numerical_summary)
  - [MachineLearning](#class-machinelearning)
    - [select_best_features()](#select_best_features)
    - [plot_accuracy_vs_features()](#plot_accuracy_vs_features)
    - [evaluate_model()](#evaluate_model)
    - [plot_iteration_metrics()](#plot_iteration_metrics)
  - [Plotting](#class-plotting)
    - [draw_categorical_plots()](#draw_categorical_plots)
    - [draw_numerical_plots()](#draw_numerical_plots)
    - [plot_algorithm_comparison()](#plot_algorithm_comparison)
- [Contributing](#contributing)
- [License](#license)
- [Authors](#authors)
- [Acknowledgements](#acknowledgements)

## About The Project

> [!Warning]
>
> **Package is Unstable**: <br>This is a beta release of PrismaML. As such, you may encounter issues or incomplete functionalities. Some features may not work as expected, and others might not be fully implemented yet. We appreciate your patience and feedback as we continue to improve the package. If you encounter any problems or have suggestions for improvements, please report them through the [Report Bug](https://github.com/Yousinator/PrismaML/issues/new?template=bug_report.md) link.

> [!Important]
>
> **Tests are Yet to Be Added**: <br>This is a beta release of PrismaML. As such, Unit tests have not yet been added as of this release. However, unit tests are a neccessity and will be added in the future.

PrismaML is a comprehensive package designed to streamline and simplify the machine learning model building process. With features for data analysis, feature selection, model evaluation, and visualization, PrismaML empowers data scientists and machine learning practitioners to build and refine their models more efficiently.

### Key Features

- **Data Summarization**: Automatically generate detailed summaries of your datasets, including metadata, statistical measures, and visualizations.
- **Feature Selection**: Select the best features for your models using various feature selection techniques, enhancing model performance and interpretability.
- **Model Evaluation**: Evaluate machine learning models with robust metrics tailored for both classification and regression tasks.
- **Visualization**: Create powerful visualizations to communicate insights effectively, including categorical and numerical data distributions and algorithm performance comparisons.

---

## Built With

<p align="center">
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Written%20in-Python-blue.svg">
  </a>
  <a href="https://pandas.pydata.org/">
    <img src="https://img.shields.io/badge/Data%20Analysis-Pandas-orange.svg">
  </a>
  <a href="https://scikit-learn.org/">
    <img src="https://img.shields.io/badge/Modeling-Scikit--Learn-yellow.svg">
  </a>
  <a href="https://seaborn.pydata.org/">
    <img src="https://img.shields.io/badge/Visualization-Seaborn-green.svg">
  </a>
  <a href="https://matplotlib.org/">
    <img src="https://img.shields.io/badge/Visualization-Matplotlib-red.svg">
  </a>
</p>

## Getting Started

### Prerequisites

Before you begin, ensure you have met the following requirements:

- **Python**: PrismaML requires Python 3.6 or later. You can download the latest version of Python from the [official website](https://www.python.org/downloads/).

- **pip**: Ensure you have the latest version of `pip` installed. You can upgrade `pip` using the following command:

  ```bash
  python -m pip install --upgrade pip
  ```

- **Poetry**: If you prefer using Poetry for package management, you can install it by following the instructions on the [Poetry website](https://python-poetry.org/docs/#installation).

### Installation

You can install PrismaML using either pip or Poetry.

To install PrismaML, use pip:

```bash
pip install prisma_ml
```

Or using Poetry

```bash
poetry add prisma-ml
```

## Usage

### <h2>Class: `Plotting`</h2>

The Plotting class provides functionalities for visualizing categorical and numerical data in a pandas DataFrame, as well as comparing the performance of different algorithms using bar charts.

**Initialization**:

```python
Plotting(df)
```

#### `draw_categorical_plots()`

**Description**:

This method identifies all categorical columns in the DataFrame and generates count plots for each column. The plots are arranged in a grid layout with a specified number of columns per row.

**Parameters**:

- `n_cols (int)`: Number of columns per row in the plot grid. Default is 3.

**Returns**:

- `None`: This method displays the plot directly.

**Example Usage**:

```python
df = pd.read_csv("your_dataset.csv")
plotter = Plotting(df)
plotter.draw_categorical_plots(n_cols=3)
```

---

#### `draw_numerical_plots()`

**Description**:

This method identifies all numerical columns in the DataFrame and generates histograms for each column. The plots are arranged in a grid layout with a specified number of columns per row.

**Parameters**:

- `n_cols (int)`: Number of columns per row in the plot grid. Default is 3.

**Returns**:

- `None`: This method displays the plot directly.

**Example Usage**:

```python
df = pd.read_csv("your_dataset.csv")
plotter = Plotting(df)
plotter.draw_numerical_plots(n_cols=3)
```

---

#### `plot_algorithm_comparison()`

**Description**:

This method generates a bar chart comparing the performance of different algorithms based on various scores. Each algorithm is represented by a different color, and the scores are displayed as bars.

**Parameters**:

- `algorithm_scores (dict)`: A dictionary where keys are algorithm names and values are dictionaries containing score names as keys and scores as values.
- `bar_width (float)`: Width of the bars in the plot. Default is 0.2.

**Returns**:

- `None`: This method displays the plot directly.

**Example Usage**:

```python
scores = {
    'Algorithm1': {'Accuracy': 0.95, 'Precision': 0.90, 'Recall': 0.92},
    'Algorithm2': {'Accuracy': 0.93, 'Precision': 0.88, 'Recall': 0.91}
}
plotter = Plotting(df)
plotter.plot_algorithm_comparison(scores, bar_width=0.2)
```

---

### <h2>Class: `MachineLearning`</h2>

The MachineLearning class provides functionalities for feature selection, model evaluation, and visualization of model performance using scikit-learn models.

**Initialization**:

```python
MachineLearning()
```

#### `select_best_features()`

**Description**:

This method iteratively applies a feature selection strategy to the dataset, fits the specified machine learning model, and evaluates its performance. It tracks performance metrics for each number of features used and identifies the optimal number of features that yield the best performance score. Additionally, it returns the names of the best features along with the fitted feature selector for the best model.

**Parameters**:

- `X (pd.DataFrame)`: The input features DataFrame.
- `y (pd.Series)`: The target variable Series.
- `model (sklearn estimator)`: An instance of a scikit-learn regression or classification model.
- `feature_selector (function)`: A function that returns a feature selector object from scikit-learn (e.g., SelectKBest). The object must have `fit_transform` and `get_support` methods.

**Returns**:

- `int`: The optimal number of features yielding the best performance score.
- `pd.DataFrame`: A DataFrame containing the performance metrics for each number of features.
- `list`: The names of the best features selected by the feature selection method.
- `selector`: The fitted feature selector object for the best model.

**Example Usage**:

```python
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, f_regression

ml = MachineLearning()
model = LinearRegression()
feature_selector = lambda k: SelectKBest(f_regression, k=k)
best_features, results, best_feature_names, best_selector = ml.select_best_features(X, y, model, feature_selector)
print("Best number of features:", best_features)
print("Best features:", best_feature_names)

```

---

#### `plot_accuracy_vs_features()`

**Description**:

This method should be run after the `select_best_features` method. It uses the results from `select_best_features` to create the plot. The method automatically detects the model type (classification or regression) based on the metrics available in the results and plots the relevant metric.

**Parameters**:

- `save_path (str, optional)`: Absolute path to save the plot. If None, the plot is not saved to a file.

**Returns**:

- `None`: This method displays the plot directly.

**Example Usage**:

```python
ml = MachineLearning()
ml.plot_accuracy_vs_features(save_path='/absolute/path/to/save/plot.png')

```

---

#### `evaluate_model()`

**Description**:

This method evaluates the given model over multiple train-test splits and computes average performance metrics, which are different for classifiers and regressors. It returns the metrics recorded for each iteration, the average performance metrics, and the model fitted in the last iteration.

**Parameters**:

- `X (pd.DataFrame)`: The input features DataFrame.
- `y (pd.Series)`: The target variable Series.
- `test_size (float)`: The proportion of the dataset to include in the test split.
- `iterations_number (int)`: The number of iterations to repeat the train-test split and evaluation.
- `model (sklearn estimator)`: An instance of a scikit-learn regression or classification model.

**Returns**:

- `pd.DataFrame`: DataFrame containing the metrics recorded for each iteration.
- `dict`: Average performance metrics over all iterations, rounded to three decimal places.
- `model`: The model fitted in the last iteration.

**Example Usage**:

```python
ml = MachineLearning()
from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor(n_neighbors=3)  # or any classifier
iteration_metrics, average_metrics, final_model = ml.evaluate_model(X, y, 0.3, 30, model)
print(average_metrics)

```

---

#### `plot_iteration_metrics()`

**Description**:

This method plots boxplots for the metrics recorded in each iteration by the `evaluate_model` method. It requires that `evaluate_model` has been run beforehand to generate iteration metrics.

**Returns**:

- `None`: This method displays the plot directly.

**Example Usage**:

```python
ml = MachineLearning()
from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor(n_neighbors=3)
_, _, iteration_metrics = ml.evaluate_model(X, y, 0.3, 30, model)
ml.plot_iteration_metrics()

```

---

### <h2>Class: `DatasetInformation`</h2>

The DatasetInformation class provides functionalities for summarizing and visualizing various aspects of a pandas DataFrame.

**Initialization**:

```python
DatasetInformation(df)
```

**Parameters**:

- `df (pandas.DataFrame)`: The DataFrame to be summarized.

#### `dataframe_summary()`

**Description**:

This method provides a detailed summary of the DataFrame, including the shape, column data types, count of null values, and duplicated rows. If column metadata is provided, it is included in the summary.

**Parameters**:

- `col_metadata (dict, optional)`: A dictionary with column names as keys and their descriptions as values. If provided, these descriptions are included in the summary.

**Returns**:

- `None`: This method displays the summary directly.

**Example Usage**:

```python
df = pd.read_csv("your_dataset.csv")
col_metadata = {
    'column1': 'Description for column1',
    'column2': 'Description for column2',
    # ... other columns
}
info = DatasetInformation(df)
info.dataframe_summary(col_metadata)

```

---

#### `categorical_summary()`

**Description**:

This method summarizes categorical columns in the DataFrame, showing the number of unique values, the most frequent value, and its percentage for each column. For columns with fewer than 20 unique values, it also displays a detailed list of unique values and their distributions.

**Returns**:

- `None`: This method displays the summary directly.

**Example Usage**:

```python
df = pd.read_csv("your_dataset.csv")
info = DatasetInformation(df)
info.categorical_summary()
```

---

#### `numerical_summary()`

**Description**:

This method summarizes numerical columns in the DataFrame by providing statistical measures such as mean, median, mode, standard deviation, variance, range, minimum, and maximum for each column. It also displays a correlation matrix and a heatmap of the correlations.

**Returns**:

- `None`: This method displays the summary directly.

**Example Usage**:

```python
df = pd.read_csv("your_dataset.csv")
info = DatasetInformation(df)
info.numerical_summary()
```

---

## Contributing

We welcome contributions to PrismaML! Whether you want to report a bug, request a feature, or contribute code, your input is valuable to us. Here's how you can get started:

### Reporting Bugs

If you encounter any bugs while using PrismaML, please let us know by creating a new issue. Provide as much detail as possible to help us understand and address the issue. [Report a Bug]()

### Requesting Features

Have an idea for a new feature? We’d love to hear it! Use the link below to request a new feature. [Request a feature]()

### Submitting Pull Requests

We appreciate your contributions! Follow these steps to submit a pull request:

1. **Fork the Repository**: Click the "Fork" button at the top right of this page to create a copy of the repository on your GitHub account.
2. **Clone Your Fork**: Clone your fork to your local machine.

   ```bash
   git clone https://github.com/yourusername/PrismaML.git
   cd PrismaML
   ```

3. **Create a New Branch**: Create a new branch for your changes.

   ```bash
   git checkout -b feature-or-bugfix-description
   ```

4. **Make Your Changes**: Make your changes to the code.
5. **Commit Your Changes**: Commit your changes with a descriptive commit message.
   ```bash
    git add .
    git commit -m "Description of your changes"
   ```
6. **Push to Your Fork**: Push your changes to your fork on GitHub.
   ```bash
   git push origin feature-or-bugfix-description
   ```
7. **Open a Pull Request**: Go to the original repository and click the "New Pull Request" button. Provide a clear description of your changes and why they are needed.

## License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.

## Authors

- **Yousinator** - _AI Engineer_ - [Yousinator](https://github.com/Yousinator/) - _Wrote the codes and README_\

## Acknowledgements

- [Yousinator](https://github.com/Yousinator)
- [ImgShields](https://shields.io/)
