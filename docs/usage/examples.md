# Usage Examples

- [Example 1: Dataset Information](#example-1-dataset-information)
- [Example 2: Machine Learning](#example-2-machine-learning)
- [Example 3: Plotting](#example-3-plotting)

## Example 1: Dataset Information

```python
import pandas as pd
from prisma_ml import DatasetInformation

df = pd.read_csv("your_dataset.csv")
info = DatasetInformation(df)
info.dataframe_summary()
info.categorical_summary()
info.numerical_summary()
```

## Example 2: Machine Learning

```python
import pandas as pd
from prisma_ml import MachineLearning
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, f_regression

df = pd.read_csv("your_dataset.csv")
X = df.drop(columns=['target'])
y = df['target']

ml = MachineLearning()
model = LinearRegression()
feature_selector = lambda k: SelectKBest(f_regression, k=k)
best_features, results, best_feature_names, best_selector = ml.select_best_features(X, y, model, feature_selector)
print("Best number of features:", best_features)
print("Best features:", best_feature_names)
```

## Example 3: Plotting

```python
import pandas as pd
from prisma_ml import Plotting

df = pd.read_csv("your_dataset.csv")
plotter = Plotting(df)
plotter.draw_categorical_plots()
plotter.draw_numerical_plots()

scores = {
    'Algorithm1': {'Accuracy': 0.95, 'Precision': 0.90, 'Recall': 0.92},
    'Algorithm2': {'Accuracy': 0.93, 'Precision': 0.88, 'Recall': 0.91}
}
plotter.plot_algorithm_comparison(scores)
```
