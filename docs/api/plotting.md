# Detailed Method Descriptions for `Plotting` Class

## Table of Contents

- [draw_categorical_plots(self, n_cols=3)](#draw_categorical_plotsself-n_cols3)
- [draw_numerical_plots(self, n_cols=3)](#draw_numerical_plotsself-n_cols3)
- [plot_algorithm_comparison(self, algorithm_scores, bar_width=0.2)](#plot_algorithm_comparisonself-algorithm_scores-bar_width02)
- [Notes](#notes)

## draw_categorical_plots(self, n_cols=3)

**Functionality**:<br> This method identifies all categorical columns in the DataFrame and generates count plots for each column. The plots are arranged in a grid layout with a specified number of columns per row.

**Usage**:

- The method first checks if there are any categorical columns in the DataFrame. If none are found, it prints a message and returns.
- It calculates the number of rows needed based on the number of categorical columns and the specified number of columns per row.
- It sets up a matplotlib figure and axes, and generates count plots for each categorical column.
- Any unused subplots are hidden, and the layout is adjusted to fit all plots.

**Example**:

```python
df = pd.read_csv("your_dataset.csv")
plotter = Plotting(df)
plotter.draw_categorical_plots(n_cols=3)
```

**Detailed Steps**:

1. **Check Categorical Columns**: Identifies all categorical columns in the DataFrame. If none are found, prints "No categorical columns to plot." and returns.
2. **Calculate Layout**: Determines the number of rows needed based on the number of categorical columns and the specified number of columns per row.
3. **Set Up Figure and Axes**: Sets up a matplotlib figure and axes for the plots.
4. **Generate Plots**:
   - Iterates over each categorical column and generates a count plot using seaborn.
   - Sets the title, xlabel, and ylabel for each plot.
   - Hide Unused Subplots: Any unused subplots are hidden.
   - Adjust Layout: Adjusts the layout to fit all plots neatly using plt.tight_layout().

## draw_numerical_plots(self, n_cols=3)

**Functionality**:<br> This method identifies all numerical columns in the DataFrame and generates histograms for each column. The plots are arranged in a grid layout with a specified number of columns per row.

**Usage**:<br>

- The method first checks if there are any numerical columns in the DataFrame. If none are found, it prints a message and returns.
- It calculates the number of rows needed based on the number of numerical columns and the specified number of columns per row.
- It sets up a matplotlib figure and axes, and generates histograms for each numerical column.
- Any unused subplots are hidden, and the layout is adjusted to fit all plots.

**Example**:

```python
df = pd.read_csv("your_dataset.csv")
plotter = Plotting(df)
plotter.draw_numerical_plots(n_cols=3)
```

**Detailed Steps**:

1. **Check Numerical Columns**: Identifies all numerical columns in the DataFrame. If none are found, prints "No numerical columns to plot." and returns.
2. **Calculate Layout**: Determines the number of rows needed based on the number of numerical columns and the specified number of columns per row.
3. **Set Up Figure and Axes**: Sets up a matplotlib figure and axes for the plots.
4. **Generate Plots**:
   - Iterates over each numerical column and generates a histogram using seaborn.
   - Sets the title, xlabel, and ylabel for each plot.
5. **Hide Unused Subplots**: Any unused subplots are hidden.
6. **Adjust Layout**: Adjusts the layout to fit all plots neatly using plt.tight_layout().

## plot_algorithm_comparison(self, algorithm_scores, bar_width=0.2)

**Functionality**:<br>
This method generates a bar chart comparing the performance of different algorithms based on various scores. Each algorithm is represented by a different color, and the scores are displayed as bars.

**Usage**:<br>

- The method extracts score names and algorithm names from the algorithm_scores dictionary.
- It sets up a matplotlib figure and axes, and creates a bar plot for each algorithm.
- The labels and title are set, and the layout is adjusted to fit all plots.

**Example**:

```python
scores = {
'Algorithm1': {'Accuracy': 0.95, 'Precision': 0.90, 'Recall': 0.92},
'Algorithm2': {'Accuracy': 0.93, 'Precision': 0.88, 'Recall': 0.91}
}
plotter = Plotting(df)
plotter.plot_algorithm_comparison(scores, bar_width=0.2)
```

**Detailed Steps**:

1. **Extract Score and Algorithm Names**: Extracts score names and algorithm names from the algorithm_scores dictionary.
2. **Set Up Figure and Axes**: Sets up a matplotlib figure and axes for the bar chart.
3. **Generate Bar Plots**:

- Iterates over each algorithm and generates a bar plot for its scores.
- Adjusts the positions of the bars to ensure they are spaced correctly.

4. **Set Labels and Title**: Sets the xticks, xticklabels, ylabel, xlabel, and title for the bar chart.
5. **Adjust Layout**: Adjusts the layout to fit all plots neatly using plt.tight_layout().

## Notes

- Ensure that your DataFrame is correctly loaded and contains the necessary columns before using the Plotting class.
- The draw_categorical_plots and draw_numerical_plots methods will automatically adjust the layout to fit all plots, making it easy to visualize large datasets.
- The plot_algorithm_comparison method provides a flexible way to compare algorithm performance, with customizable bar widths.
