import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math


class Plotting:
    def __init__(self, df):
        """
        Initialize the Plotting object with a pandas DataFrame.

        Parameters:
        df (pandas.DataFrame): The DataFrame for which the plots are to be generated.
        """
        self.df = df

    def draw_categorical_plots(self, n_cols=3):
        """
        Draws combined plots for all categorical columns in the DataFrame.

        This method generates individual count plots for each categorical column
        and arranges them in a grid layout.

        Parameters:
        n_cols (int): Number of columns per row in the plot grid.

        Returns:
        None: This method displays the plot directly and does not return anything.
        """
        # Identify categorical columns
        categorical_columns = self.df.select_dtypes(
            include=["object", "category"]
        ).columns

        if len(categorical_columns) == 0:
            print("No categorical columns to plot.")
            return

        # Determine the number of rows needed for subplots
        n_rows = math.ceil(len(categorical_columns) / n_cols)

        # Set up the matplotlib figure
        fig, axes = plt.subplots(
            nrows=n_rows, ncols=n_cols, figsize=(5 * n_cols, 5 * n_rows)
        )
        axes = axes.flatten() if n_rows > 1 else [axes]

        # Generate plots for each categorical column
        for i, col in enumerate(categorical_columns):
            sns.countplot(x=col, data=self.df, ax=axes[i], palette="viridis")
            axes[i].set_title(f"Count Plot for {col}")
            axes[i].set_xlabel(col)
            axes[i].set_ylabel("Count")

        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        plt.show()

    def draw_numerical_plots(self, n_cols=3):
        """
        Draws combined plots for all numerical columns in the DataFrame.

        This method generates individual histograms for each numerical column
        and arranges them in a grid layout.

        Parameters:
        n_cols (int): Number of columns per row in the plot grid.

        Returns:
        None: This method displays the plot directly and does not return anything.
        """
        # Identify numerical columns
        numerical_columns = self.df.select_dtypes(include=["number"]).columns

        if len(numerical_columns) == 0:
            print("No numerical columns to plot.")
            return

        # Determine the number of rows needed for subplots
        n_rows = math.ceil(len(numerical_columns) / n_cols)

        # Set up the matplotlib figure
        fig, axes = plt.subplots(
            nrows=n_rows, ncols=n_cols, figsize=(5 * n_cols, 5 * n_rows)
        )
        axes = axes.flatten() if n_rows > 1 else [axes]

        # Generate histograms for each numerical column
        for i, col in enumerate(numerical_columns):
            sns.histplot(self.df[col], ax=axes[i], kde=True, color="skyblue")
            axes[i].set_title(f"Histogram for {col}")
            axes[i].set_xlabel(col)
            axes[i].set_ylabel("Frequency")

        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        plt.show()

    def plot_algorithm_comparison(self, algorithm_scores, bar_width=0.2):
        """
        Plots a bar chart comparing different algorithms based on various scores.
        Each algorithm is represented by a different color.

        Parameters:
        - algorithm_scores (dict): A dictionary where keys are algorithm names and values are dictionaries
          containing score names as keys and scores as values.
        - bar_width (float): Width of the bars in the plot.

        Example Usage:
        >>> scores = {
        >>>     'Algorithm1': {'Accuracy': 0.95, 'Precision': 0.90, 'Recall': 0.92},
        >>>     'Algorithm2': {'Accuracy': 0.93, 'Precision': 0.88, 'Recall': 0.91}
        >>> }
        >>> plotter = Plotting(df)
        >>> plotter.plot_algorithm_comparison(scores)
        """
        # Extract score names
        score_labels = list(next(iter(algorithm_scores.values())).keys())
        algorithms = list(algorithm_scores.keys())
        n_algorithms = len(algorithms)

        # Set up the figure and axes
        fig, ax = plt.subplots(figsize=(10, 6))

        # Create a bar plot for each algorithm
        for i, algorithm in enumerate(algorithms):
            scores = [algorithm_scores[algorithm][score] for score in score_labels]
            positions = [x + i * bar_width for x in range(len(score_labels))]
            ax.bar(positions, scores, width=bar_width, label=algorithm)

        # Set the labels and title
        mid_positions = [
            p + (bar_width * (n_algorithms - 1) / 2) for p in range(len(score_labels))
        ]
        ax.set_xticks(mid_positions)
        ax.set_xticklabels(score_labels)
        ax.set_ylabel("Scores")
        ax.set_xlabel("Metrics")
        ax.set_title("Algorithm Performance Comparison")
        ax.legend()

        plt.tight_layout()
        plt.show()


# Example usage
# df = pd.read_csv("your_dataset.csv")
# plotter = Plotting(df)
# plotter.draw_categorical_plots()
# plotter.draw_numerical_plots()
# algorithm_scores = {
#     'Algorithm1': {'Accuracy': 0.95, 'Precision': 0.90, 'Recall': 0.92},
#     'Algorithm2': {'Accuracy': 0.93, 'Precision': 0.88, 'Recall': 0.91}
# }
# plotter.plot_algorithm_comparison(algorithm_scores)
