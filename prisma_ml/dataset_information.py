import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import Markdown, display


class DatasetInformation:
    def __init__(self, df):
        """
        Initialize the DatasetInformation object with a pandas DataFrame.

        Parameters:
        df (pandas.DataFrame): The DataFrame to be summarized.
        """
        self.df = df

    def dataframe_summary(self, col_metadata=None):
        """
        Display a comprehensive summary of the DataFrame.

        This method provides a detailed summary including the shape of the DataFrame,
        information about each column (data types, count of null values), and the
        count of duplicated rows. If provided, descriptions for each column are
        also displayed.

        Parameters:
        col_metadata (dict, optional): A dictionary with column names as keys and
                                       their descriptions as values. If provided,
                                       these descriptions are included in the summary.

        Returns:
        None: This method displays the summary directly and does not return anything.
        """

        def print_md(text):
            display(Markdown(text))

        pd.set_option("display.max_colwidth", None)

        print_md("### Shape:")
        display(self.df.shape)

        print_md("### Columns and Metadata:")
        columns_df = pd.DataFrame(
            {
                "Data Type": [str(t) for t in self.df.dtypes],
                "Null Values": self.df.isna().sum(),
                "Percentage of Nulls": [
                    "{:.1f}".format(val)
                    for val in ((self.df.isna().sum() / self.df.shape[0]) * 100)
                ],
            }
        )
        if col_metadata:
            columns_df["Description"] = self.df.columns.map(col_metadata).fillna(
                "No description available"
            )

        styled_df = columns_df.style.set_properties(
            **{"text-align": "left"}
        ).set_table_styles([dict(selector="th", props=[("text-align", "left")])])
        display(styled_df)

        pd.reset_option("display.max_colwidth")

        print_md("### Duplicated Rows:")
        display(
            pd.DataFrame(
                [self.df.duplicated().sum()],
                columns=["Duplicated Rows Count"],
                index=["Total"],
            )
        )

    def categorical_summary(self):
        """
        Display a summary of categorical columns in the DataFrame.

        This method provides:
        - A summary DataFrame with the number of unique values for each categorical column,
          the most frequent value, and its percentage.
        - For columns with fewer than 20 unique values, it displays:
          - A list of unique values.
          - The value counts and percentage distribution for each unique value.

        Returns:
        None: This method displays the summary directly and does not return anything.
        """

        def print_md(text):
            display(Markdown(text))

        categorical_columns = self.df.select_dtypes(
            include=["object", "category"]
        ).columns

        summary_data = []
        for col in categorical_columns:
            unique_count = self.df[col].nunique()
            top_value = self.df[col].value_counts().idxmax()
            top_percentage = (
                self.df[col].value_counts().max() / self.df[col].count()
            ) * 100
            summary_data.append(
                {
                    "Column": col,
                    "Unique Values Count": unique_count,
                    "Top Value": top_value,
                    "Top Value Percentage": f"{top_percentage:.2f}%",
                }
            )

        summary_df = pd.DataFrame(summary_data)
        print_md("### Categorical Columns Summary:")
        display(
            summary_df.style.set_properties(**{"text-align": "left"}).set_table_styles(
                [dict(selector="th", props=[("text-align", "left")])]
            )
        )

        for col in categorical_columns:
            unique_count = self.df[col].nunique()
            if unique_count < 20:
                print_md(f"### Column: {col}")
                print_md("#### Value Counts and Percentage Distribution:")
                value_counts = self.df[col].value_counts().reset_index()
                value_counts.columns = ["Value", "Count"]
                value_counts["Percentage"] = (
                    value_counts["Count"] / value_counts["Count"].sum()
                ) * 100
                display(value_counts.style.set_properties(**{"text-align": "left"}))
                display(Markdown("---"))

    def numerical_summary(self):
        """
        Display a summary of numerical columns in the DataFrame.

        This method provides:
        - A summary DataFrame with statistical measures (mean, median, mode,
          standard deviation, variance, range, minimum, and maximum) for each
          numerical column.
        - A correlation matrix for the numerical columns.
        - A heatmap visualization of the correlation matrix.

        Returns:
        None: This method displays the summary directly and does not return anything.
        """

        def print_md(text):
            display(Markdown(text))

        numerical_columns = self.df.select_dtypes(include=["number"]).columns

        stats_summary = pd.DataFrame()
        for col in numerical_columns:
            stats = {
                "Mean": self.df[col].mean(),
                "Median": self.df[col].median(),
                "Mode": self.df[col].mode()[0],
                "Std Dev": self.df[col].std(),
                "Variance": self.df[col].var(),
                "Range": self.df[col].max() - self.df[col].min(),
                "Min": self.df[col].min(),
                "Max": self.df[col].max(),
            }
            stats_summary[col] = pd.Series(stats)

        print_md("### Numerical Columns Statistical Summary:")
        display(stats_summary)

        correlation_matrix = self.df[numerical_columns].corr()

        print_md("### Correlation Matrix:")
        display(correlation_matrix)

        print_md("### Correlation Matrix Heatmap:")
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5
        )
        plt.show()
