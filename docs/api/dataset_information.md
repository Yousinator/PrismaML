# Detailed Method Descriptions for `DatasetInformation` Class

## Table of Contents

- [dataframe_summary(self, col_metadata=None)](#dataframe_summaryself-col_metadatanone)
- [categorical_summary(self)](#categorical_summaryself)
- [numerical_summary(self)](#numerical_summaryself)
- [Notes](#notes)

## dataframe_summary(self, col_metadata=None)

**Functionality**:<br>
This method provides a comprehensive summary of the DataFrame, including its shape, column metadata, and duplicated rows.

**Usage**:

- The method first sets the display option for maximum column width to ensure long descriptions are fully visible.
- It prints the shape of the DataFrame.
- It generates a summary DataFrame with data types, count of null values, and percentage of null values for each column.
- If column metadata is provided, it includes descriptions for each column.
- Displays the summary DataFrame with pandas styling.
- Resets the display option for maximum column width.
- Prints the count of duplicated rows.

**Example**:

```python
df = pd.read_csv("your_dataset.csv")
col_metadata = {
'column1': 'Description for column1',
'column2': 'Description for column2', # ... other columns
}
info = DatasetInformation(df)
info.dataframe_summary(col_metadata)
```

**Detailed Steps**:

1. **Display Options**: Sets the display option for maximum column width to ensure long descriptions are visible.
2. **Print Shape**: Prints the shape of the DataFrame.
3. **Generate Summary DataFrame**:
   - Creates a DataFrame with column data types, count of null values, and percentage of null values.
   - If column metadata is provided, includes descriptions for each column.
4. **Display Summary**: Uses pandas styling to display the summary DataFrame.
5. **Reset Display Options**: Resets the display option for maximum column width to default.
6. **Print Duplicates**: Prints the count of duplicated rows.

## categorical_summary(self)

**Functionality**:<br>
This method displays a summary of categorical columns in the DataFrame, including the number of unique values, the most frequent value, and its percentage.

**Usage**:

- The method identifies all categorical columns in the DataFrame.
- It generates a summary DataFrame with the number of unique values, the most frequent value, and its percentage for each categorical column.
- For columns with fewer than 20 unique values, it displays detailed value counts and their percentage distribution.

**Example**:

```python
df = pd.read_csv("your_dataset.csv")
info = DatasetInformation(df)
info.categorical_summary()
```

**Detailed Steps**:

1. **Identify Categorical Columns**: Identifies all categorical columns in the DataFrame.
2. **Generate Summary Data**:

- Creates a summary DataFrame with the number of unique values, the most frequent value, and its percentage for each categorical column.
- Displays the summary DataFrame using pandas styling.

3. **Detailed Value Counts**: For columns with fewer than 20 unique values, displays detailed value counts and their percentage distribution.

## numerical_summary(self)

**Functionality**:<br>
This method displays a summary of numerical columns in the DataFrame, including statistical measures and a correlation matrix with a heatmap.

**Usage**:

- The method identifies all numerical columns in the DataFrame.
- It generates a summary DataFrame with statistical measures such as mean, median, mode, standard deviation, variance, range, minimum, and - maximum for each numerical column.
- It calculates the correlation matrix for numerical columns and displays it.
- It creates and displays a heatmap of the correlation matrix.

**Example**:

```python
df = pd.read_csv("your_dataset.csv")
info = DatasetInformation(df)
info.numerical_summary()
```

**Detailed Steps**:

1. **Identify Numerical Columns**: Identifies all numerical columns in the DataFrame.
2. **Generate Statistical Summary**:

   - Creates a summary DataFrame with statistical measures (mean, median, mode, standard deviation, variance, range, minimum, and maximum) for each numerical column.
   - Displays the statistical summary DataFrame.

3. **Correlation Matrix**:

   - Calculates the correlation matrix for numerical columns.
   - Displays the correlation matrix.

4. **Heatmap**: Creates and displays a heatmap of the correlation matrix using seaborn.

## Notes

- Ensure that your DataFrame is correctly loaded and contains the necessary columns before using the `DatasetInformation` class.
- The `dataframe_summary` method will automatically adjust to display column metadata if provided.
- The `categorical_summary` and `numerical_summary` methods provide detailed insights into the respective column types, making it easy to understand the data distribution and relationships.
