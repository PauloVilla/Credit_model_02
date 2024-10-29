import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class LoanEDA:
    def __init__(self, data: pd.DataFrame, target_column: str):
        self.data = data
        self.target_column = target_column
        self.categorical_features = self.data.select_dtypes(include=['object']).columns.tolist()
        self.numeric_features = self.data.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        # Exclude the target column from feature lists to avoid plotting it as a feature
        if self.target_column in self.categorical_features:
            self.categorical_features.remove(self.target_column)
        if self.target_column in self.numeric_features:
            self.numeric_features.remove(self.target_column)

    def plot_numeric_histograms(self):
        """Plot histograms with KDE for numeric features."""
        num_features = len(self.numeric_features)
        fig, axes = plt.subplots(num_features, 1, figsize=(8, num_features * 4))
        
        for idx, column in enumerate(self.numeric_features):
            sns.histplot(data=self.data, x=column, bins=30, kde=True, color='blue', ax=axes[idx])
            axes[idx].set_title(f'Distribution of {column}')
        
        plt.tight_layout()
        plt.show()

    def plot_numeric_violinplots(self):
        """Plot violin plots for numeric features by the target label."""
        num_features = len(self.numeric_features)
        fig, axes = plt.subplots(num_features, 1, figsize=(8, num_features * 4))
        
        for idx, column in enumerate(self.numeric_features):
            sns.violinplot(data=self.data, x=self.target_column, y=column, palette='viridis', ax=axes[idx])
            axes[idx].set_title(f'{column} by {self.target_column}')
        
        plt.tight_layout()
        plt.show()

    def plot_categorical_features(self):
        """Plot stacked bar charts for categorical features by the target label."""
        for column in self.categorical_features:
            loan_status_counts = self.data.groupby([column, self.target_column]).size().unstack()
            loan_status_counts.plot(kind='bar', stacked=True, figsize=(10, 6))
            plt.title(f'Stacked Bar Chart of {column} by {self.target_column}')
            plt.xlabel(column)
            plt.ylabel('Count')
            plt.legend(title=self.target_column)
            plt.show()
    
    @staticmethod
    def replace_outliers_with_median(series, perc):
        """
        Replace outlier values in a series (column of a DataFrame) with the median.
        Outliers are defined as those that are above the 90th percentile.

        Parameters:
        ----------
        series (pd.Series): The column of the DataFrame to which the function will be applied.
        percentile (float): The percentile above which values will be replaced.

        Returns:
        -------
        pd.Series: The column with outlier values replaced by the median.
        """
        # Calculates the percentil stablished
        p = series.quantile(perc)
        
        # Calculates the median
        median = series.median()
        
        # Replace the values greater than the selected percentile with the median
        return series.apply(lambda x: median if x > p else x)
    
    def replace_outliers(self, columns: list, perc=0.95):
        """Function that calls the replace outliers method in an iterable 

        Args:
            columns (list): List of the columns with outliers to be replaced
            perc (float, optional): Percentil limit for the replacement. Defaults to 0.95.
        """
        df = self.data
        for i in columns:
            df[i] = self.replace_outliers_with_median(df[i], perc=perc)
        # Update the Dataset
        self.data = df
    
    def handle_missing_values(self, impute_dict: dict = None):
        """This function allows to impute missing values for both numeric and categorical features based on the given dictionary.

        Args:
            impute_dict (dict, optional): Dictionary with the form {col1: strategy, ..., coln: strategy}. Defaults to None.

        Returns:
            data: The dataset with the imputed values
        """
        if impute_dict is None:
            impute_dict = {}
        
        for column in self.numeric_features:
            # Get the imputer type for the column, or default to 'median'
            imputer_type = impute_dict.get(column, 'median')
            if imputer_type == 'mean':
                self.data[column].fillna(self.data[column].mean(), inplace=True)
            elif imputer_type == 'median':
                self.data[column].fillna(self.data[column].median(), inplace=True)
            elif imputer_type == 'mode':
                self.data[column].fillna(self.data[column].mode()[0], inplace=True)
            else:
                self.data[column].fillna(imputer_type, inplace=True)  # If a specific value is given
        
        for column in self.categorical_features:
            # Get the imputer type for the column, or default to 'mode'
            imputer_type = impute_dict.get(column, 'mode')
            if imputer_type == 'mode':
                self.data[column].fillna(self.data[column].mode()[0], inplace=True)
            else:
                self.data[column].fillna(imputer_type, inplace=True)  # If a specific value is given
        
        return self.data
    
    def get_data_with_handled_missing_values(self, impute_dict=None):
        """Return data after handling missing values."""
        return self.handle_missing_values(impute_dict)
