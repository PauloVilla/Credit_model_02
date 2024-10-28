import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import plotly.graph_objects as go

class LoanApprovalModel:
    def __init__(self, data, target_column):
        """Initializer method of the class

        Args:
            data (pd.DataFrame): Dataset for the model
            target_column (str): Column that is the objective.
        """

        # Probability of Default
        self.prd = None
        # Exposure at Default
        self.ead = None

        # Other params for the class
        self.data = data
        self.target_column = target_column
        self.categorical_features = self.data.select_dtypes(include=['object']).columns.tolist()
        self.numeric_features = self.data.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        # Exclude the target column from feature lists to avoid plotting and modelling it as a feature
        if self.target_column in self.categorical_features:
            self.categorical_features.remove(self.target_column)
        if self.target_column in self.numeric_features:
            self.numeric_features.remove(self.target_column)
    
    def preprocess_data(self, X):
        """Preprocessing method for categorical features, tranform to one hot encoding.

        Args:
            X (pd.DataFrame): Dataset with the categorical columns.

        Returns:
            X: Transformed Dataset
        """
        self.onehot_encoder = OneHotEncoder(drop="first")
        categorical_features = self.categorical_features
        # Fit and transform the categorical features
        X_encoded = pd.DataFrame(self.onehot_encoder.fit_transform(X[categorical_features]).toarray())
        # Replace categorical features in X with encoded values
        X = X.drop(categorical_features, axis=1)
        X = pd.concat([X.reset_index(drop=True), X_encoded.reset_index(drop=True)], axis=1)
        
        return X
    
    def benchmark_logistic_regression(self):
        """Function to benchmark the dataset with a basic Logistic Regression model

        Returns:
            y_test: True labels of the test dataset.
            y_prob: Probabilities predicted by the model.
        """
        # Filter the data only for numeric features
        X = self.data[self.numeric_features]
        X.fillna(0, inplace=True)  # Fill missing values with 0
        # Select Loan status as the objective feature
        y = self.data[self.target_column]
        # Divide the data into train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        # Train the model
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        # Predict probabilities for the test set
        y_prob = model.predict_proba(X_test)[:, 1]
        return y_test, y_prob

    def plot_roc_auc(self, y_test, y_prob):
        """Plot the ROC-AUC Curve.

        Args:
            y_test (np.array): True labels of the test dataset.
            y_prob (np.array): Probabilities predicted by the model.
        """
        auc_score = roc_auc_score(y_test, y_prob)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        # Create the Plotly figure for the ROC curve
        fig = go.Figure()

        # Add the ROC curve
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"ROC curve (AUC = {auc_score:.4f})"))
        
        # Reference line for random guessing
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name="Random guess"))

        # Configure the layout of the graph
        fig.update_layout(
            title=f"ROC Curve (AUC = {auc_score:.4f})",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            showlegend=True,
            width=800,
            height=600,
        )
        
        # Show the interactive plot
        fig.show()

    def prepare_data(self):
        """Separate the data into X and y, train and test."""
        # Prepare your features and target variable
        X = self.data.drop(columns=self.target_column)
        X = self.preprocess_data(X)
        y = self.data[self.target_column]

        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def calculate_scale_pos_weight(self, y):
        """This function computes the weight for the positive class based on the 
            ratio of negative to positive samples in the training data.

        Args:
            y (pd.Series): Binary labels of the training dataset (1 for positive class, 0 for negative class).

        Returns:
            The scale_pos_weight value.
        """
        num_positive = sum(y)
        num_negative = len(y) - num_positive
        return num_negative / num_positive

    def train_xgboost_model(self):
        """Train the XGBoost Model."""
        # Split data and transform
        self.prepare_data()
        scale_pos_weight = self.calculate_scale_pos_weight(self.y_train)
        self.model = XGBClassifier(eval_metric='auc', scale_pos_weight=scale_pos_weight)
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        """Predict and evaluate the XGBoost Model"""
        predictions = self.model.predict(self.X_test)
        y_prob = self.model.predict_proba(self.X_test)[:, 1]

        self.plot_roc_auc(y_prob=y_prob, y_test=self.y_test)
        print("-"*120)
        print(classification_report(self.y_test, predictions))
    
    def calculate_pd(self):
        """Calculate Probability of Default based on model predictions."""
        self.prd = self.model.predict_proba(self.X_test)[:, 1]  # Probability of default
        return self.prd

    def calculate_ead(self):
        """Set EAD as the loan amount for approved applications."""
        self.ead = self.X_test['loan_amnt']
        return self.ead