import pickle
import pandas as pd

class LoanApprovalPredictor:
    def __init__(self, model_path, encoder_path):
        """
        Initialize the predictor class by loading the model and encoder.

        Parameters:
        ----------
        model_path (str): Path to the saved XGBoost model file.
        encoder_path (str): Path to the saved OneHotEncoder file.
        """
        # Load the model
        with open(model_path, 'rb') as model_file:
            self.model = pickle.load(model_file)

        # Load the encoder
        with open(encoder_path, 'rb') as encoder_file:
            self.encoder = pickle.load(encoder_file)
        
        # Define the categorical features
        self.categorical_features = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']

    def preprocess_input(self, input_data):
        """
        Preprocess the input data using the loaded encoder for categorical features.

        Parameters:
        ----------
        input_data (pd.DataFrame): The input data with categorical features.

        Returns:
        -------
        pd.DataFrame: The preprocessed data ready for model prediction.
        """
        # Encode the categorical features
        input_encoded = pd.DataFrame(self.encoder.transform(input_data[self.categorical_features]).toarray())
        
        # Drop original categorical features and add encoded features
        input_data = input_data.drop(columns=self.categorical_features).reset_index(drop=True)
        input_data = pd.concat([input_data, input_encoded], axis=1)

        return input_data

    def predict_default_probability(self, input_data):
        """
        Predict the probability of default for the given input data.

        Parameters:
        ----------
        input_data (pd.DataFrame): The input data for prediction.
        categorical_features (list): List of categorical feature names in the input data.

        Returns:
        -------
        pd.Series: The predicted probabilities of default.
        """
        # Preprocess the input data
        input_data = self.preprocess_input(input_data)
        
        # Predict the probability of default
        probabilities = self.model.predict_proba(input_data)[:, 1]
        
        return pd.Series(probabilities, name="Predicted Default Probability")
