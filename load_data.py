import pandas as pd

class DataLoader:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.data = None

    def load_data(self):
        """Loads data from the given file path."""
        self.data = pd.read_csv(self.filepath)
        return self.data

    def get_data_summary(self):
        """Returns a summary of the data."""
        return self.data.describe()
    
    def get_data_info(self):
        """Returns basic info of the data."""
        return self.data.info()
