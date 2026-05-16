import pandas as pd

def load_preprocessed(filepath: str) -> pd.DataFrame:
    """
    Load hasil preprocessing
    """
    df = pd.read_csv(filepath)

    return df
