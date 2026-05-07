import pandas as pd
import numpy as np

def load_preprocessed(filepath: str) -> pd.DataFrame:
    """
    Load hasil preprocessing
    """
    df = pd.read_csv(filepath)

    print(f"[load_preprocessed] {len(df)} domba, {len(df.columns)} kolom")

    return df
