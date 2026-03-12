import pandas as pd


def load_data(parquet_path, art_id=None):
    """
    Load dataset and optionally filter a specific SKU.
    """

    df = pd.read_parquet(parquet_path)

    if art_id is not None:
        df = df[df["art_id"] == art_id].copy()

    df = df.sort_values("date").reset_index(drop=True)

    return df
