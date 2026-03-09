# Datta/data/data.py
import pandas as pd
from pathlib import Path

def load_data() -> pd.DataFrame:
    base = Path(__file__).resolve().parent
    csv_path = base / "df_master_dummy.csv"
    df = pd.read_csv(csv_path)

    # Normalización de etiqueta (CDMX -> Cdmx)
    if "Ciudad Atria" in df.columns:
        df["Ciudad Atria"] = (
            df["Ciudad Atria"]
            .astype(str)
            .str.strip()
            .replace({"CDMX": "Cdmx"})
        )

    return df