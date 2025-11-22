"""
Construye un dataset único a partir de todos los CSV de data/processed.

Uso:
  py -m src.data.build_dataset
"""

from pathlib import Path
import pandas as pd

PROCESSED_DIR = Path("data/processed")
OUT_PATH = Path("data/processed/all_landmarks.csv")

def main():
    csv_files = sorted(PROCESSED_DIR.glob("*.csv"))
    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        df["source"] = f.stem  # para saber de qué video viene cada fila
        dfs.append(df)
        print(f"[OK] Cargado {f.name} ({len(df)} filas)")
    full = pd.concat(dfs, ignore_index=True)
    full.to_csv(OUT_PATH, index=False)
    print(f"\n[OK] Dataset combinado guardado en: {OUT_PATH} | Filas totales: {len(full)}")

if __name__ == "__main__":
    main()
