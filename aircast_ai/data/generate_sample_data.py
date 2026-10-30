"""
Regenerate data/sample_air_quality.csv (SYNTHETIC data, not real measurements).

Run from the project root:
    python data/generate_sample_data.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_processing import save_synthetic_csv  # noqa: E402

if __name__ == "__main__":
    out = save_synthetic_csv(ROOT / "data" / "sample_air_quality.csv")
    print(f"Synthetic sample dataset written to: {out}")
