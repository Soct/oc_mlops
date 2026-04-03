"""
Sépare holdout_sample.parquet en deux fichiers :
  - drift_reference.parquet  : données de référence pour Evidently (50 %)
  - holdout_test.parquet     : données de test pour le dashboard Streamlit (50 %)

Usage :
    python -m app.scripts.split_holdout          # depuis la racine du projet
    python split_holdout.py                       # depuis app/scripts/
"""

from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def split_holdout(
    source: Path | None = None,
    dest_dir: Path | None = None,
    reference_frac: float = 0.5,
    random_state: int = 42,
) -> tuple[Path, Path]:
    """Sépare le holdout en référence drift et test Streamlit."""
    source = source or (DATA_DIR / "holdout_sample.parquet")
    dest_dir = dest_dir or DATA_DIR

    df = pd.read_parquet(source)
    ref = df.sample(frac=reference_frac, random_state=random_state)
    test = df.drop(ref.index)

    ref_path = dest_dir / "drift_reference.parquet"
    test_path = dest_dir / "holdout_test.parquet"

    ref.to_parquet(ref_path, index=False)
    test.to_parquet(test_path, index=False)

    print(f"Reference : {len(ref)} lignes -> {ref_path}")
    print(f"Test      : {len(test)} lignes -> {test_path}")
    return ref_path, test_path


if __name__ == "__main__":
    split_holdout()
