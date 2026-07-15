"""
Centralized path constants for the project. Resolves directories relative
to this file's location so that imports work regardless of the caller's
working directory.

Usage::

    from src.scripts.paths import DATA_DIR, SRC_DIR
    df = pd.read_csv(DATA_DIR / "historical_data.csv")
"""

from pathlib import Path

# Project root is two levels up from this file: src/scripts/paths.py → project root
ROOT_DIR = Path(__file__).parents[2]

# Directory containing raw CSV data files
DATA_DIR = ROOT_DIR / "data"

# Directory containing the source package
SRC_DIR = ROOT_DIR / "src"
