"""Project-wide paths and settings."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("EARNML_DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
UNIVERSE_FILE = ROOT / "config" / "universe.txt"

# The SEC asks automated clients to identify themselves with a contact address:
# https://www.sec.gov/os/accessing-edgar-data . Set SEC_USER_AGENT to your own.
SEC_USER_AGENT = os.environ.get(
    "SEC_USER_AGENT", "earnml-research-project research@example.org"
)

# Holding-company reorganizations give some firms a new CIK with little XBRL
# history. Predecessor registrants (names verified on EDGAR) are merged in;
# the current CIK wins where both report the same fiscal quarter.
PREDECESSOR_CIKS: dict[str, list[int]] = {
    "XOM": [34088],      # EXXON MOBIL CORP -> ExxonMobil Holdings Corp
    "BLK": [1364742],    # BlackRock Finance, Inc. (old BlackRock, Inc.)
    "DIS": [1001039],    # TWDC Enterprises 18 Corp. (old Walt Disney Co)
    "GOOGL": [1288776],  # GOOGLE INC. -> Alphabet Inc.
    "MDT": [64670],      # MEDTRONIC INC -> Medtronic plc
}

PRICE_START = "2009-01-01"
MARKET_TICKER = "SPY"

# Expanding-window design: first out-of-sample year and last year evaluated.
FIRST_TEST_YEAR = 2015
RANDOM_STATE = 42


def load_universe(path: Path = UNIVERSE_FILE) -> list[str]:
    lines = path.read_text().splitlines()
    return [ln.strip().upper() for ln in lines if ln.strip() and not ln.startswith("#")]
