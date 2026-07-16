# Market Sentiment & Microstructure: A Cryptocurrency Trading EDA

> How does macroeconomic fear alter the way traders execute orders, and who profits from the panic?

## Overview

This project investigates the relationship between **macroeconomic market sentiment** (measured by the Bitcoin Fear & Greed Index) and **high-frequency trader behavior** on a cryptocurrency exchange. Rather than predicting price direction, the analysis focuses on _how_ psychological market extremes systematically alter order execution quality, risk tolerance, and structural profitability across different classes of market participants.

The work is structured as a modular Python codebase with a narrative-driven Jupyter notebook suitable for presentation as an analytical report.

## Key Findings

| # | Finding | Insight |
| - | ------- | ------- |
| 1 | **Quantifiable Capitulation** | Retail traders spike their use of aggressive Taker orders during "Extreme Fear," willingly paying premium fees to exit positions immediately. |
| 2 | **Institutional Liquidity Absorption** | The top 5% of accounts by volume ("Whales") remain net-positive during crashes, absorbing panicked retail liquidity at discounted prices. |
| 3 | **The Over-Trading Trap** | K-Means clustering reveals that trade _frequency_ — not size — is the strongest predictor of negative Net PnL, independent of market sentiment. |

## Project Structure

```md
market-sentiment-EDA/
├── data/
│   ├── historical_data.csv        # ~211K individual trade executions
│   └── fear_greed_index.csv       # ~2.6K daily Fear & Greed Index readings
├── notebooks/
│   └── eda.ipynb                  # Jupyter notebook
├── src/
│   └── scripts/
│       ├── __init__.py            # Package initializer
│       ├── paths.py               # Centralized path constants
│       ├── data_loader.py         # Data ingestion and timestamp-aligned merging
│       ├── feature_builder.py     # Feature engineering (Net PnL, Execution Type)
│       └── visualizer.py          # All visualization and clustering logic
├── pyproject.toml                 # Project metadata and dependencies
├── requirements.in                # Top-level dependency pins
├── requirements.txt               # Fully resolved dependency lock file
└── README.md
```

## Data Sources

### Trade Ledger — `historical_data.csv`

Individual trade executions from a cryptocurrency exchange, containing:

| Column | Description |
| ------ | ----------- |
| `Account` | Anonymized wallet address |
| `Coin` | Traded asset identifier |
| `Execution Price` | Fill price in USD |
| `Size Tokens` | Position size in native tokens |
| `Size USD` | Position size in USD |
| `Side` | `BUY` or `SELL` |
| `Timestamp` | Execution timestamp (UNIX milliseconds) |
| `Start Position` | Open position size before this trade |
| `Direction` | Trade direction (`Buy` / `Sell`) |
| `Closed PnL` | Realized profit/loss before fees |
| `Crossed` | `TRUE` if the order crossed the spread (Taker); `FALSE` for Maker |
| `Fee` | Exchange fee paid on this execution |
| `Trade ID` | Unique trade identifier |

### Sentiment Index — `fear_greed_index.csv`

Daily Bitcoin Fear & Greed Index from [alternative.me](https://alternative.me/crypto/fear-and-greed-index/):

| Column | Description |
| ------ | ----------- |
| `timestamp` | Date (UNIX seconds) |
| `value` | Index value (0 = Extreme Fear, 100 = Extreme Greed) |
| `classification` | Human-readable label: `Extreme Fear`, `Fear`, `Neutral`, `Greed`, `Extreme Greed` |

## Setup & Installation

### Prerequisites

- Python ≥ 3.10
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/Pixie2468/market-sentiment-EDA.git
cd market-sentiment-EDA

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -e .

# 4. Launch Jupyter
jupyter notebook notebooks/eda.ipynb
```

> **Note:** The `pyproject.toml` includes all necessary dependencies (pandas, numpy, matplotlib, seaborn, scikit-learn, jupyter, etc.). Running `pip install -e .` installs everything in one step.

## Methodology

### Data Integration

1. **Timestamp Standardization** — Sub-second UNIX execution timestamps and daily sentiment timestamps are both converted to UTC, then joined on a shared date key.
2. **Look-Ahead Bias Prevention** — A left-join with forward-fill ensures no future sentiment leaks into historical trade evaluation.
3. **Friction-Adjusted PnL** — `Net_PnL = Closed PnL − Fee` captures the true economic result of each trade after exchange costs.

### Analyses

| Analysis | Method | Key Variable |
| -------- | ------ | ----------- |
| **Execution Urgency** | Taker % by sentiment regime | `Crossed` (spread-crossing boolean) |
| **Volume Segmentation** | Percentile-based tiering (75/95/100) | `Size USD` per account |
| **Behavioral Clustering** | K-Means (k=3) on standardized features | Trade frequency, win rate, avg size |

## Module Reference

| Module | Purpose |
| ------ | ------- |
| `data_loader.py` | Loads CSVs, converts timestamps to UTC, merges trades with sentiment via date-aligned left-join, forward-fills gaps |
| `feature_builder.py` | Engineers `Net_PnL` (closed PnL minus fees) and `Execution_Type` (Taker/Maker label from `Crossed` boolean) |
| `visualizer.py` | Contains three analysis functions: panic metric bar chart, whale vs. retail PnL segmentation, and K-Means behavioral clustering scatter plot |
| `paths.py` | Resolves project root, data, and source directories as `pathlib.Path` constants |

## Future Work

- **Predictive Order Flow** — Train XGBoost on Taker/Maker ratio time series to forecast liquidity crunches.
- **Holding Time Analytics** — Engineer position durations to compute risk-adjusted Sharpe Ratios per trader cluster.
- **Interactive Dashboard** — Deploy a Streamlit app with real-time sentiment tracking via WebSocket feeds.
- **Causal Inference** — Apply difference-in-differences around sentiment threshold crossings to distinguish correlation from causation.

## License

This project was developed as an analytical assignment. All data is used for educational and research purposes only.
