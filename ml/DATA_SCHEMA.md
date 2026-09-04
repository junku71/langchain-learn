# Point-in-time ML panel data

## Production pipeline

The redesigned filter uses a daily combined KOSPI/KOSDAQ market-cap universe:

```text
training universe   daily KOSPI top 100 + KOSDAQ top 100 (200 total)
prediction universe latest KOSPI top 50 + KOSDAQ top 50 (100 total)
selection           top 10 per market (20 total)
target              10-session market excess return > 3%
weights             RF 0.20 / LGBM classifier 0.30 / LGBM ranker 0.50
```

Run each resumable stage in order:

```powershell
uv run python -m ml.collect_data --start 2018-01-01
uv run python -m ml.build_dataset
uv run python -m ml.validate_dataset
uv run python -m ml.train_model
uv run python -m ml.predict_top_stocks --top 10
```

While KRX approval is pending, all other sources can be checked without writing
their caches:

```powershell
uv run python -m ml.check_data_sources --ticker 005930.KS --days 45
```

The command exits with code 1 when an API fails or an existing history CSV has
only a header and no usable rows.

Use `--limit-sessions 1 --skip-flow --skip-fundamental` on `collect_data` for a
quick KRX/Yahoo smoke test. KIS flow collection is only available during its
published service window and resumes from its checkpoint.

Primary Parquet artifacts:

```text
data/ml/raw/universe_history.parquet
data/ml/raw/price_history.parquet
data/ml/processed/feature_panel.parquet
data/ml/reports/data_quality.json
```

The universe table is keyed by `date,ticker` and contains `market_cap`,
`market_cap_rank`, `training_universe`, and `prediction_universe`. Rolling price
features and forward labels are calculated from continuous price histories
before rows are restricted to that day's universe.

Foreign US-market and commodity observations are lagged one Korean session.
Fundamentals are backward-as-of joined from their DART publication date. Flow
features are net purchase value divided by contemporaneous market cap. No
future values are backfilled.

`KRX_API_KEY` is mandatory for the historical universe. The blocked web-screen
JSON is not used as a fallback because silently using today's constituents for
past dates would introduce survivorship bias.

The ensemble trains on each date's KOSPI top 100 and KOSDAQ top 100 universe.
Price history and the universe are generated automatically. Optional historical
data must be point-in-time safe; never copy today's fundamentals into old rows.

## Flow history

Path: `data/ml/flow_history.csv`

Required columns:

```text
date,ticker,foreign_net,institution_net
```

One row represents flows known for that ticker and trading date.

## Fundamental history

Path: `data/ml/fundamental_history.csv`

Required columns:

```text
available_date,ticker,per,pbr,psr,pcr,ev_ebitda,roe
```

`available_date` is the first trading date on which the financial information
was public. Values are joined backward-as-of and are never backfilled into dates
before publication.

## Preparing five years of data

Run the collectors separately (both commands are resumable):

```powershell
uv run python -m ml.collect_flow_history --years 5
uv run python -m ml.collect_fundamental_history --years 5
```

Or prepare flow, fundamentals, price, economy and the merged panel together:

```powershell
uv run python -m ml.prepare_training_data --years 5
```

This also writes `data/ml/feature_panel.csv`. Each row is one ticker and one
trading date, containing OHLCV, every column in `ml.features.FEATURE_COLUMNS`,
and the forward targets used during training. Technical columns such as RSI,
MACD, ADX, ATR and Bollinger position are derived from OHLCV rather than fetched
from a separate API.

Set `KIS_*`, `ECOS_API_KEY`, and `DART_API_KEY` in `.env` first. Use
`--limit 1` for an API smoke test. Price/technical/volume features are derived
from OHLCV, market features from `data/economy/economy_data.csv`, and sector and
cross-sectional features are calculated across the daily universe.

## Sector mapping

Path: `data/ml/universe_top200.csv`

KIS `bstp_kor_isnm` is cached as `kis_sector`. `ml_sector` is initialized from
that raw value and is the field used by sector-return features through the
compatibility `sector` column. Edit only `ml_sector` when a coarser investment
classification is needed; refreshes preserve that override. KIS values are
stored in `data/ml/kis_sector_cache.csv`, so only new or unresolved tickers are
requested again. Foreign-listed `950xxx` stocks without a KIS business type use
`FOREIGN_LISTED` as the ML fallback while retaining `kis_sector=UNKNOWN`.
