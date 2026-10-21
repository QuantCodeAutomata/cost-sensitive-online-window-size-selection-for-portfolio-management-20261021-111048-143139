# Data Directory

Historical market data is fetched at runtime via the QuantX Dataloader API.
No raw OHLCV files are committed to this repository.

## Required Environment Variables
- `DATALOADER_API_KEY`: from https://quantx.limex.pro/
- `MASSIVE_API_KEY`: from https://massive.com/

## Assets Used
- **DJIA study (exp_2)**: 30 DJIA tickers + BIL, 2021-03-04 to 2024-02-25
- **S&P 500 study (exp_3)**: 481 S&P 500 components + BIL, 2020-01-02 to 2024-08-01
- **Synthetic (exp_1, exp_4)**: Generated in-memory; no data download needed.
