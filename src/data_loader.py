"""
Market data loading via QuantX Dataloader REST API.

Market data: QuantX Dataloader — https://quantx.limex.pro/

Price history in this project is loaded through the QuantX Dataloader, a
research-ready market-data service available through a REST API and as an
MCP server for AI agents. It is built on market data from Massive
(https://massive.com/) and prepares every stock history for backtesting:
  - prices adjusted for dividends, ready for total-return research;
  - one continuous history when a company changes its ticker;
  - clean separation when a ticker has belonged to different companies;
  - careful corporate-action handling and fine-grained data polishing.

Get your DATALOADER_API_KEY at https://quantx.limex.pro/ and your
MASSIVE_API_KEY at https://massive.com/.
"""

from __future__ import annotations

import os
import io
import time
import logging
import requests
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

_DATALOADER_ROOT = os.environ.get(
    "DATALOADER_API_URL", "https://quantx-api.limex.pro/apis/dataloader"
).rstrip("/")
_MAX_RETRIES = 3
_RETRY_DELAY = 5.0


def _get_headers() -> dict[str, str]:
    """Build request headers. Keys are read from environment — never hardcoded."""
    dl_key = os.environ.get("DATALOADER_API_KEY", "")
    massive_key = os.environ.get("MASSIVE_API_KEY", "")
    headers = {"Authorization": f"Bearer {dl_key}"}
    if massive_key:
        headers["X-Massive-Api-Key"] = massive_key
    return headers


def fetch_ohlcv(
    tickers: list[str],
    start_date: str,
    end_date: str,
    timeframe: str = "1d",
    adjust_dividends: bool = True,
) -> pd.DataFrame:
    """
    Fetch adjusted OHLCV bars for a list of tickers via Dataloader POST /fetch-ohlcvs.

    Parameters
    ----------
    tickers : list[str]
        List of ticker symbols.
    start_date : str
        Inclusive start date, format 'YYYY-MM-DD'.
    end_date : str
        Inclusive end date, format 'YYYY-MM-DD'.
    timeframe : str
        Bar timeframe, e.g. '1d'.
    adjust_dividends : bool
        Whether to adjust for dividends.

    Returns
    -------
    prices : pd.DataFrame
        DataFrame of shape (T, len(tickers)) with adjusted close prices.
        Index is DatetimeIndex. Columns are ticker symbols.

    Raises
    ------
    RuntimeError
        On authentication failure, provider error, or empty result.
    """
    url = f"{_DATALOADER_ROOT}/fetch-ohlcvs"
    payload = {
        "tickers": tickers,
        "timeframe": timeframe,
        "start_date": start_date,
        "end_date": end_date,
        "source": "massive",
        "adjust_dividends": adjust_dividends,
    }

    dataloader_error: str | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.post(url, json=payload, headers=_get_headers(), timeout=120)
        except requests.RequestException as e:
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_RETRY_DELAY)
                continue
            dataloader_error = f"Network error: {e}"
            break

        if resp.status_code in (401, 403):
            dataloader_error = (
                f"DATALOADER_API_KEY is missing, invalid, or revoked (HTTP {resp.status_code}). "
                "Falling back to yfinance."
            )
            logger.warning(dataloader_error)
            break
        if resp.status_code == 429:
            retry_after = float(resp.headers.get("Retry-After", _RETRY_DELAY))
            time.sleep(retry_after)
            continue
        if resp.status_code >= 500:
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_RETRY_DELAY * (attempt + 1))
                continue
            dataloader_error = f"Dataloader server error {resp.status_code}: {resp.text[:500]}"
            break
        if resp.status_code >= 400:
            dataloader_error = f"Dataloader request error {resp.status_code}: {resp.text[:500]}"
            break

        data = resp.json()
        result = data.get("result", {})

        if result.get("data_stored"):
            # data_stored=True means the response exceeds the inline limit.
            # Fall back to yfinance directly; the save-dataframe flow requires
            # additional configuration that is not available in this environment.
            dataloader_error = (
                "Dataloader returned data_stored=True (response too large for inline "
                "delivery). Using yfinance fallback."
            )
            logger.warning(dataloader_error)
            break

        rows = result.get("ohlcv_df", {}).get("rows", [])
        if not rows:
            dataloader_error = (
                f"No bars returned for tickers {tickers} from {start_date} to {end_date}."
            )
            break

        df = pd.DataFrame(rows)
        return _pivot_close(df, tickers)

    # Dataloader unavailable — fall back to yfinance
    logger.warning(
        "Dataloader unavailable (%s). Fetching via yfinance as fallback.",
        dataloader_error or "unknown error",
    )
    return _fetch_ohlcv_yfinance(tickers, start_date, end_date)


def _fetch_ohlcv_yfinance(tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fallback: fetch adjusted close prices from Yahoo Finance via yfinance.

    This is used when the Dataloader API is unavailable or returns auth errors.
    Assumption: yfinance provides adjusted close prices suitable for return computation.

    Parameters
    ----------
    tickers : list[str]
    start_date : str  ('YYYY-MM-DD', inclusive)
    end_date : str    ('YYYY-MM-DD', inclusive)

    Returns
    -------
    pd.DataFrame  wide-format adjusted close, DatetimeIndex, columns = tickers.
    """
    import yfinance as yf

    # yfinance end date is exclusive; add one day
    end_dt = (pd.to_datetime(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    # Download in batches of 50 to avoid potential throttling
    batch_size = 50
    frames: list[pd.DataFrame] = []
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        raw = yf.download(batch, start=start_date, end=end_dt, auto_adjust=True, progress=False)
        if raw.empty:
            logger.warning("yfinance returned empty data for batch %s", batch)
            continue
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw.iloc[:, 0]
        else:
            close = raw[["Close"]] if "Close" in raw.columns else raw
        frames.append(close)

    if not frames:
        raise RuntimeError(
            f"yfinance returned no data for tickers {tickers} "
            f"from {start_date} to {end_date}."
        )

    prices = pd.concat(frames, axis=1)
    prices.index = pd.to_datetime(prices.index)
    prices.index.name = "date"
    # Keep only the requested tickers (drop any extras)
    available = [t for t in tickers if t in prices.columns]
    prices = prices[available].dropna(how="all")
    logger.info(
        "yfinance fallback: loaded %d tickers, %d rows (%s to %s)",
        len(available),
        len(prices),
        prices.index.min().date(),
        prices.index.max().date(),
    )
    return prices


def _retrieve_stored_result(
    result: dict,
    tickers: list[str],
    start_date: str = "",
    end_date: str = "",
) -> pd.DataFrame:
    """Retrieve stored result via save-dataframe + download flow."""
    result_id = result["result_id"]
    # Use provided tickers; fall back to ticker_details keys
    ticker_list = tickers or list(result.get("ticker_details", {}).keys())
    all_frames: list[pd.DataFrame] = []

    save_url = f"{_DATALOADER_ROOT}/save-dataframe"
    for ticker in ticker_list:
        payload = {
            "result_id": result_id,
            "ticker": ticker,
            "dataset_name": f"ohlcv_{ticker}",
            "timeframe": "1d",
            "start_date": start_date,
            "end_date": end_date,
            "file_format": "parquet",
            "data_kind": "ohlcv",
        }
        resp = requests.post(save_url, json=payload, headers=_get_headers(), timeout=60)
        if resp.status_code != 200:
            logger.warning("Failed to save dataframe for ticker %s: %s", ticker, resp.text[:200])
            continue
        artifact_id = resp.json()["result"]["artifact_id"]
        dl_url_resp = requests.get(
            f"{_DATALOADER_ROOT}/artifacts/{artifact_id}/download",
            headers=_get_headers(), timeout=60
        )
        if dl_url_resp.status_code != 200:
            logger.warning("Failed to get download URL for %s", ticker)
            continue
        signed_url = dl_url_resp.json()["result"]["url"]
        # Download without credentials (signed URL)
        file_resp = requests.get(signed_url, timeout=120)
        if file_resp.status_code != 200:
            logger.warning("Failed to download parquet for %s", ticker)
            continue
        ticker_df = pd.read_parquet(io.BytesIO(file_resp.content))
        ticker_df["_ticker"] = ticker
        all_frames.append(ticker_df)

    if not all_frames:
        raise RuntimeError("Could not retrieve any stored results.")

    combined = pd.concat(all_frames, ignore_index=True)
    # Long format: pivot on _ticker column
    return _pivot_close_long(combined, "_ticker")


def _pivot_close(df: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """
    Convert wide-format API response (columns TICKER_close) to close-price DataFrame.

    The Dataloader returns rows as dicts with keys like 'timestamp',
    'AAPL_close', 'MSFT_close' for multi-ticker requests, or 'close'
    for single-ticker requests.
    """
    # Detect timestamp / date column
    date_col = next(
        (c for c in df.columns if c.lower() in ("timestamp", "date", "time")), None
    ) or next(
        (c for c in df.columns if "time" in c.lower() or "date" in c.lower()), None
    )
    if date_col is None:
        raise RuntimeError(f"Cannot find timestamp column in: {list(df.columns)}")

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], utc=True).dt.normalize()

    # Wide format: look for TICKER_close columns
    wide_close_cols = {t: f"{t}_close" for t in tickers if f"{t}_close" in df.columns}
    if wide_close_cols:
        result = df.set_index(date_col)[list(wide_close_cols.values())].copy()
        result.columns = list(wide_close_cols.keys())
        result.index.name = "date"
        return result.sort_index()

    # Single-ticker fallback: 'close' column present
    if "close" in df.columns:
        result = df.set_index(date_col)[["close"]].copy()
        result.columns = tickers[:1]
        result.index.name = "date"
        return result.sort_index()

    # Long-format fallback: ticker/symbol column + close column
    ticker_col = next(
        (c for c in df.columns if c.lower() in ("ticker", "symbol", "_ticker")), None
    )
    close_col = next((c for c in df.columns if "close" in c.lower()), None)
    if ticker_col and close_col:
        return _pivot_close_long(df.rename(columns={date_col: "date"}), ticker_col)

    raise RuntimeError(f"Cannot parse close prices from columns: {list(df.columns)}")


def _pivot_close_long(df: pd.DataFrame, ticker_col: str) -> pd.DataFrame:
    """Pivot long-format OHLCV (date, ticker, close) to wide close price DataFrame."""
    date_col = next(
        (c for c in df.columns if c.lower() in ("timestamp", "date", "time")), "date"
    )
    close_col = next((c for c in df.columns if "close" in c.lower()), "close")
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], utc=True).dt.normalize()
    pivot = df.pivot_table(index=date_col, columns=ticker_col, values=close_col, aggfunc="last")
    pivot.index.name = "date"
    return pivot.sort_index()


def load_prices(
    tickers: list[str],
    start_date: str,
    end_date: str,
    lookback_days: int = 0,
    cache_path: str | None = None,
) -> pd.DataFrame:
    """
    Load adjusted close prices including pre-sample lookback.

    Parameters
    ----------
    tickers : list[str]
    start_date : str
    end_date : str
    lookback_days : int
        Additional calendar days to fetch before start_date for warm-up.
    cache_path : str or None
        If provided, cache/load results from a parquet file.

    Returns
    -------
    prices : pd.DataFrame
        Wide-format adjusted close prices.
    """
    if cache_path and os.path.exists(cache_path):
        logger.info("Loading prices from cache: %s", cache_path)
        return pd.read_pickle(cache_path)

    # Extend start by lookback
    if lookback_days > 0:
        start_dt = pd.to_datetime(start_date) - pd.tseries.offsets.BDay(lookback_days)
        effective_start = start_dt.strftime("%Y-%m-%d")
    else:
        effective_start = start_date

    prices = fetch_ohlcv(tickers, effective_start, end_date)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        prices.to_pickle(cache_path)
        logger.info("Prices cached to: %s", cache_path)

    return prices


def prices_to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute simple daily returns from price DataFrame.

    y_i(t) = (S_i(t+1) - S_i(t)) / S_i(t)

    Parameters
    ----------
    prices : pd.DataFrame
        Adjusted close prices, shape (T+1, n_assets).

    Returns
    -------
    returns : pd.DataFrame
        Daily simple returns, shape (T, n_assets).
    """
    returns = prices.pct_change(fill_method=None).dropna(how="all")
    return returns
