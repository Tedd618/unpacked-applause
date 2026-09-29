"""Daily market data: Yahoo Finance prices, Fama-French factors and exchange session closes."""

from __future__ import annotations

import io
import zipfile

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import requests
import yfinance as yf

from .config import COMPANIES, EXTRA_TICKERS, PRICE_START, PRICES_DIR, Company

FF_URLS = {
    "ff3": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip",
    "mom": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip",
}
FF_FILE = PRICES_DIR / "ff_factors_daily.parquet"


def price_file(ticker: str):
    return PRICES_DIR / f"{ticker.replace('^', 'idx_').replace('.', '_')}.parquet"


def all_tickers() -> list[str]:
    return sorted({c.ticker for c in COMPANIES.values()} | set(EXTRA_TICKERS))


def download_prices(tickers: list[str] | None = None, start: str = PRICE_START,
                    end: str | None = None) -> None:
    tickers = tickers or all_tickers()
    end = end or (pd.Timestamp.today().normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False,
                       group_by="ticker", threads=True)
    PRICES_DIR.mkdir(parents=True, exist_ok=True)
    for t in tickers:
        df = data[t].dropna(subset=["Close"]).copy()
        df.index = pd.to_datetime(df.index).tz_localize(None).rename("date")
        df.columns = [c.lower() for c in df.columns]
        df.to_parquet(price_file(t))
        print(f"prices: {t:10s} {df.index.min():%Y-%m-%d} .. {df.index.max():%Y-%m-%d} ({len(df)} days)")


EARNINGS_FILE = PRICES_DIR / "earnings_dates.parquet"


def download_earnings(limit: int = 100) -> None:
    """Historical earnings-announcement timestamps (Yahoo Finance) for every company."""
    frames = []
    for c in COMPANIES.values():
        df = yf.Ticker(c.ticker).get_earnings_dates(limit=limit)
        ts = pd.to_datetime(df.index)
        ts = ts.tz_localize("America/New_York") if ts.tz is None else ts
        frames.append(pd.DataFrame({"ticker": c.ticker, "announced_utc": ts.tz_convert("UTC")}))
    out = pd.concat(frames, ignore_index=True).drop_duplicates()
    out.to_parquet(EARNINGS_FILE, index=False)
    print(f"earnings: {len(out)} announcements for {out.ticker.nunique()} tickers")


def load_earnings(ticker: str) -> pd.Series:
    if not EARNINGS_FILE.exists():
        return pd.Series(dtype="datetime64[ns, UTC]")
    df = pd.read_parquet(EARNINGS_FILE)
    return df.loc[df.ticker == ticker, "announced_utc"].sort_values().reset_index(drop=True)


def load_prices(ticker: str) -> pd.DataFrame:
    return pd.read_parquet(price_file(ticker))


def _parse_french_csv(text: str) -> pd.DataFrame:
    """Parse the first (daily) table of a Ken French data-library CSV; values are percent."""
    lines = text.splitlines()
    header_idx = next(i for i, line in enumerate(lines) if line.startswith(","))
    header = ["date"] + [h.strip() for h in lines[header_idx].split(",")[1:]]
    rows = []
    for line in lines[header_idx + 1 :]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts[0]) != 8 or not parts[0].isdigit():
            break
        rows.append(parts)
    df = pd.DataFrame(rows, columns=header)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    return df.set_index("date").astype(float) / 100


def download_ff_factors() -> None:
    frames = []
    for url in FF_URLS.values():
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            frames.append(_parse_french_csv(z.read(z.namelist()[0]).decode("latin-1")))
    ff = pd.concat(frames, axis=1, join="inner")
    ff.columns = [c.replace("-", "_").lower() for c in ff.columns]  # mkt_rf, smb, hml, rf, mom
    ff.to_parquet(FF_FILE)
    print(f"ff factors: {ff.index.min():%Y-%m-%d} .. {ff.index.max():%Y-%m-%d} ({len(ff)} days)")


def load_ff_factors() -> pd.DataFrame | None:
    return pd.read_parquet(FF_FILE) if FF_FILE.exists() else None


def session_closes(exchange: str, index: pd.DatetimeIndex) -> pd.Series:
    """UTC close time of each trading day in ``index`` (handles early closes and rule changes)."""
    cal = xcals.get_calendar(exchange, start=index.min().strftime("%Y-%m-%d"),
                             end=index.max().strftime("%Y-%m-%d"))
    closes = cal.schedule["close"].copy()
    closes.index = pd.to_datetime(closes.index).tz_localize(None)
    out = closes.reindex(index)
    if out.isna().any():  # a day Yahoo reports but the calendar lacks: use the typical close
        typical = (closes - closes.index.tz_localize("UTC")).median()
        missing = out.isna()
        out[missing] = index[missing].tz_localize("UTC") + typical
    return out


def stock_panel(company: Company, benchmark: str | None = None) -> pd.DataFrame:
    """Daily frame on the stock's own trading days: ret, mkt, log volume, close time, factors."""
    stock = load_prices(company.ticker)
    mkt = load_prices(benchmark or company.market)
    # Yahoo lists some exchange holidays (e.g. Korea's 2015-08-14 temporary holiday) as
    # zero-volume rows; a day with no volume and no index print is not a trading day.
    stock = stock[~((stock["volume"] == 0) & ~stock.index.isin(mkt.index))]
    df = pd.DataFrame(index=stock.index)
    df["ret"] = stock["close"].pct_change()
    # Zero-volume days are trading halts (e.g. Samsung's 50:1 split in May 2018) whose flat
    # prices would masquerade as zero returns.
    df.loc[stock["volume"] == 0, "ret"] = np.nan
    df["mkt"] = mkt["close"].pct_change().reindex(df.index)
    # Robustness benchmark: Nasdaq-100 for US listings (KOSPI stays the market for Samsung).
    ndx = load_prices("^NDX") if company.currency == "USD" else mkt
    df["mkt_ndx"] = ndx["close"].pct_change().reindex(df.index)
    df["logvol"] = np.log1p(stock["volume"].astype(float))
    df["close_utc"] = session_closes(company.exchange, df.index)
    ff = load_ff_factors()
    if ff is not None and company.currency == "USD":
        df = df.join(ff, how="left")
    return df.iloc[1:]
