"""Yahoo Finance OHLC/options pipeline and dynamic fragility classifier.

One source file, one SQLite data store.

The classifier follows the audited MASS relation:

    excess_252 = alpha - beta * variance_252 + error

    beta > 0  -> FRAGILE
    beta < 0  -> ANTIFRAGILE

The current label uses a 252-observation rolling regression.  A label computed
after the close of session t is a signal for the next session, never for the
return of t.

Yahoo exposes current option chains, not a discoverable archive of expired
chains.  Every run therefore stores every currently listed expiration as a
timestamped snapshot.  Repeated runs build the historical option-snapshot
archive with exact observation timestamps.

OHLC rows keep both their market event timestamp and their Yahoo ingestion
timestamp, and every derived row is versioned by run_id. Historical regimes
are event-time causal inside that downloaded Yahoo vintage; Yahoo does not
provide a historical vendor-vintage archive.
"""

from __future__ import annotations

import argparse
import gzip
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, time as clock_time, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import statsmodels.api as sm
import yfinance as yf
from scipy.stats import linregress


RF = 0.02
VAR_WINDOW = 252
BETA_WINDOW = 252
MAX_STALENESS_DAYS = 10
MIN_HISTORY_COVERAGE = 0.80
BOOTSTRAP_REPS = 500
BOOTSTRAP_BLOCK = 21
NY = ZoneInfo("America/New_York")

TICKERS = tuple(
    """
A AA AAL AAPL ABBV ABT ACN ADBE ADI ADM ADP ADSK AEP AFG AFL AGNC AIG AKAM ALB ALGN ALK ALL
AMAT AMCX AMD AMGN AMP AMT AMZN AON APA APD APH APTV ASML AUPH AVB AVGO AXP AZO BA BABA BAC
BAP BAX BB BBD BBWI BBY BDX BEN BIDU BIIB BKNG BKR BLDP BLK BLNK BMRN BMY BOX BRK-B BSX BUD BX
C CAH CAT CB CBOE CCI CCJ CDNS CFG CGC CHKP CHT CHTR CI CIB CIM CL CLS CLX CMCSA CME CMI CNC COF
COHR COP COR COST COTY CPRI CRM CSCO CSX CTAS CTSH CVS CVX D DAL DD DE DG DHI DHR DIS DLR DLTR DTE
DUK DVN DXC DXCM EA EBAY EC ECL ED EDU EL EMN EMR ENPH EOG EPD EQIX EQR ESS ETD ETN ETSY EW EXC
EXPE F FANG FAST FCEL FCX FDX FIS FITB FOSL FSLR FTI FTNT FTV GD GE GFI GILD GIS GLW GM GNRC GOOG
GPN GPRO GRPN GRVY GS GT GWW HAL HBAN HCA HD HDB HLF HLT HOG HON HPE HPQ HRB HSBC HSY HTHT HUBS
HUM IAC IBM IBN ICE ICL IDXX ILMN INCY INFY INO INSG INSM INTC INTU IOVA IP IQV IRDM ISRG ITW IVZ
JBL JCI JD JNJ JPM KDP KEY KEYS KGC KHC KKR KLAC KMI KO KODK KR KSS KTOS LBTYK LC LEN LHX LITE LLY
LMT LOW LPL LRCX LULU LUMN LUV LVS LYB M MA MANU MAR MARA MAT MCD MCHP MCK MCO MDLZ MDT MELI MET
META MFA MFC MGM MLM MMM MNST MO MOH MOMO MPC MPLX MPWR MRK MRVL MS MSFT MTB MTCH MU NBIX NBR NCLH
NEE NEM NFLX NKE NOC NOW NTES NTRS NUE NVAX NVDA NVS NWS NXPI OIS OKE OMC ORCL ORLY OXY PANW PAYX
PCAR PCG PEG PENN PEP PFE PG PGR PH PLAY PLD PLUG PM PNC POOL PPG PPL PSA PSX PVH PYPL QCOM RACE
RCL REGN RF RGLD RIG RIOT RL RNG ROK ROP ROST RSG RTX SBUX SCHW SENS SHOP SHW SIG SIRI SLB SMG SNPS
SO SPG SPGI SRE STRL STT STX STZ SWK SWKS SYF SYK SYY T TCOM TDOC TEAM TEL TER TEVA TFC TGT TJX TM
TMO TMUS TRIP TRMB TROW TRV TSLA TSM TSN TTWO TWLO TXN UA UAL UL ULTA UMC UNH UNP UPS USB UUUU V
VALE VFC VIRT VLO VMC VRTX VTRS VZ W WDAY WDC WEN WFC WKHS WM WMB WMT WPM WU WY WYNN XEL XOM XRX
YUM Z ZBH ZBRA ZTS
""".split()
)

# Current Yahoo symbols for securities that changed ticker without breaking the
# requested identity.  Results remain keyed by the user's original ticker.
YAHOO_SYMBOL = {"IAC": "PPLI", "LC": "HAPN"}

assert len(TICKERS) == 419 and len(set(TICKERS)) == 419


SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    started_utc TEXT NOT NULL,
    completed_utc TEXT,
    as_of_date TEXT NOT NULL,
    ticker_count INTEGER NOT NULL,
    status TEXT NOT NULL,
    error TEXT
);

CREATE TABLE IF NOT EXISTS ohlc (
    run_id TEXT NOT NULL,
    ticker TEXT NOT NULL,
    yahoo_symbol TEXT NOT NULL,
    date TEXT NOT NULL,
    fetched_utc TEXT NOT NULL,
    event_close_utc TEXT NOT NULL,
    available_utc TEXT NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    adjusted_close REAL,
    volume REAL,
    dividends REAL,
    stock_splits REAL,
    repaired INTEGER,
    PRIMARY KEY (run_id, ticker, date)
);

CREATE TABLE IF NOT EXISTS option_snapshots (
    run_id TEXT NOT NULL,
    snapshot_utc TEXT NOT NULL,
    ticker TEXT NOT NULL,
    yahoo_symbol TEXT NOT NULL,
    expiration TEXT NOT NULL,
    option_type TEXT NOT NULL,
    contract_symbol TEXT NOT NULL,
    last_trade_utc TEXT,
    strike REAL,
    last_price REAL,
    bid REAL,
    ask REAL,
    mid REAL,
    change REAL,
    percent_change REAL,
    volume REAL,
    open_interest REAL,
    implied_volatility REAL,
    in_the_money INTEGER,
    contract_size TEXT,
    currency TEXT,
    dte INTEGER,
    underlying_spot REAL,
    underlying_quote_utc TEXT,
    PRIMARY KEY (run_id, ticker, contract_symbol)
);

CREATE TABLE IF NOT EXISTS regime_history (
    run_id TEXT NOT NULL,
    ticker TEXT NOT NULL,
    date TEXT NOT NULL,
    available_utc TEXT NOT NULL,
    adjusted_close REAL,
    log_return REAL,
    variance_252 REAL,
    excess_252 REAL,
    regression_observations_rolling INTEGER,
    rolling_mean_variance REAL,
    rolling_mean_excess REAL,
    rolling_covariance REAL,
    rolling_variance_x REAL,
    regression_slope REAL,
    beta_126 REAL,
    beta_rolling REAL,
    beta_504 REAL,
    alpha_rolling REAL,
    r2_rolling REAL,
    predicted_excess REAL,
    regression_residual REAL,
    regime TEXT,
    leg TEXT,
    sign_consistency_63 REAL,
    sign_consistency_126 REAL,
    sign_consistency_252 REAL,
    window_sign_consensus INTEGER,
    signal_effective_date TEXT,
    signal_for_next_session INTEGER NOT NULL,
    PRIMARY KEY (run_id, ticker, date)
);

CREATE TABLE IF NOT EXISTS classifications (
    run_id TEXT NOT NULL,
    ticker TEXT NOT NULL,
    yahoo_symbol TEXT NOT NULL,
    first_price_date TEXT,
    last_price_date TEXT,
    history_years REAL,
    staleness_days INTEGER,
    coverage_ratio REAL,
    price_basis TEXT,
    eligible_10y INTEGER NOT NULL,
    observations INTEGER NOT NULL,
    regression_observations INTEGER NOT NULL,
    beta_full REAL,
    alpha_full REAL,
    r2_full REAL,
    beta_rolling REAL,
    alpha_rolling REAL,
    r2_rolling REAL,
    regime_date TEXT,
    regime_available_utc TEXT,
    regime TEXT,
    error TEXT,
    PRIMARY KEY (run_id, ticker)
);

CREATE TABLE IF NOT EXISTS classification_validation (
    run_id TEXT NOT NULL,
    ticker TEXT NOT NULL,
    leg TEXT NOT NULL,
    beta_126 REAL,
    beta_252 REAL,
    beta_504 REAL,
    independent_beta_error REAL,
    classification_rule_valid INTEGER NOT NULL DEFAULT 0,
    hac_beta REAL,
    hac_se REAL,
    hac_ci_low REAL,
    hac_ci_high REAL,
    hac_pvalue REAL,
    hac_fdr_qvalue REAL,
    hac_valid INTEGER NOT NULL,
    bootstrap_sign_probability REAL,
    bootstrap_ci_low REAL,
    bootstrap_ci_high REAL,
    bootstrap_valid INTEGER NOT NULL,
    window_sign_consensus INTEGER NOT NULL,
    delete_block_sign_stable INTEGER NOT NULL,
    winsorized_sign_stable INTEGER NOT NULL,
    sign_consistency_63 REAL,
    sign_consistency_126 REAL,
    sign_consistency_252 REAL,
    convexity_coefficient REAL,
    convexity_pvalue REAL,
    jensen_convexity REAL,
    k_threshold_05 REAL,
    k_threshold_01 REAL,
    left_tail_semi_vega_05 REAL,
    left_tail_semi_vega_01 REAL,
    right_tail_semi_vega_95 REAL,
    right_tail_semi_vega_99 REAL,
    tail_asymmetry_05_95 REAL,
    tail_asymmetry_01_99 REAL,
    left_tail_robust INTEGER NOT NULL,
    right_tail_beneficial INTEGER NOT NULL,
    mass_validated INTEGER NOT NULL,
    taleb_validated INTEGER NOT NULL,
    pair_eligible INTEGER NOT NULL,
    passed_checks INTEGER NOT NULL,
    classification_confidence TEXT NOT NULL,
    PRIMARY KEY (run_id, ticker)
);

CREATE TABLE IF NOT EXISTS aligned_snapshots (
    run_id TEXT NOT NULL,
    ohlc_run_id TEXT NOT NULL,
    ticker TEXT NOT NULL,
    yahoo_symbol TEXT NOT NULL,
    option_status TEXT NOT NULL,
    option_snapshot_start_utc TEXT,
    option_snapshot_end_utc TEXT,
    alignment_cutoff_utc TEXT,
    ohlc_date TEXT,
    ohlc_available_utc TEXT,
    alignment_lag_seconds REAL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    adjusted_close REAL,
    volume REAL,
    regime TEXT,
    regime_date TEXT,
    regime_available_utc TEXT,
    beta_rolling REAL,
    alpha_rolling REAL,
    r2_rolling REAL,
    option_expirations INTEGER,
    option_contracts INTEGER,
    underlying_spot REAL,
    call_open_interest REAL,
    put_open_interest REAL,
    put_call_oi_ratio REAL,
    call_volume REAL,
    put_volume REAL,
    put_call_volume_ratio REAL,
    atm_expiration TEXT,
    atm_call_iv REAL,
    atm_put_iv REAL,
    atm_iv_mean REAL,
    iv_skew_90p_110c REAL,
    parity_median_abs_error REAL,
    PRIMARY KEY (run_id, ticker)
);

CREATE TABLE IF NOT EXISTS download_status (
    run_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    ticker TEXT NOT NULL,
    status TEXT NOT NULL,
    item_count INTEGER NOT NULL,
    details TEXT,
    PRIMARY KEY (run_id, kind, ticker)
);

CREATE INDEX IF NOT EXISTS idx_ohlc_available ON ohlc(run_id, ticker, available_utc);
CREATE INDEX IF NOT EXISTS idx_regime_available ON regime_history(run_id, ticker, available_utc);
CREATE INDEX IF NOT EXISTS idx_options_ticker_snapshot ON option_snapshots(ticker, snapshot_utc);
"""


def open_database(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=60)
    connection.execute("PRAGMA foreign_keys=ON")
    connection.executescript(SCHEMA)
    aligned_columns = {
        row[1] for row in connection.execute("PRAGMA table_info(aligned_snapshots)").fetchall()
    }
    if "ohlc_run_id" not in aligned_columns:
        connection.execute("ALTER TABLE aligned_snapshots ADD COLUMN ohlc_run_id TEXT")
    regime_columns = {
        row[1] for row in connection.execute("PRAGMA table_info(regime_history)").fetchall()
    }
    regime_additions = {
        "regression_observations_rolling": "INTEGER",
        "rolling_mean_variance": "REAL",
        "rolling_mean_excess": "REAL",
        "rolling_covariance": "REAL",
        "rolling_variance_x": "REAL",
        "regression_slope": "REAL",
        "beta_126": "REAL",
        "beta_504": "REAL",
        "predicted_excess": "REAL",
        "regression_residual": "REAL",
        "leg": "TEXT",
        "sign_consistency_63": "REAL",
        "sign_consistency_126": "REAL",
        "sign_consistency_252": "REAL",
        "window_sign_consensus": "INTEGER",
        "signal_effective_date": "TEXT",
    }
    for column, sql_type in regime_additions.items():
        if column not in regime_columns:
            connection.execute(f'ALTER TABLE regime_history ADD COLUMN "{column}" {sql_type}')
    validation_columns = {
        row[1] for row in connection.execute(
            "PRAGMA table_info(classification_validation)"
        ).fetchall()
    }
    validation_additions = {
        "independent_beta_error": "REAL",
        "classification_rule_valid": "INTEGER NOT NULL DEFAULT 0",
    }
    for column, sql_type in validation_additions.items():
        if column not in validation_columns:
            connection.execute(
                f'ALTER TABLE classification_validation ADD COLUMN "{column}" {sql_type}'
            )
    connection.commit()
    return connection


def upsert(connection: sqlite3.Connection, table: str, frame: pd.DataFrame, columns: list[str]) -> None:
    if frame.empty:
        return
    clean = frame.reindex(columns=columns).astype(object).where(pd.notna(frame.reindex(columns=columns)), None)
    quoted = ",".join(f'"{column}"' for column in columns)
    placeholders = ",".join("?" for _ in columns)
    connection.executemany(
        f'INSERT OR REPLACE INTO "{table}" ({quoted}) VALUES ({placeholders})',
        clean.itertuples(index=False, name=None),
    )
    connection.commit()


def extract_ohlc(
    raw: pd.DataFrame,
    provider: str,
    ticker: str,
    as_of: date,
    cutoff_utc: datetime,
    run_id: str,
    fetched_utc: datetime,
) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        selected = None
        for level in range(raw.columns.nlevels):
            values = raw.columns.get_level_values(level).astype(str)
            if provider in set(values):
                selected = raw.xs(provider, axis=1, level=level, drop_level=True).copy()
                break
        if selected is None:
            return pd.DataFrame()
    else:
        selected = raw.copy()

    selected.columns = [
        str(column[0] if isinstance(column, tuple) else column).strip().lower().replace(" ", "_").replace("?", "")
        for column in selected.columns
    ]
    selected = selected.rename(columns={"adj_close": "adjusted_close", "stock_splits": "stock_splits"})
    if "close" not in selected.columns:
        return pd.DataFrame()
    if "adjusted_close" not in selected.columns:
        selected["adjusted_close"] = selected["close"]

    index = pd.to_datetime(selected.index, errors="coerce")
    if getattr(index, "tz", None) is not None:
        index = index.tz_localize(None)
    selected = selected.loc[index.notna()].copy()
    selected.index = index[index.notna()].normalize()
    selected = selected.loc[~selected.index.duplicated(keep="last")].sort_index()

    for column in (
        "open", "high", "low", "close", "adjusted_close", "volume", "dividends", "stock_splits"
    ):
        if column not in selected.columns:
            selected[column] = np.nan if column not in {"dividends", "stock_splits"} else 0.0
        selected[column] = pd.to_numeric(selected[column], errors="coerce")

    selected = selected.loc[
        selected[["open", "high", "low", "close", "adjusted_close"]].notna().any(axis=1)
    ].copy()
    selected = selected.loc[selected.index.date <= as_of]
    if selected.empty:
        return selected

    available = [
        datetime.combine(ts.date(), clock_time(16, 30), tzinfo=NY).astimezone(timezone.utc)
        for ts in selected.index
    ]
    selected["available_dt"] = available
    selected = selected.loc[selected["available_dt"] <= cutoff_utc].copy()
    if selected.empty:
        return selected

    selected["ticker"] = ticker
    selected["yahoo_symbol"] = provider
    selected["run_id"] = run_id
    selected["fetched_utc"] = fetched_utc.isoformat()
    selected["date"] = selected.index.strftime("%Y-%m-%d")
    selected["event_close_utc"] = selected["available_dt"].map(lambda value: value.isoformat())
    selected["available_utc"] = selected["event_close_utc"]
    repaired = selected["repaired"] if "repaired" in selected.columns else False
    selected["repaired"] = pd.Series(repaired, index=selected.index).fillna(False).astype(bool).astype(int)
    return selected.reset_index(drop=True)


def download_ohlc(
    connection: sqlite3.Connection,
    tickers: list[str],
    run_id: str,
    as_of: date,
    cutoff_utc: datetime,
    batch_size: int,
) -> None:
    columns = [
        "run_id", "ticker", "yahoo_symbol", "date", "fetched_utc", "event_close_utc",
        "available_utc", "open", "high", "low", "close",
        "adjusted_close", "volume", "dividends", "stock_splits", "repaired",
    ]
    status_columns = ["run_id", "kind", "ticker", "status", "item_count", "details"]

    for offset in range(0, len(tickers), batch_size):
        batch = tickers[offset : offset + batch_size]
        providers = [YAHOO_SYMBOL.get(ticker, ticker) for ticker in batch]
        raw = pd.DataFrame()
        last_error = ""
        for attempt in range(3):
            try:
                raw = yf.download(
                    providers,
                    period="max",
                    interval="1d",
                    auto_adjust=False,
                    actions=True,
                    repair=True,
                    group_by="ticker",
                    multi_level_index=True,
                    threads=True,
                    progress=False,
                    timeout=30,
                )
                break
            except Exception as exc:  # Yahoo can fail transiently.
                last_error = str(exc)
                time.sleep(2**attempt)

        for ticker, provider in zip(batch, providers):
            frame = extract_ohlc(
                raw, provider, ticker, as_of, cutoff_utc, run_id, datetime.now(timezone.utc)
            )
            if frame.empty:
                # A partial batch response is retried once as an isolated symbol.
                try:
                    isolated = yf.download(
                        provider,
                        period="max",
                        interval="1d",
                        auto_adjust=False,
                        actions=True,
                        repair=True,
                        group_by="ticker",
                        multi_level_index=True,
                        threads=False,
                        progress=False,
                        timeout=30,
                    )
                    frame = extract_ohlc(
                        isolated, provider, ticker, as_of, cutoff_utc, run_id, datetime.now(timezone.utc)
                    )
                except Exception as exc:
                    last_error = str(exc)

            if frame.empty:
                status = pd.DataFrame(
                    [[run_id, "OHLC", ticker, "ERROR", 0, last_error or "no Yahoo OHLC rows"]],
                    columns=status_columns,
                )
            else:
                upsert(connection, "ohlc", frame, columns)
                status = pd.DataFrame(
                    [[run_id, "OHLC", ticker, "OK", len(frame), f"{frame.date.iloc[0]}..{frame.date.iloc[-1]}"]],
                    columns=status_columns,
                )
            upsert(connection, "download_status", status, status_columns)

        print(f"OHLC {min(offset + len(batch), len(tickers))}/{len(tickers)}")


def build_regimes(
    connection: sqlite3.Connection,
    tickers: list[str],
    run_id: str,
    as_of: date,
    cutoff_utc: datetime,
) -> pd.DataFrame:
    history_columns = [
        "run_id", "ticker", "date", "available_utc", "adjusted_close", "log_return", "variance_252",
        "excess_252", "regression_observations_rolling", "rolling_mean_variance",
        "rolling_mean_excess", "rolling_covariance", "rolling_variance_x", "regression_slope",
        "beta_126", "beta_rolling", "beta_504", "alpha_rolling", "r2_rolling",
        "predicted_excess", "regression_residual", "regime", "leg",
        "sign_consistency_63", "sign_consistency_126", "sign_consistency_252",
        "window_sign_consensus", "signal_effective_date", "signal_for_next_session",
    ]
    summary_rows: list[dict] = []
    cutoff_10y = pd.Timestamp(as_of) - pd.DateOffset(years=10)

    for number, ticker in enumerate(tickers, start=1):
        prices = pd.read_sql_query(
            """
            SELECT date, available_utc, close, adjusted_close
            FROM ohlc
            WHERE run_id=? AND ticker=? AND date<=? AND available_utc<=?
            ORDER BY date
            """,
            connection,
            params=(run_id, ticker, as_of.isoformat(), cutoff_utc.isoformat()),
        )
        error = None
        if prices.empty:
            summary_rows.append(
                {
                    "run_id": run_id, "ticker": ticker, "yahoo_symbol": YAHOO_SYMBOL.get(ticker, ticker),
                    "first_price_date": None, "last_price_date": None, "history_years": None,
                    "staleness_days": None, "coverage_ratio": 0.0, "price_basis": None,
                    "eligible_10y": 0, "observations": 0, "regression_observations": 0,
                    "beta_full": None, "alpha_full": None, "r2_full": None,
                    "beta_rolling": None, "alpha_rolling": None, "r2_rolling": None,
                    "regime_date": None, "regime_available_utc": None, "regime": "UNCLASSIFIED",
                    "error": "no OHLC data",
                }
            )
            continue

        prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
        prices = prices.dropna(subset=["date"]).drop_duplicates("date", keep="last").sort_values("date")
        adjusted = pd.to_numeric(prices["adjusted_close"], errors="coerce")
        close = pd.to_numeric(prices["close"], errors="coerce")
        adjusted_coverage = float(adjusted.gt(0).mean()) if len(adjusted) else 0.0
        close_coverage = float(close.gt(0).mean()) if len(close) else 0.0
        if adjusted_coverage >= 0.95:
            prices["price"] = adjusted.where(adjusted.gt(0))
            price_basis = "adjusted_close"
        elif close_coverage >= 0.95:
            prices["price"] = close.where(close.gt(0))
            price_basis = "close"
        else:
            prices["price"] = np.nan
            price_basis = None
        prices = prices.dropna(subset=["price"])
        if prices.empty:
            summary_rows.append(
                {
                    "run_id": run_id, "ticker": ticker, "yahoo_symbol": YAHOO_SYMBOL.get(ticker, ticker),
                    "first_price_date": None, "last_price_date": None, "history_years": None,
                    "staleness_days": None, "coverage_ratio": 0.0, "price_basis": price_basis,
                    "eligible_10y": 0, "observations": 0, "regression_observations": 0,
                    "beta_full": None, "alpha_full": None, "r2_full": None,
                    "beta_rolling": None, "alpha_rolling": None, "r2_rolling": None,
                    "regime_date": None, "regime_available_utc": None, "regime": "UNCLASSIFIED",
                    "error": "no coherent positive price series",
                }
            )
            continue

        first_date = prices["date"].min()
        last_date = prices["date"].max()
        staleness_days = int((pd.Timestamp(as_of) - last_date).days)
        expected = max(1, len(pd.bdate_range(cutoff_10y, pd.Timestamp(as_of))))
        observed = int(prices.loc[prices["date"].ge(cutoff_10y), "date"].nunique())
        coverage_ratio = observed / expected
        eligible = bool(
            first_date <= cutoff_10y
            and staleness_days <= MAX_STALENESS_DAYS
            and coverage_ratio >= MIN_HISTORY_COVERAGE
        )
        history_years = (pd.Timestamp(as_of) - first_date).days / 365.2425

        log_price = np.log(prices["price"])
        prices["log_return"] = log_price.diff()
        prices["variance_252"] = prices["log_return"].rolling(VAR_WINDOW, min_periods=VAR_WINDOW).var() * 252.0
        prices["excess_252"] = log_price.diff(VAR_WINDOW) - RF

        rolling_cov = prices["variance_252"].rolling(BETA_WINDOW, min_periods=BETA_WINDOW).cov(prices["excess_252"])
        rolling_var = prices["variance_252"].rolling(BETA_WINDOW, min_periods=BETA_WINDOW).var()
        prices["beta_rolling"] = -(rolling_cov / rolling_var.replace(0.0, np.nan))
        prices["beta_126"] = -(
            prices["variance_252"].rolling(126, min_periods=126).cov(prices["excess_252"])
            / prices["variance_252"].rolling(126, min_periods=126).var().replace(0.0, np.nan)
        )
        prices["beta_504"] = -(
            prices["variance_252"].rolling(504, min_periods=504).cov(prices["excess_252"])
            / prices["variance_252"].rolling(504, min_periods=504).var().replace(0.0, np.nan)
        )
        mean_x = prices["variance_252"].rolling(BETA_WINDOW, min_periods=BETA_WINDOW).mean()
        mean_y = prices["excess_252"].rolling(BETA_WINDOW, min_periods=BETA_WINDOW).mean()
        valid_pair = prices[["variance_252", "excess_252"]].notna().all(axis=1).astype(int)
        prices["regression_observations_rolling"] = valid_pair.rolling(
            BETA_WINDOW, min_periods=1
        ).sum()
        prices["rolling_mean_variance"] = mean_x
        prices["rolling_mean_excess"] = mean_y
        prices["rolling_covariance"] = rolling_cov
        prices["rolling_variance_x"] = rolling_var
        prices["regression_slope"] = -prices["beta_rolling"]
        prices["alpha_rolling"] = mean_y + prices["beta_rolling"] * mean_x
        prices["r2_rolling"] = prices["variance_252"].rolling(
            BETA_WINDOW, min_periods=BETA_WINDOW
        ).corr(prices["excess_252"]) ** 2
        prices["predicted_excess"] = (
            prices["alpha_rolling"] - prices["beta_rolling"] * prices["variance_252"]
        )
        prices["regression_residual"] = prices["excess_252"] - prices["predicted_excess"]
        prices["regime"] = np.select(
            [prices["beta_rolling"].gt(0), prices["beta_rolling"].lt(0)],
            ["FRAGILE", "ANTIFRAGILE"],
            default=None,
        )
        prices["leg"] = np.select(
            [prices["beta_rolling"].gt(0), prices["beta_rolling"].lt(0)],
            ["R+", "R-"],
            default=None,
        )
        valid_sign = prices["beta_rolling"].notna()
        positive_sign = prices["beta_rolling"].gt(0).where(valid_sign)
        for window in (63, 126, 252):
            positive_share = positive_sign.rolling(window, min_periods=1).mean()
            negative_share = 1.0 - positive_share
            prices[f"sign_consistency_{window}"] = np.where(
                prices["beta_rolling"].gt(0),
                positive_share,
                np.where(prices["beta_rolling"].lt(0), negative_share, np.nan),
            )
        signs = pd.concat(
            [
                np.sign(prices["beta_126"]),
                np.sign(prices["beta_rolling"]),
                np.sign(prices["beta_504"]),
            ],
            axis=1,
        )
        prices["window_sign_consensus"] = (
            signs.notna().all(axis=1) & signs.eq(signs.iloc[:, 0], axis=0).all(axis=1)
        ).astype(int)

        history = pd.DataFrame(
            {
                "run_id": run_id,
                "ticker": ticker,
                "date": prices["date"].dt.strftime("%Y-%m-%d"),
                "available_utc": prices["available_utc"],
                "adjusted_close": prices["price"],
                "log_return": prices["log_return"],
                "variance_252": prices["variance_252"],
                "excess_252": prices["excess_252"],
                "regression_observations_rolling": prices["regression_observations_rolling"],
                "rolling_mean_variance": prices["rolling_mean_variance"],
                "rolling_mean_excess": prices["rolling_mean_excess"],
                "rolling_covariance": prices["rolling_covariance"],
                "rolling_variance_x": prices["rolling_variance_x"],
                "regression_slope": prices["regression_slope"],
                "beta_126": prices["beta_126"],
                "beta_rolling": prices["beta_rolling"],
                "beta_504": prices["beta_504"],
                "alpha_rolling": prices["alpha_rolling"],
                "r2_rolling": prices["r2_rolling"],
                "predicted_excess": prices["predicted_excess"],
                "regression_residual": prices["regression_residual"],
                "regime": prices["regime"],
                "leg": prices["leg"],
                "sign_consistency_63": prices["sign_consistency_63"],
                "sign_consistency_126": prices["sign_consistency_126"],
                "sign_consistency_252": prices["sign_consistency_252"],
                "window_sign_consensus": prices["window_sign_consensus"],
                "signal_effective_date": prices["date"].shift(-1).dt.strftime("%Y-%m-%d"),
                "signal_for_next_session": 1,
            }
        )
        upsert(connection, "regime_history", history, history_columns)

        regression = prices[["variance_252", "excess_252"]].replace([np.inf, -np.inf], np.nan).dropna()
        beta_full = alpha_full = r2_full = np.nan
        if len(regression) >= 100 and regression["variance_252"].var() > 0:
            fit = linregress(regression["variance_252"], regression["excess_252"])
            beta_full, alpha_full, r2_full = -fit.slope, fit.intercept, fit.rvalue**2

        current = prices.iloc[-1]
        beta_current = current["beta_rolling"]
        if not eligible:
            reasons = []
            if first_date > cutoff_10y:
                reasons.append(f"first Yahoo price {first_date.date()} is after {cutoff_10y.date()}")
            if staleness_days > MAX_STALENESS_DAYS:
                reasons.append(f"last Yahoo price is stale by {staleness_days} days")
            if coverage_ratio < MIN_HISTORY_COVERAGE:
                reasons.append(f"10-year business-day coverage is {coverage_ratio:.1%}")
            regime = "EXCLUDED_HISTORY"
            error = "; ".join(reasons)
        elif not np.isfinite(beta_current):
            regime = "UNCLASSIFIED"
            error = "rolling beta unavailable"
        elif beta_current > 0:
            regime = "FRAGILE"
        elif beta_current < 0:
            regime = "ANTIFRAGILE"
        else:
            regime = "UNCLASSIFIED"
            error = "rolling beta equals zero"

        summary_rows.append(
            {
                "run_id": run_id,
                "ticker": ticker,
                "yahoo_symbol": YAHOO_SYMBOL.get(ticker, ticker),
                "first_price_date": first_date.strftime("%Y-%m-%d"),
                "last_price_date": last_date.strftime("%Y-%m-%d"),
                "history_years": history_years,
                "staleness_days": staleness_days,
                "coverage_ratio": coverage_ratio,
                "price_basis": price_basis,
                "eligible_10y": int(eligible),
                "observations": len(prices),
                "regression_observations": len(regression),
                "beta_full": beta_full,
                "alpha_full": alpha_full,
                "r2_full": r2_full,
                "beta_rolling": beta_current,
                "alpha_rolling": current["alpha_rolling"],
                "r2_rolling": current["r2_rolling"],
                "regime_date": last_date.strftime("%Y-%m-%d"),
                "regime_available_utc": current["available_utc"],
                "regime": regime,
                "error": error,
            }
        )
        if number % 25 == 0 or number == len(tickers):
            print(f"REGIMES {number}/{len(tickers)}")

    summaries = pd.DataFrame(summary_rows)
    summary_columns = [
        "run_id", "ticker", "yahoo_symbol", "first_price_date", "last_price_date", "history_years",
        "staleness_days", "coverage_ratio", "price_basis",
        "eligible_10y", "observations", "regression_observations", "beta_full", "alpha_full", "r2_full",
        "beta_rolling", "alpha_rolling", "r2_rolling", "regime_date", "regime_available_utc",
        "regime", "error",
    ]
    upsert(connection, "classifications", summaries, summary_columns)
    return summaries


def validate_classifications(
    connection: sqlite3.Connection,
    run_id: str,
    classifications: pd.DataFrame,
) -> pd.DataFrame:
    rng = np.random.default_rng(20260711)
    rows: list[dict] = []

    def beta_value(x: np.ndarray, y: np.ndarray) -> float:
        x0 = x - x.mean()
        y0 = y - y.mean()
        denominator = float(x0 @ x0)
        return -float(x0 @ y0 / denominator) if denominator > 0 else np.nan

    for number, item in enumerate(classifications.itertuples(index=False), start=1):
        ticker = item.ticker
        beta_current = float(item.beta_rolling) if pd.notna(item.beta_rolling) else np.nan
        current_sign = int(np.sign(beta_current)) if np.isfinite(beta_current) else 0
        leg = "R+" if current_sign > 0 else ("R-" if current_sign < 0 else "UNCLASSIFIED")
        data = pd.read_sql_query(
            """
            SELECT date,log_return,variance_252,excess_252,beta_126,beta_rolling,beta_504,
                   regime,sign_consistency_63,sign_consistency_126,sign_consistency_252
            FROM regime_history
            WHERE run_id=? AND ticker=?
            ORDER BY date
            """,
            connection,
            params=(run_id, ticker),
        )
        valid = data[["variance_252", "excess_252"]].replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        empty = {
            "run_id": run_id, "ticker": ticker, "leg": leg,
            "beta_126": np.nan, "beta_252": beta_current, "beta_504": np.nan,
            "independent_beta_error": np.nan,
            "classification_rule_valid": 0,
            "hac_beta": np.nan, "hac_se": np.nan, "hac_ci_low": np.nan,
            "hac_ci_high": np.nan, "hac_pvalue": np.nan,
            "bootstrap_sign_probability": np.nan, "bootstrap_ci_low": np.nan,
            "bootstrap_ci_high": np.nan, "window_sign_consensus": 0,
            "delete_block_sign_stable": 0, "winsorized_sign_stable": 0,
            "sign_consistency_63": np.nan, "sign_consistency_126": np.nan,
            "sign_consistency_252": np.nan, "convexity_coefficient": np.nan,
            "convexity_pvalue": np.nan, "jensen_convexity": np.nan,
            "k_threshold_05": np.nan, "k_threshold_01": np.nan,
            "left_tail_semi_vega_05": np.nan, "left_tail_semi_vega_01": np.nan,
            "right_tail_semi_vega_95": np.nan, "right_tail_semi_vega_99": np.nan,
            "tail_asymmetry_05_95": np.nan, "tail_asymmetry_01_99": np.nan,
            "left_tail_robust": 0, "right_tail_beneficial": 0,
        }
        if current_sign == 0 or len(valid) < 504:
            rows.append(empty)
            continue

        x_all = valid["variance_252"].to_numpy(float)
        y_all = valid["excess_252"].to_numpy(float)
        betas = {
            window: beta_value(x_all[-window:], y_all[-window:])
            for window in (126, 252, 504)
        }
        x = x_all[-252:]
        y = y_all[-252:]
        robust = sm.OLS(y, sm.add_constant(x)).fit().get_robustcov_results(
            cov_type="HAC", maxlags=63, use_correction=True
        )
        hac_beta = -float(robust.params[1])
        hac_se = float(robust.bse[1])
        hac_pvalue = float(robust.pvalues[1])
        hac_ci_low = hac_beta - 1.96 * hac_se
        hac_ci_high = hac_beta + 1.96 * hac_se

        blocks = int(np.ceil(252 / BOOTSTRAP_BLOCK))
        starts = rng.integers(0, 252, size=(BOOTSTRAP_REPS, blocks))
        indices = (
            starts[:, :, None] + np.arange(BOOTSTRAP_BLOCK)[None, None, :]
        ) % 252
        indices = indices.reshape(BOOTSTRAP_REPS, -1)[:, :252]
        xb = x[indices]
        yb = y[indices]
        xb0 = xb - xb.mean(axis=1, keepdims=True)
        yb0 = yb - yb.mean(axis=1, keepdims=True)
        bootstrap_betas = -(xb0 * yb0).sum(axis=1) / (xb0 * xb0).sum(axis=1)
        bootstrap_probability = float(
            np.mean(np.sign(bootstrap_betas) == current_sign)
        )
        bootstrap_ci_low = float(np.quantile(bootstrap_betas, 0.025))
        bootstrap_ci_high = float(np.quantile(bootstrap_betas, 0.975))

        deleted = []
        for start in range(0, 252, BOOTSTRAP_BLOCK):
            keep = np.ones(252, dtype=bool)
            keep[start:min(start + BOOTSTRAP_BLOCK, 252)] = False
            deleted.append(beta_value(x[keep], y[keep]))
        delete_stable = int(np.all(np.sign(deleted) == current_sign))
        x_winsor = np.clip(x, np.quantile(x, 0.01), np.quantile(x, 0.99))
        y_winsor = np.clip(y, np.quantile(y, 0.01), np.quantile(y, 0.99))
        winsor_stable = int(np.sign(beta_value(x_winsor, y_winsor)) == current_sign)

        standardized = (x - x.mean()) / x.std(ddof=1)
        quadratic = sm.OLS(
            y,
            np.column_stack(
                [np.ones(len(standardized)), standardized, standardized**2]
            ),
        ).fit().get_robustcov_results(
            cov_type="HAC", maxlags=21, use_correction=True
        )
        convexity = float(quadratic.params[2])
        convexity_pvalue = float(quadratic.pvalues[2])

        classified = data.dropna(
            subset=["regime", "variance_252", "log_return"]
        ).copy()
        classified["next_return"] = pd.to_numeric(
            classified["log_return"], errors="coerce"
        ).shift(-1)
        classified = classified.dropna(subset=["next_return"])
        q_low = float(classified["variance_252"].quantile(0.25))
        q_high = float(classified["variance_252"].quantile(0.75))
        low_stress = classified.loc[classified["variance_252"].le(q_low)]
        high_stress = classified.loc[classified["variance_252"].ge(q_high)]
        k05 = float(classified["next_return"].quantile(0.05))
        k01 = float(classified["next_return"].quantile(0.01))
        k95 = float(classified["next_return"].quantile(0.95))
        k99 = float(classified["next_return"].quantile(0.99))

        def tail_difference(threshold: float, side: str) -> float:
            if side == "left":
                low = (threshold - low_stress["next_return"]).clip(lower=0).mean()
                high = (threshold - high_stress["next_return"]).clip(lower=0).mean()
            else:
                low = (low_stress["next_return"] - threshold).clip(lower=0).mean()
                high = (high_stress["next_return"] - threshold).clip(lower=0).mean()
            return float(high - low)

        left05 = tail_difference(k05, "left")
        left01 = tail_difference(k01, "left")
        right95 = tail_difference(k95, "right")
        right99 = tail_difference(k99, "right")
        asymmetry05 = right95 - left05
        asymmetry01 = right99 - left01
        current_row = data.dropna(subset=["beta_rolling"]).iloc[-1]
        rows.append(
            {
                **empty,
                "beta_126": betas[126], "beta_252": betas[252],
                "beta_504": betas[504], "hac_beta": hac_beta, "hac_se": hac_se,
                "independent_beta_error": abs(betas[252] - beta_current),
                "classification_rule_valid": int(
                    np.sign(betas[252]) == current_sign
                ),
                "hac_ci_low": hac_ci_low, "hac_ci_high": hac_ci_high,
                "hac_pvalue": hac_pvalue,
                "bootstrap_sign_probability": bootstrap_probability,
                "bootstrap_ci_low": bootstrap_ci_low,
                "bootstrap_ci_high": bootstrap_ci_high,
                "window_sign_consensus": int(
                    all(np.sign(value) == current_sign for value in betas.values())
                ),
                "delete_block_sign_stable": delete_stable,
                "winsorized_sign_stable": winsor_stable,
                "sign_consistency_63": current_row["sign_consistency_63"],
                "sign_consistency_126": current_row["sign_consistency_126"],
                "sign_consistency_252": current_row["sign_consistency_252"],
                "convexity_coefficient": convexity,
                "convexity_pvalue": convexity_pvalue,
                "jensen_convexity": convexity,
                "k_threshold_05": k05, "k_threshold_01": k01,
                "left_tail_semi_vega_05": left05,
                "left_tail_semi_vega_01": left01,
                "right_tail_semi_vega_95": right95,
                "right_tail_semi_vega_99": right99,
                "tail_asymmetry_05_95": asymmetry05,
                "tail_asymmetry_01_99": asymmetry01,
                "left_tail_robust": int(asymmetry05 > 0 and asymmetry01 > 0),
                "right_tail_beneficial": int(right95 > 0 and right99 > 0),
            }
        )
        if number % 25 == 0 or number == len(classifications):
            print(f"VALIDATION {number}/{len(classifications)}")

    validations = pd.DataFrame(rows)
    finite = validations["hac_pvalue"].notna()
    pvalues = validations.loc[finite, "hac_pvalue"].to_numpy(float)
    order = np.argsort(pvalues)
    ranked = pvalues[order]
    adjusted = np.minimum.accumulate(
        (ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1]
    )[::-1].clip(max=1.0)
    qvalues = np.empty_like(adjusted)
    qvalues[order] = adjusted
    validations["hac_fdr_qvalue"] = np.nan
    validations.loc[finite, "hac_fdr_qvalue"] = qvalues
    validations["hac_valid"] = (
        validations["hac_fdr_qvalue"].lt(0.05)
        & (
            validations["hac_ci_low"].gt(0)
            | validations["hac_ci_high"].lt(0)
        )
    ).astype(int)
    validations["bootstrap_valid"] = (
        validations["bootstrap_sign_probability"].ge(0.95)
        & (
            validations["bootstrap_ci_low"].gt(0)
            | validations["bootstrap_ci_high"].lt(0)
        )
    ).astype(int)
    check_columns = [
        "hac_valid", "bootstrap_valid", "window_sign_consensus",
        "delete_block_sign_stable", "winsorized_sign_stable",
    ]
    validations["recent_sign_valid"] = (
        validations["sign_consistency_126"].ge(0.80)
    ).astype(int)
    validations["passed_checks"] = (
        validations[check_columns].sum(axis=1)
        + validations["recent_sign_valid"]
    ).astype(int)
    eligible_lookup = classifications.set_index("ticker")["eligible_10y"]
    validations["mass_validated"] = (
        validations["classification_rule_valid"].eq(1)
        & validations["ticker"].map(eligible_lookup).fillna(0).eq(1)
    ).astype(int)
    validations["taleb_validated"] = (
        validations["leg"].eq("R-")
        & validations["classification_rule_valid"].eq(1)
        & validations["convexity_coefficient"].gt(0)
        & validations["convexity_pvalue"].lt(0.05)
        & validations["left_tail_robust"].eq(1)
        & validations["right_tail_beneficial"].eq(1)
    ).astype(int)
    validations["pair_eligible"] = validations["mass_validated"].astype(int)
    validations["classification_confidence"] = np.select(
        [
            validations["passed_checks"].eq(6),
            validations["passed_checks"].ge(4),
        ],
        ["STRONG", "MEDIUM"],
        default="WEAK",
    )
    validation_columns = [
        "run_id", "ticker", "leg", "beta_126", "beta_252", "beta_504",
        "independent_beta_error", "classification_rule_valid",
        "hac_beta", "hac_se", "hac_ci_low", "hac_ci_high", "hac_pvalue",
        "hac_fdr_qvalue", "hac_valid", "bootstrap_sign_probability",
        "bootstrap_ci_low", "bootstrap_ci_high", "bootstrap_valid",
        "window_sign_consensus", "delete_block_sign_stable",
        "winsorized_sign_stable", "sign_consistency_63",
        "sign_consistency_126", "sign_consistency_252",
        "convexity_coefficient", "convexity_pvalue", "jensen_convexity",
        "k_threshold_05", "k_threshold_01", "left_tail_semi_vega_05",
        "left_tail_semi_vega_01", "right_tail_semi_vega_95",
        "right_tail_semi_vega_99", "tail_asymmetry_05_95",
        "tail_asymmetry_01_99", "left_tail_robust",
        "right_tail_beneficial", "mass_validated", "taleb_validated",
        "pair_eligible", "passed_checks", "classification_confidence",
    ]
    upsert(
        connection,
        "classification_validation",
        validations,
        validation_columns,
    )
    return validations


def download_options_one(ticker: str, run_id: str) -> tuple[pd.DataFrame, dict]:
    provider = YAHOO_SYMBOL.get(ticker, ticker)
    instrument = yf.Ticker(provider)
    expirations: tuple[str, ...] = ()
    errors: list[str] = []
    clean_empty_responses = 0
    for attempt in range(3):
        try:
            expirations = tuple(instrument.options or ())
            if expirations:
                break
            clean_empty_responses += 1
            if attempt < 2:
                time.sleep((15, 30)[attempt])
        except Exception as exc:
            errors.append(f"expirations:{exc}")
            time.sleep((15, 30, 60)[attempt])

    if not expirations:
        status = "NO_OPTIONS" if clean_empty_responses == 3 and not errors else "ERROR"
        return pd.DataFrame(), {
            "run_id": run_id, "kind": "OPTIONS", "ticker": ticker, "status": status,
            "item_count": 0, "details": "; ".join(errors)[-2000:] or "no listed expirations",
        }

    pieces: list[pd.DataFrame] = []
    expirations_ok = 0
    for expiration in expirations:
        chain = None
        for attempt in range(3):
            try:
                chain = instrument.option_chain(expiration, tz="UTC")
                break
            except Exception as exc:
                errors.append(f"{expiration}:{exc}")
                time.sleep((5, 15, 30)[attempt])
        if chain is None:
            continue

        fetched = datetime.now(timezone.utc)
        underlying = chain.underlying if isinstance(chain.underlying, dict) else {}
        spot = next(
            (
                float(underlying[key])
                for key in ("regularMarketPrice", "postMarketPrice", "preMarketPrice", "previousClose")
                if underlying.get(key) is not None and np.isfinite(float(underlying[key]))
            ),
            np.nan,
        )
        quote_epoch = underlying.get("regularMarketTime")
        try:
            quote_utc = datetime.fromtimestamp(float(quote_epoch), tz=timezone.utc).isoformat()
        except (TypeError, ValueError, OSError):
            quote_utc = None

        pieces_before = len(pieces)
        has_calls = chain.calls is not None and not chain.calls.empty
        has_puts = chain.puts is not None and not chain.puts.empty
        for frame, option_type in ((chain.calls, "call"), (chain.puts, "put")):
            if frame is None or frame.empty:
                continue
            part = frame.copy()
            part = part.rename(
                columns={
                    "contractSymbol": "contract_symbol", "lastTradeDate": "last_trade_utc",
                    "lastPrice": "last_price", "percentChange": "percent_change",
                    "openInterest": "open_interest", "impliedVolatility": "implied_volatility",
                    "inTheMoney": "in_the_money", "contractSize": "contract_size",
                }
            )
            for column in (
                "strike", "last_price", "bid", "ask", "change", "percent_change", "volume",
                "open_interest", "implied_volatility",
            ):
                if column not in part.columns:
                    part[column] = np.nan
                part[column] = pd.to_numeric(part[column], errors="coerce")
            for column in ("contract_symbol", "last_trade_utc", "in_the_money", "contract_size", "currency"):
                if column not in part.columns:
                    part[column] = None

            bid_ask_valid = part["bid"].gt(0) & part["ask"].gt(0) & part["ask"].ge(part["bid"])
            # A stale last trade is stored separately and is never treated as a
            # contemporaneous quote midpoint.
            part["mid"] = ((part["bid"] + part["ask"]) / 2.0).where(bid_ask_valid)
            last_trade = pd.to_datetime(part["last_trade_utc"], errors="coerce", utc=True)
            part["last_trade_utc"] = last_trade.map(lambda value: value.isoformat() if pd.notna(value) else None)
            part["run_id"] = run_id
            part["snapshot_utc"] = fetched.isoformat()
            part["ticker"] = ticker
            part["yahoo_symbol"] = provider
            part["expiration"] = expiration
            part["option_type"] = option_type
            part["dte"] = (pd.Timestamp(expiration).date() - fetched.date()).days
            part["underlying_spot"] = spot
            part["underlying_quote_utc"] = quote_utc
            part["in_the_money"] = part["in_the_money"].fillna(False).astype(bool).astype(int)
            pieces.append(part)
        if len(pieces) > pieces_before and has_calls and has_puts:
            expirations_ok += 1
            time.sleep(0.25)
        elif len(pieces) > pieces_before:
            errors.append(f"{expiration}:missing calls or puts")
        else:
            errors.append(f"{expiration}:empty calls and puts")

    raw = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    status = "OK" if expirations_ok == len(expirations) else "PARTIAL" if expirations_ok else "ERROR"
    return raw, {
        "run_id": run_id,
        "kind": "OPTIONS",
        "ticker": ticker,
        "status": status,
        "item_count": len(raw),
        "details": f"expirations={expirations_ok}/{len(expirations)}" + (
            f"; {'; '.join(errors)[-1500:]}" if errors else ""
        ),
    }


def summarize_options(raw: pd.DataFrame, status: dict) -> dict:
    base = {
        "option_status": status["status"],
        "option_snapshot_start_utc": None,
        "option_snapshot_end_utc": None,
        "option_information_cutoff_utc": None,
        "option_expirations": 0,
        "option_contracts": 0,
        "underlying_spot": np.nan,
        "call_open_interest": np.nan,
        "put_open_interest": np.nan,
        "put_call_oi_ratio": np.nan,
        "call_volume": np.nan,
        "put_volume": np.nan,
        "put_call_volume_ratio": np.nan,
        "atm_expiration": None,
        "atm_call_iv": np.nan,
        "atm_put_iv": np.nan,
        "atm_iv_mean": np.nan,
        "iv_skew_90p_110c": np.nan,
        "parity_median_abs_error": np.nan,
    }
    if raw.empty:
        return base

    frame = raw.copy()
    frame["snapshot_dt"] = pd.to_datetime(frame["snapshot_utc"], errors="coerce", utc=True)
    frame = frame.sort_values("snapshot_dt")
    base["option_snapshot_start_utc"] = frame["snapshot_dt"].min().isoformat()
    base["option_snapshot_end_utc"] = frame["snapshot_dt"].max().isoformat()
    # All expirations are aligned to the state available before collection
    # started. Underlying quote time is retained as a staleness field in the
    # raw table, not confused with information availability.
    base["option_information_cutoff_utc"] = frame["snapshot_dt"].min().isoformat()
    base["option_expirations"] = int(frame["expiration"].nunique())
    base["option_contracts"] = int(frame["contract_symbol"].nunique())

    spot_series = pd.to_numeric(frame["underlying_spot"], errors="coerce").dropna()
    spot = float(spot_series.iloc[-1]) if not spot_series.empty else np.nan
    base["underlying_spot"] = spot
    calls = frame.loc[frame["option_type"].eq("call")]
    puts = frame.loc[frame["option_type"].eq("put")]
    call_oi = float(pd.to_numeric(calls["open_interest"], errors="coerce").fillna(0).sum())
    put_oi = float(pd.to_numeric(puts["open_interest"], errors="coerce").fillna(0).sum())
    call_volume = float(pd.to_numeric(calls["volume"], errors="coerce").fillna(0).sum())
    put_volume = float(pd.to_numeric(puts["volume"], errors="coerce").fillna(0).sum())
    base.update(
        {
            "call_open_interest": call_oi,
            "put_open_interest": put_oi,
            "put_call_oi_ratio": put_oi / call_oi if call_oi > 0 else np.nan,
            "call_volume": call_volume,
            "put_volume": put_volume,
            "put_call_volume_ratio": put_volume / call_volume if call_volume > 0 else np.nan,
        }
    )

    if np.isfinite(spot) and spot > 0:
        expiries = frame[["expiration", "dte"]].drop_duplicates().sort_values("dte")
        valid = expiries.loc[expiries["dte"].ge(7)]
        if valid.empty:
            valid = expiries.loc[expiries["dte"].ge(0)]
        if not valid.empty:
            expiry = valid.iloc[0]["expiration"]
            near = frame.loc[frame["expiration"].eq(expiry)].copy()
            near["strike"] = pd.to_numeric(near["strike"], errors="coerce")
            near["implied_volatility"] = pd.to_numeric(near["implied_volatility"], errors="coerce")
            near_calls = near.loc[near["option_type"].eq("call")].dropna(subset=["strike"])
            near_puts = near.loc[near["option_type"].eq("put")].dropna(subset=["strike"])
            call_atm = near_calls.loc[(near_calls["strike"] - spot).abs().idxmin()] if not near_calls.empty else None
            put_atm = near_puts.loc[(near_puts["strike"] - spot).abs().idxmin()] if not near_puts.empty else None
            call_iv = float(call_atm["implied_volatility"]) if call_atm is not None else np.nan
            put_iv = float(put_atm["implied_volatility"]) if put_atm is not None else np.nan
            iv_values = [value for value in (call_iv, put_iv) if np.isfinite(value)]
            base["atm_expiration"] = expiry
            base["atm_call_iv"] = call_iv
            base["atm_put_iv"] = put_iv
            base["atm_iv_mean"] = float(np.mean(iv_values)) if iv_values else np.nan
            if not near_calls.empty and not near_puts.empty:
                call_110 = near_calls.loc[(near_calls["strike"] - 1.10 * spot).abs().idxmin()]
                put_90 = near_puts.loc[(near_puts["strike"] - 0.90 * spot).abs().idxmin()]
                base["iv_skew_90p_110c"] = float(
                    put_90["implied_volatility"] - call_110["implied_volatility"]
                )

        paired = calls[["expiration", "strike", "mid", "dte", "underlying_spot"]].merge(
            puts[["expiration", "strike", "mid", "underlying_spot"]],
            on=["expiration", "strike"],
            suffixes=("_call", "_put"),
        )
        paired["pair_spot"] = pd.to_numeric(
            paired["underlying_spot_call"], errors="coerce"
        ).where(
            pd.to_numeric(paired["underlying_spot_call"], errors="coerce").gt(0),
            pd.to_numeric(paired["underlying_spot_put"], errors="coerce"),
        )
        paired = paired.dropna(subset=["mid_call", "mid_put", "strike", "dte", "pair_spot"])
        paired = paired.loc[paired["dte"].gt(0)]
        if not paired.empty:
            maturity = paired["dte"] / 365.25
            parity = (paired["mid_call"] - paired["mid_put"]) - (
                paired["pair_spot"] - paired["strike"] * np.exp(-RF * maturity)
            )
            base["parity_median_abs_error"] = float(parity.abs().median())
    return base


def export_daily_calculations(
    connection: sqlite3.Connection,
    source_run_id: str,
    destination: Path,
) -> int:
    query = """
        SELECT
            h.run_id AS source_run_id,
            h.ticker,
            c.yahoo_symbol,
            h.date,
            h.available_utc AS calculation_available_utc,
            h.signal_effective_date,
            h.signal_for_next_session,
            o.open,
            o.high,
            o.low,
            o.close,
            o.adjusted_close,
            o.volume,
            o.dividends,
            o.stock_splits,
            o.repaired,
            h.adjusted_close AS calculation_price,
            c.price_basis,
            h.log_return,
            252 AS variance_window,
            h.variance_252,
            0.02 AS risk_free_252,
            h.excess_252,
            252 AS regression_window,
            h.regression_observations_rolling,
            h.rolling_mean_variance,
            h.rolling_mean_excess,
            h.rolling_covariance,
            h.rolling_variance_x,
            h.regression_slope,
            h.beta_126,
            h.beta_rolling AS beta_252,
            h.beta_504,
            h.alpha_rolling,
            h.r2_rolling,
            h.predicted_excess,
            h.regression_residual,
            h.leg,
            h.sign_consistency_63,
            h.sign_consistency_126,
            h.sign_consistency_252,
            h.window_sign_consensus,
            CASE
                WHEN c.eligible_10y=0 THEN 'EXCLUDED_HISTORY'
                WHEN h.regime IS NULL THEN 'UNCLASSIFIED'
                ELSE h.regime
            END AS daily_regime,
            c.eligible_10y,
            c.first_price_date,
            c.last_price_date,
            c.history_years,
            c.staleness_days,
            c.coverage_ratio,
            c.error AS classification_error
        FROM regime_history h
        JOIN ohlc o
          ON o.run_id=h.run_id AND o.ticker=h.ticker AND o.date=h.date
        JOIN classifications c
          ON c.run_id=h.run_id AND c.ticker=h.ticker
        WHERE h.run_id=?
        ORDER BY h.ticker,h.date
    """
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    header = True
    if destination.name.lower().endswith(".gz"):
        output = gzip.open(destination, "wt", encoding="utf-8", newline="")
    else:
        output = destination.open("w", encoding="utf-8", newline="")
    with output:
        for chunk in pd.read_sql_query(
            query, connection, params=(source_run_id,), chunksize=100_000
        ):
            chunk.to_csv(
                output,
                index=False,
                header=header,
                lineterminator="\n",
                float_format="%.12g",
            )
            rows += len(chunk)
            header = False
            print(f"DAILY EXPORT {rows} rows")
    if rows == 0:
        raise RuntimeError(f"no daily calculations found for run {source_run_id}")
    print(f"DAILY EXPORT COMPLETE | rows={rows} | file={destination}")
    return rows


def export_classification_results(
    connection: sqlite3.Connection,
    source_run_id: str,
    destination: Path,
) -> int:
    results = pd.read_sql_query(
        """
        SELECT c.*,v.leg,v.beta_126,v.beta_252,v.beta_504,
               v.independent_beta_error,v.classification_rule_valid,
               v.hac_beta,v.hac_se,v.hac_ci_low,v.hac_ci_high,
               v.hac_pvalue,v.hac_fdr_qvalue,v.hac_valid,
               v.bootstrap_sign_probability,v.bootstrap_ci_low,
               v.bootstrap_ci_high,v.bootstrap_valid,
               v.window_sign_consensus,v.delete_block_sign_stable,
               v.winsorized_sign_stable,v.sign_consistency_63,
               v.sign_consistency_126,v.sign_consistency_252,
               v.convexity_coefficient,v.convexity_pvalue,
               v.jensen_convexity,v.k_threshold_05,v.k_threshold_01,
               v.left_tail_semi_vega_05,v.left_tail_semi_vega_01,
               v.right_tail_semi_vega_95,v.right_tail_semi_vega_99,
               v.tail_asymmetry_05_95,v.tail_asymmetry_01_99,
               v.left_tail_robust,v.right_tail_beneficial,
               v.mass_validated,v.taleb_validated,v.pair_eligible,
               v.passed_checks,v.classification_confidence
        FROM classifications c
        JOIN classification_validation v
          ON v.run_id=c.run_id AND v.ticker=c.ticker
        WHERE c.run_id=?
        ORDER BY c.ticker
        """,
        connection,
        params=(source_run_id,),
    )
    if results.empty:
        raise RuntimeError(
            f"no classification validation found for run {source_run_id}"
        )
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(
        destination,
        index=False,
        lineterminator="\n",
        float_format="%.12g",
    )
    print(
        f"CLASSIFICATION EXPORT COMPLETE | rows={len(results)} | file={destination}"
    )
    return len(results)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download Yahoo OHLC and all current option chains, align them causally, and classify regimes."
    )
    parser.add_argument("--db", default="yahoo_fragility.sqlite")
    parser.add_argument("--as-of", default=date.today().isoformat())
    parser.add_argument("--ohlc-batch-size", type=int, default=40)
    parser.add_argument("--option-workers", type=int, default=2)
    parser.add_argument("--tickers", help="Optional comma-separated subset for a controlled run")
    parser.add_argument(
        "--daily-export",
        default="fragility_daily_history.csv.gz",
        help="Daily calculation/classification export (.csv or .csv.gz)",
    )
    parser.add_argument(
        "--classification-export",
        default="fragility_classification_results.csv",
        help="Complete current R+/R- validation export",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Rebuild and export the latest stored daily calculations without downloading Yahoo again",
    )
    parser.add_argument("--skip-options", action="store_true", help="Skip option snapshots; OHLC/regimes still run")
    parser.add_argument(
        "--options-only",
        action="store_true",
        help="Reuse the latest complete OHLC/classification run and download only option snapshots",
    )
    args = parser.parse_args()

    as_of = pd.Timestamp(args.as_of).date()
    today = date.today()
    if as_of > today:
        raise ValueError("as-of cannot be in the future")
    if not args.skip_options and not args.export_only and as_of != today:
        raise ValueError(
            "Yahoo option chains are current snapshots; use today's as-of or pass --skip-options"
        )
    tickers = list(TICKERS)
    if args.tickers:
        requested = [value.strip().upper() for value in args.tickers.split(",") if value.strip()]
        unknown = sorted(set(requested) - set(TICKERS))
        if unknown:
            raise ValueError(f"tickers outside the locked universe: {unknown}")
        tickers = list(dict.fromkeys(requested))
    if args.ohlc_batch_size < 1 or args.option_workers < 1:
        raise ValueError("batch size and worker count must be positive")
    if args.skip_options and args.options_only:
        raise ValueError("--skip-options and --options-only are mutually exclusive")
    if args.export_only and (args.skip_options or args.options_only or args.tickers):
        raise ValueError("--export-only cannot be combined with --skip-options, --options-only or --tickers")

    database = Path(args.db).resolve()
    if args.export_only:
        connection = open_database(database)
        try:
            source = connection.execute(
                """
                SELECT r.run_id,r.started_utc
                FROM runs r
                WHERE r.as_of_date=? AND r.status IN ('COMPLETE','PARTIAL')
                  AND EXISTS (
                      SELECT 1 FROM regime_history h WHERE h.run_id=r.run_id
                  )
                ORDER BY r.started_utc DESC
                LIMIT 1
                """,
                (as_of.isoformat(),),
            ).fetchone()
            if source is None:
                raise RuntimeError("no stored daily calculation run is available for --export-only")
            source_run_id, source_started_utc = source
            source_tickers = [
                row[0] for row in connection.execute(
                    "SELECT ticker FROM classifications WHERE run_id=? ORDER BY ticker",
                    (source_run_id,),
                ).fetchall()
            ]
            classifications = build_regimes(
                connection,
                source_tickers,
                source_run_id,
                as_of,
                pd.Timestamp(source_started_utc).to_pydatetime(),
            )
            validate_classifications(
                connection, source_run_id, classifications
            )
            export_classification_results(
                connection, source_run_id, Path(args.classification_export)
            )
            export_daily_calculations(
                connection, source_run_id, Path(args.daily_export)
            )
            return 0
        finally:
            connection.close()

    started = datetime.now(timezone.utc)
    run_id = started.strftime("%Y%m%dT%H%M%S.%fZ")
    connection = open_database(database)
    connection.execute(
        "INSERT INTO runs(run_id,started_utc,as_of_date,ticker_count,status) VALUES(?,?,?,?,?)",
        (run_id, started.isoformat(), as_of.isoformat(), len(tickers), "RUNNING"),
    )
    connection.commit()

    try:
        print(f"RUN {run_id} | tickers={len(tickers)} | as_of={as_of} | db={database}")
        if args.options_only:
            source = connection.execute(
                """
                SELECT r.run_id
                FROM runs r
                WHERE r.status='COMPLETE' AND r.as_of_date=? AND r.ticker_count=?
                  AND (SELECT COUNT(*) FROM classifications c WHERE c.run_id=r.run_id)=?
                ORDER BY r.started_utc DESC
                LIMIT 1
                """,
                (as_of.isoformat(), len(tickers), len(tickers)),
            ).fetchone()
            if source is None:
                raise RuntimeError("no complete OHLC/classification run is available for --options-only")
            ohlc_run_id = source[0]
            classifications = pd.read_sql_query(
                "SELECT * FROM classifications WHERE run_id=? ORDER BY ticker",
                connection,
                params=(ohlc_run_id,),
            )
            print(f"REUSE OHLC RUN {ohlc_run_id}")
        else:
            ohlc_run_id = run_id
            download_ohlc(connection, tickers, run_id, as_of, started, args.ohlc_batch_size)
            classifications = build_regimes(connection, tickers, run_id, as_of, started)
            validate_classifications(connection, ohlc_run_id, classifications)
            export_classification_results(
                connection, ohlc_run_id, Path(args.classification_export)
            )
            export_daily_calculations(
                connection, ohlc_run_id, Path(args.daily_export)
            )

        option_columns = [
            "run_id", "snapshot_utc", "ticker", "yahoo_symbol", "expiration", "option_type",
            "contract_symbol", "last_trade_utc", "strike", "last_price", "bid", "ask", "mid",
            "change", "percent_change", "volume", "open_interest", "implied_volatility",
            "in_the_money", "contract_size", "currency", "dte", "underlying_spot",
            "underlying_quote_utc",
        ]
        status_columns = ["run_id", "kind", "ticker", "status", "item_count", "details"]
        summaries: dict[str, dict] = {}
        if args.skip_options:
            for ticker in tickers:
                status = {
                    "run_id": run_id, "kind": "OPTIONS", "ticker": ticker, "status": "SKIPPED",
                    "item_count": 0, "details": "--skip-options",
                }
                summaries[ticker] = summarize_options(pd.DataFrame(), status)
                upsert(connection, "download_status", pd.DataFrame([status]), status_columns)
        else:
            with ThreadPoolExecutor(max_workers=args.option_workers) as executor:
                futures = {executor.submit(download_options_one, ticker, run_id): ticker for ticker in tickers}
                for number, future in enumerate(as_completed(futures), start=1):
                    ticker = futures[future]
                    try:
                        raw_options, status = future.result()
                    except Exception as exc:
                        raw_options = pd.DataFrame()
                        status = {
                            "run_id": run_id, "kind": "OPTIONS", "ticker": ticker, "status": "ERROR",
                            "item_count": 0, "details": str(exc)[-2000:],
                        }
                    upsert(connection, "option_snapshots", raw_options, option_columns)
                    upsert(connection, "download_status", pd.DataFrame([status]), status_columns)
                    summaries[ticker] = summarize_options(raw_options, status)
                    print(f"OPTIONS {number}/{len(tickers)} | {ticker} | {status['status']} | rows={len(raw_options)}")

        aligned_rows: list[dict] = []
        classification_lookup = classifications.set_index("ticker").to_dict("index")
        for ticker in tickers:
            option = summaries[ticker]
            alignment_cutoff = option["option_information_cutoff_utc"] or started.isoformat()
            ohlc = pd.read_sql_query(
                """
                SELECT date,available_utc,open,high,low,close,adjusted_close,volume
                FROM ohlc
                WHERE run_id=? AND ticker=? AND date<=? AND available_utc<=?
                ORDER BY available_utc DESC LIMIT 1
                """,
                connection,
                params=(ohlc_run_id, ticker, as_of.isoformat(), alignment_cutoff),
            )
            regime = pd.read_sql_query(
                """
                SELECT date,available_utc,beta_rolling,alpha_rolling,r2_rolling,regime
                FROM regime_history
                WHERE run_id=? AND ticker=? AND date<=? AND available_utc<=?
                ORDER BY available_utc DESC LIMIT 1
                """,
                connection,
                params=(ohlc_run_id, ticker, as_of.isoformat(), alignment_cutoff),
            )
            market = ohlc.iloc[0].to_dict() if not ohlc.empty else {}
            state = regime.iloc[0].to_dict() if not regime.empty else {}
            final_classification = classification_lookup[ticker]
            final_regime = state.get("regime")
            if not bool(final_classification.get("eligible_10y")):
                final_regime = final_classification.get("regime")
            elif final_classification.get("regime") == "UNCLASSIFIED":
                final_regime = "UNCLASSIFIED"
            market_available = market.get("available_utc")
            lag = np.nan
            if market_available:
                lag = (
                    pd.Timestamp(alignment_cutoff) - pd.Timestamp(market_available)
                ).total_seconds()
                if lag < 0:
                    raise RuntimeError(f"temporal alignment leak for {ticker}")
            aligned_rows.append(
                {
                    "run_id": run_id,
                    "ohlc_run_id": ohlc_run_id,
                    "ticker": ticker,
                    "yahoo_symbol": YAHOO_SYMBOL.get(ticker, ticker),
                    **option,
                    "alignment_cutoff_utc": alignment_cutoff,
                    "ohlc_date": market.get("date"),
                    "ohlc_available_utc": market_available,
                    "alignment_lag_seconds": lag,
                    "open": market.get("open"),
                    "high": market.get("high"),
                    "low": market.get("low"),
                    "close": market.get("close"),
                    "adjusted_close": market.get("adjusted_close"),
                    "volume": market.get("volume"),
                    "regime": final_regime,
                    "regime_date": state.get("date"),
                    "regime_available_utc": state.get("available_utc"),
                    "beta_rolling": state.get("beta_rolling"),
                    "alpha_rolling": state.get("alpha_rolling"),
                    "r2_rolling": state.get("r2_rolling"),
                }
            )

        aligned = pd.DataFrame(aligned_rows)
        aligned_columns = [
            "run_id", "ohlc_run_id", "ticker", "yahoo_symbol", "option_status",
            "option_snapshot_start_utc",
            "option_snapshot_end_utc", "alignment_cutoff_utc", "ohlc_date", "ohlc_available_utc",
            "alignment_lag_seconds",
            "open", "high", "low", "close", "adjusted_close", "volume", "regime", "beta_rolling",
            "regime_date", "regime_available_utc", "alpha_rolling", "r2_rolling",
            "option_expirations", "option_contracts", "underlying_spot",
            "call_open_interest", "put_open_interest", "put_call_oi_ratio", "call_volume", "put_volume",
            "put_call_volume_ratio", "atm_expiration", "atm_call_iv", "atm_put_iv", "atm_iv_mean",
            "iv_skew_90p_110c", "parity_median_abs_error",
        ]
        upsert(connection, "aligned_snapshots", aligned, aligned_columns)

        completed = datetime.now(timezone.utc)
        bad_downloads = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM download_status
                WHERE run_id=? AND status IN ('ERROR','PARTIAL')
                """,
                (run_id,),
            ).fetchone()[0]
        )
        final_status = "COMPLETE" if bad_downloads == 0 else "PARTIAL"
        connection.execute(
            "UPDATE runs SET completed_utc=?,status=? WHERE run_id=?",
            (completed.isoformat(), final_status, run_id),
        )
        connection.commit()
        counts = classifications["regime"].value_counts(dropna=False).to_dict()
        print(f"CLASSIFICATION {counts}")
        print(
            "No-lookahead check: every aligned OHLC/regime timestamp is from the current run "
            "and is <= the conservative information cutoff."
        )
        print(f"{final_status} | bad_downloads={bad_downloads} | database={database}")
        return 0 if bad_downloads == 0 else 2
    except Exception as exc:
        connection.execute(
            "UPDATE runs SET completed_utc=?,status='ERROR',error=? WHERE run_id=?",
            (datetime.now(timezone.utc).isoformat(), str(exc)[-4000:], run_id),
        )
        connection.commit()
        raise
    finally:
        connection.close()


if __name__ == "__main__":
    raise SystemExit(main())
