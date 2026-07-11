# Dynamic R+/R- Fragility Classifier

This repository contains a deterministic pipeline that downloads and stores Yahoo Finance data, builds a date-aligned panel without look-ahead leakage, and assigns every eligible asset to exactly one response leg on every classified date:

- `R+`: fragile response leg.
- `R-`: antifragile response leg.

The option layer is kept separate from the R+/R- classification. Missing option chains do not cancel or alter a classification derived from OHLC data.

## Repository contents

| File | Description |
|---|---|
| `yahoo_fragility_pipeline.py` | Single-file data pipeline and R+/R- classifier. |
| `yahoo_fragility.sqlite.gz` | Compressed SQLite store for downloaded and derived data. Restore it locally before running database queries. |
| `fragility_classification_results.csv` | Current cross-sectional classifications and diagnostics. |
| `fragility_daily_history.csv.gz` | Complete daily calculations and classifications. |
| `dynamic_rplus_rminus_classifier_en.pdf` | English methodology and empirical results. |
| `README.md` | Project documentation and option-data fallback policy. |
| `LICENSE` | Proprietary license terms. |

## Restoring the SQLite store

GitHub LFS rejects individual objects above 2 GB on this account tier. The SQLite store is therefore committed as `yahoo_fragility.sqlite.gz`. Restore the working database locally with:

```powershell
python -c "import gzip, shutil; shutil.copyfileobj(gzip.open('yahoo_fragility.sqlite.gz','rb'), open('yahoo_fragility.sqlite','wb'))"
```

The uncompressed `yahoo_fragility.sqlite` file is intentionally ignored by Git.

## Synthetic option pricing when market chains are missing

Synthetic option values can be generated when an option chain is unavailable, but they must remain model estimates. Black-Scholes and related models do not price the underlying asset: they price an option from the observed underlying price and a set of assumptions.

The classifier should therefore follow this logic:

```text
OHLC data available
    -> compute the response coefficient
    -> assign R+ or R-

Option chain unavailable
    -> set option_status = MISSING
    -> preserve the OHLC-based R+/R- classification
    -> optionally generate synthetic option values in a separate model layer
```

Synthetic values must never be written into fields representing observed market data. In particular, the pipeline must not fabricate `bid`, `ask`, `volume`, `open_interest`, or market `implied_volatility`.

## Model selection by instrument

| Instrument | Appropriate baseline model | Main considerations |
|---|---|---|
| European equity option | Black-Scholes-Merton | Continuous dividend yield and volatility estimate. |
| American equity or ETF option | Binomial tree or finite-difference model | Discrete dividends and early exercise. |
| FX option | Garman-Kohlhagen | Domestic and foreign interest rates, quote convention, and FX volatility. |
| European spot-index option | Black-Scholes-Merton with dividend yield, or a forward model | Index dividend yield, settlement, and exercise convention. |
| Option on an index future | Black-76 | Futures price matching the option maturity. |
| American-style index option | Binomial tree or finite-difference model | Early exercise and contract-specific settlement rules. |

The exercise and settlement convention must be verified for each contract. For example, SPX options are European-style and cash-settled, while OEX options are American-style and XEO options are European-style.

## Equities and ETFs

For a European option with a continuous dividend yield, the Black-Scholes-Merton equations are:

$$
d_1 = \frac{\ln(S/K) + (r-q+\sigma^2/2)T}{\sigma\sqrt{T}},
\qquad
d_2 = d_1-\sigma\sqrt{T}
$$

$$
C = S e^{-qT}N(d_1)-K e^{-rT}N(d_2)
$$

$$
P = K e^{-rT}N(-d_2)-S e^{-qT}N(-d_1)
$$

where:

- $S$ is the underlying spot price;
- $K$ is the strike price;
- $T$ is the time to maturity in years;
- $r$ is the continuously compounded risk-free rate;
- $q$ is the continuous dividend yield;
- $\sigma$ is the annualized volatility;
- $N(\cdot)$ is the standard normal cumulative distribution function.

The main uncertainty is volatility. When no market chain exists, there is no observed implied volatility or volatility smile. Realized volatility, EWMA volatility, or a GARCH estimate may be used, but the resulting option value remains model-dependent.

For American-style equity and ETF options, especially dividend-paying puts and calls, a binomial tree or finite-difference method should replace the European closed-form model.

## Foreign exchange options

For an FX pair quoted as domestic currency per unit of foreign currency, the Garman-Kohlhagen equations are:

$$
d_1 = \frac{\ln(S/K) + (r_d-r_f+\sigma^2/2)T}{\sigma\sqrt{T}},
\qquad
d_2 = d_1-\sigma\sqrt{T}
$$

$$
C = S e^{-r_fT}N(d_1)-K e^{-r_dT}N(d_2)
$$

$$
P = K e^{-r_dT}N(-d_2)-S e^{-r_fT}N(-d_1)
$$

where $r_d$ is the domestic interest rate and $r_f$ is the foreign interest rate.

Every FX valuation must record:

- the quote convention, such as EUR/USD;
- the domestic and foreign currencies;
- both yield curves;
- business-day calendars and settlement rules;
- the volatility source;
- the market delta convention when Greeks are reported.

## Index and index-futures options

The exact contract must be identified before selecting a model. A spot-index option may require a dividend yield or forward level, while an option on an index future should normally use Black-76.

For an option on a future:

$$
d_1 = \frac{\ln(F/K)+\sigma^2T/2}{\sigma\sqrt{T}},
\qquad
d_2 = d_1-\sigma\sqrt{T}
$$

$$
C = e^{-rT}\left[F N(d_1)-K N(d_2)\right]
$$

$$
P = e^{-rT}\left[K N(-d_2)-F N(-d_1)\right]
$$

Here, $F$ is the futures price for the maturity corresponding to the option.

## Storage policy for synthetic values

Synthetic values should be stored in explicit model fields:

```text
option_data_source = MODEL
is_synthetic = 1
pricing_model = BSM | AMERICAN_TREE | GARMAN_KOHLHAGEN | BLACK76
volatility_source = REALIZED | EWMA | GARCH | EXTERNAL_SURFACE

market_bid = NULL
market_ask = NULL
market_volume = NULL
market_open_interest = NULL

model_price
model_delta
model_gamma
model_vega
model_price_low
model_price_high
```

A simple sensitivity band can be generated from three volatility scenarios:

$$
\sigma_{low}=0.75\,\sigma_{estimated},
\qquad
\sigma_{central}=\sigma_{estimated},
\qquad
\sigma_{high}=1.25\,\sigma_{estimated}
$$

This band expresses model uncertainty; it is not a market bid-ask spread.

## Operational rule

Synthetic pricing is acceptable as a separate valuation and stress-testing layer. It is not acceptable as a silent replacement for a missing market chain, and it cannot reconstruct an actual implied-volatility surface, bid-ask spread, volume, or open interest without market observations.

## License

Copyright (c) 2026 Karim Khemiri / Khem Kapital. All rights reserved.

These lessons are provided for personal, private, non-commercial educational use only. Reproduction, redistribution, resale, commercial use, paid course reuse, mirroring, scraping, sublicensing, and derivative distribution are not permitted without prior written authorization.

This repository is not open source and is not licensed under MIT.

## References

- Options Clearing Corporation, [Options 101: Equity Options Primer](https://www.theocc.com/getmedia/dfc83aa2-4a89-42d0-8de7-69e5a71f71a2/OCC-Primer-Options-101-EquityOptions-F.pdf).
- Garman, M. B., and Kohlhagen, S. W. (1983), [Foreign currency option values](https://www.sciencedirect.com/science/article/abs/pii/S0261560683800011).
- Black, F. (1976), [The pricing of commodity contracts](https://www.sciencedirect.com/science/article/pii/0304405X76900246/pdf).
- Cboe, [S&P 500 Index Options](https://www.cboe.com/tradable-products/sp-500/spx-options/).
- Cboe, [S&P 100 Index Options](https://www.cboe.com/tradable-products/sp-100/sp-100-index-options/).




