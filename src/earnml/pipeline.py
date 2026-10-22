"""End-to-end pipeline: python -m earnml.pipeline [--stage all|data|model|evaluate]."""
from __future__ import annotations

import argparse
import json
import logging

import pandas as pd

from . import config, evaluation, features, models, prices, sec

log = logging.getLogger("earnml")
MODELS = list(models.model_zoo())


def firm_data(ciks: list[int]) -> tuple[pd.DataFrame, pd.Series]:
    """Fundamentals and release times, merged across a firm's current and predecessor CIKs."""
    funds, rels = [], []
    for prio, cik in enumerate(ciks):
        f = sec.firm_fundamentals(sec.fetch_companyfacts(cik))
        if not f.empty:
            funds.append(f.assign(_prio=prio))
        rels.append(sec.earnings_release_times(sec.fetch_submissions(cik))["accepted_et"])
    if not funds:
        return pd.DataFrame(), pd.Series(dtype="datetime64[ns]")
    fund = (pd.concat(funds).sort_values(["fq", "_prio"]).drop_duplicates("fq")
            .drop(columns="_prio").reset_index(drop=True))
    rel = pd.concat(rels).drop_duplicates().sort_values().reset_index(drop=True)
    return fund, rel


def build_panel(tickers: list[str]) -> pd.DataFrame:
    cik_map = sec.ticker_to_cik()
    px = prices.download_prices(sorted(set(tickers) | {config.MARKET_TICKER}))
    rets = prices.daily_returns(px)
    mkt = rets[config.MARKET_TICKER]
    frames, coverage = [], []
    for t in tickers:
        cik = cik_map.get(t.replace(".", "-")) or cik_map.get(t)
        if cik is None or t not in px or px[t].notna().sum() < 300:
            coverage.append({"ticker": t, "status": "no CIK or prices"})
            continue
        try:
            fund, rel = firm_data([cik, *config.PREDECESSOR_CIKS.get(t, [])])
        except Exception as exc:  # network / parsing problems are logged, not fatal
            log.warning("%s: %s", t, exc)
            coverage.append({"ticker": t, "status": f"error: {exc}"})
            continue
        if fund.empty or rel.empty:
            coverage.append({"ticker": t, "status": "no XBRL net income or no 8-K 2.02"})
            continue
        p = features.fundamental_panel(fund, rel)
        p = features.add_market_data(p, rets[t], mkt, px[t])
        p.insert(0, "ticker", t)
        p.insert(1, "cik", cik)
        frames.append(p)
        coverage.append({"ticker": t, "status": "ok", "quarters": len(p),
                         "events": int(p["event_day"].notna().sum())})
        log.info("%-6s quarters=%d events=%d", t, len(p), p["event_day"].notna().sum())
    panel = pd.concat(frames, ignore_index=True)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(config.PROCESSED_DIR / "panel.parquet")
    pd.DataFrame(coverage).to_csv(config.RESULTS_DIR / "data_coverage.csv", index=False)
    return panel


def run_models(panel: pd.DataFrame) -> pd.DataFrame:
    preds = models.walk_forward(panel)
    preds.to_parquet(config.PROCESSED_DIR / "predictions.parquet")
    return preds


def evaluate(panel: pd.DataFrame, preds: pd.DataFrame) -> dict:
    from . import plots

    R = config.RESULTS_DIR
    acc = evaluation.forecast_accuracy(preds, MODELS)
    acc.to_csv(R / "forecast_accuracy.csv", index=False)

    preds = preds.assign(ml_surprise=preds["y_w"] - preds["pred_gbrt"])
    ls_rows = []
    for outcome in ("car_m1p1", "car_p2p21"):
        for m in MODELS:
            ls_rows.append({"model": m, **evaluation.long_short_test(preds, f"pred_{m}", outcome)})
        # Look-ahead sanity check: sorting on the *realized* surprise must produce
        # a large spread if the event dates and returns are wired up correctly.
        ls_rows.append({"model": "REALIZED (look-ahead)", **evaluation.long_short_test(preds, "y", outcome)})
        # Diagnostic: the part of the realized change the ML model did NOT anticipate.
        ls_rows.append({"model": "REALIZED minus GBRT forecast (look-ahead)",
                        **evaluation.long_short_test(preds, "ml_surprise", outcome)})
    ls = pd.DataFrame(ls_rows)
    ls.to_csv(R / "long_short_returns.csv", index=False)

    fm = []
    for outcome in ("car_m1p1", "car_p2p21"):
        for spec, regs in {
            "gbrt_only": ["pred_gbrt"],
            "gbrt_plus_controls": ["pred_gbrt", "de_l1", "ret_3m", "log_assets"],
            "ridge_plus_controls": ["pred_ridge", "de_l1", "ret_3m", "log_assets"],
        }.items():
            r = evaluation.fama_macbeth(preds, outcome, regs)
            r.insert(0, "spec", spec)
            r.insert(0, "outcome", outcome)
            fm.append(r)
    fm = pd.concat(fm, ignore_index=True)
    fm.to_csv(R / "fama_macbeth.csv", index=False)

    imp = models.feature_importance(panel, config.FIRST_TEST_YEAR)
    imp.rename("perm_importance_r2").to_csv(R / "feature_importance_gbrt.csv")

    summary = {
        "n_firms": int(preds["ticker"].nunique()),
        "n_oos_events": int(len(preds)),
        "oos_period": [str(preds["event_day"].min().date()), str(preds["event_day"].max().date())],
        "n_panel_rows": int(len(panel)),
        "forecast_accuracy": acc.round(4).to_dict(orient="records"),
        "long_short": ls.round(5).to_dict(orient="records"),
    }
    (R / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    plots.make_all(preds, acc, imp, MODELS)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", default="all", choices=["all", "data", "model", "evaluate"])
    ap.add_argument("--tickers", nargs="*", help="override the universe (for quick runs)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    tickers = args.tickers or config.load_universe()
    if args.stage in ("all", "data"):
        panel = build_panel(tickers)
    else:
        panel = pd.read_parquet(config.PROCESSED_DIR / "panel.parquet")
    if args.stage in ("all", "model"):
        preds = run_models(panel)
    elif args.stage == "evaluate":
        preds = pd.read_parquet(config.PROCESSED_DIR / "predictions.parquet")
    if args.stage in ("all", "evaluate"):
        s = evaluate(panel, preds)
        print(json.dumps({k: s[k] for k in ("n_firms", "n_oos_events", "oos_period")}, indent=2))


if __name__ == "__main__":
    main()
