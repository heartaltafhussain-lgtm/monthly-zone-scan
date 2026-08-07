#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monthly Demand/Supply Zone Scanner - NSE India
==============================================
Ye script aapke TradingView Pine indicator (v3.1) ki zone logic ko Python
mein port karti hai: daily NSE data ko monthly bars mein resample karke
ACTIVE monthly demand/supply zones nikalati hai, aur check karti hai ki
aaj (ya last N dinon mein) kaunse stocks ne apni monthly zone ko touch
kiya / sweep kiya / uske paas aaye.

Usage:
  python scanner.py                     # saare NSE EQ symbols (overnight job)
  python scanner.py --limit 50          # quick test - pehle 50 symbols
  python scanner.py --days 3            # last 3 trading days mein touch
  python scanner.py --symbols mylist.txt
  python scanner.py --type SUPPLY       # sirf supply zones
  python scanner.py --months 12         # sirf pichhle 12 months ke zones
  python scanner.py --fresh             # cached data ko force refresh karo

Outputs (reports/ folder mein):
  zone_report_YYYY-MM-DD.csv   - master table
  zone_report_YYYY-MM-DD.html  - browser mein dekhein (filter/search ke saath)
"""

import argparse
import logging
import os
import sys
import time
import math
from datetime import datetime, date
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yfinance as yf

logging.getLogger("yfinance").setLevel(logging.CRITICAL)

# Yahoo par NSE symbol ka naam alag ho sakta hai (rename/delisting ke baad)
YAHOO_ALIAS = {
    "ZOMATO": "ETERNAL",      # June 2025 rename
    "ONE97": "PAYTM",         # Paytm
    "BANDHANBANK": "BANDHANBNK",
}

HERE = os.path.dirname(os.path.abspath(__file__))
NSE_LIST = os.path.join(HERE, "nse_symbols.csv")
DATA_DIR = os.path.join(HERE, "data")
REPORT_DIR = os.path.join(HERE, "reports")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# ---------------- Zone logic settings (indicator v3.1 inputs ke barabar) ----------------
CFG = {
    "atr_len": 14,
    "base_threshold": 0.50,      # Base Candle Max Body (x ATR)
    "base_range_max": 1.20,      # Base Candle Max Range (x ATR)
    "base_body_pct_max": 50.0,   # Base Max Body % of Range
    "base_mode": "Normal",       # Strict / Normal / Loose
    "max_base": 3,               # Max Base Candles
    "allow_single_base": True,   # Allow single-candle pivot base
    "single_base_ratio": 0.60,   # Single base max size (x impulse body)
    "impulse_mult": 1.00,        # Impulse Min Body (x ATR)
    "require_break": True,       # Impulse must CLOSE beyond base
    "use_body_prox": True,       # Proximal = Body (else wick)
    "min_zone_atr": 0.10,
    "max_zone_atr": 2.00,
    "near_pct": 0.5,             # prox se % upar tak "NEAR" maano
}


# --------------------------------------------------------------------------------------
#  Monthly bars + ATR
# --------------------------------------------------------------------------------------
def to_monthly(df):
    """Daily OHLCV -> monthly bars (last close = close)."""
    d = df.copy()
    if getattr(d.index, "tz", None) is not None:
        d.index = d.index.tz_localize(None)
    mo = d.resample("ME").agg({"Open": "first", "High": "max", "Low": "min",
                               "Close": "last", "Volume": "sum"}).dropna()
    mo = mo[["Open", "High", "Low", "Close"]]
    prev_close = mo["Close"].shift(1)
    tr = pd.concat([mo["High"] - mo["Low"],
                    (mo["High"] - prev_close).abs(),
                    (mo["Low"] - prev_close).abs()], axis=1).max(axis=1)
    mo["ATR"] = tr.ewm(alpha=1.0 / CFG["atr_len"], adjust=False).mean()
    return mo


def is_base(bar, atr_t):
    """Pine f_isBase() - Normal mode default."""
    body = abs(bar["Close"] - bar["Open"])
    rng = bar["High"] - bar["Low"]
    body_pct = (body / rng * 100.0) if rng > 0 else 0.0
    small_body = body <= CFG["base_threshold"] * atr_t
    small_range = rng <= CFG["base_range_max"] * atr_t
    small_pct = body_pct <= CFG["base_body_pct_max"]
    if CFG["base_mode"] == "Strict":
        return small_body and small_range
    if CFG["base_mode"] == "Normal":
        return small_body and (small_range or small_pct)
    return small_body or small_pct


def overlap_ratio(p1, d1, p2, d2):
    lo1, hi1 = min(p1, d1), max(p1, d1)
    lo2, hi2 = min(p2, d2), max(p2, d2)
    ov = min(hi1, hi2) - max(lo1, lo2)
    if ov <= 0:
        return 0.0
    return ov / max(hi2 - lo2, 1e-9)


def find_zones(mo, max_months):
    """Monthly bars se demand/supply zones. Pine v3.1 logic."""
    zones = []
    lookback = max(2, int(max_months))
    n = len(mo)
    start = max(1, n - lookback)
    closes = mo["Close"].values

    for t in range(start, n):
        atr_t = mo["ATR"].iloc[t]
        row_t = mo.iloc[t]

        # base run (pichhle candles)
        cnt = 0
        i = t - 1
        while i >= 0 and cnt < CFG["max_base"]:
            if is_base(mo.iloc[i], atr_t):
                cnt += 1
                i -= 1
            else:
                break
        if cnt == 0 and CFG["allow_single_base"]:
            imp = abs(row_t["Close"] - row_t["Open"])
            prev = abs(mo["Close"].iloc[t - 1] - mo["Open"].iloc[t - 1])
            if imp > 0 and prev <= imp * CFG["single_base_ratio"]:
                cnt = 1
        if cnt == 0:
            continue

        sl = slice(t - cnt, t)
        w_hi = mo["High"].iloc[sl].max()
        w_lo = mo["Low"].iloc[sl].min()
        b_hi = mo[["Open", "Close"]].iloc[sl].max(axis=1).max()
        b_lo = mo[["Open", "Close"]].iloc[sl].min(axis=1).min()
        imp_body = abs(row_t["Close"] - row_t["Open"])

        bull = row_t["Close"] > row_t["Open"] and imp_body >= atr_t * CFG["impulse_mult"]
        bear = row_t["Close"] < row_t["Open"] and imp_body >= atr_t * CFG["impulse_mult"]

        if bull and (not CFG["require_break"] or row_t["Close"] > w_hi):
            prox = b_hi if CFG["use_body_prox"] else w_hi
            dist = w_lo
            h = prox - dist
            if h >= atr_t * CFG["min_zone_atr"] and h <= atr_t * CFG["max_zone_atr"]:
                dup = any(z["type"] == "DEMAND" and
                          overlap_ratio(prox, dist, z["prox"], z["dist"]) > 0.60
                          for z in zones)
                if not dup:
                    zones.append(dict(type="DEMAND", prox=prox, dist=dist, idx=t,
                                      date=mo.index[t], height=h))
        if bear and (not CFG["require_break"] or row_t["Close"] < w_lo):
            prox = b_lo if CFG["use_body_prox"] else w_lo
            dist = w_hi
            h = dist - prox
            if h >= atr_t * CFG["min_zone_atr"] and h <= atr_t * CFG["max_zone_atr"]:
                dup = any(z["type"] == "SUPPLY" and
                          overlap_ratio(prox, dist, z["prox"], z["dist"]) > 0.60
                          for z in zones)
                if not dup:
                    zones.append(dict(type="SUPPLY", prox=prox, dist=dist, idx=t,
                                      date=mo.index[t], height=h))

    # validity: demand invalid agar baad ka monthly close < distal (supply: > distal)
    for z in zones:
        later = closes[z["idx"] + 1:]
        z["valid"] = (not np.any(later < z["dist"])) if z["type"] == "DEMAND" \
            else (not np.any(later > z["dist"]))
    return zones


def classify_touch(z, lo, hi, near_pct):
    """Aaj ke bar ke relative status."""
    if z["type"] == "DEMAND":
        if z["dist"] <= lo <= z["prox"]:
            return "IN_ZONE"
        if lo < z["dist"]:
            return "SWEPT"
        gap = (lo - z["prox"]) / lo * 100.0
        return "NEAR" if gap <= near_pct else "FAR"
    else:
        if z["prox"] <= hi <= z["dist"]:
            return "IN_ZONE"
        if hi > z["dist"]:
            return "SWEPT"
        gap = (z["prox"] - hi) / hi * 100.0
        return "NEAR" if gap <= near_pct else "FAR"


# --------------------------------------------------------------------------------------
#  Data fetch (cache ke saath)
# --------------------------------------------------------------------------------------
def fetch_one(sym, period, fresh):
    """yfinance se daily data. Pehle NSE symbol try karo, phir alias, phir Ticker fallback."""
    cands = [sym + ".NS"]
    if sym in YAHOO_ALIAS:
        cands.append(YAHOO_ALIAS[sym] + ".NS")
    for cand in cands:
        path = os.path.join(DATA_DIR, sym + ".csv")
        if not fresh and os.path.exists(path):
            try:
                mtime = datetime.fromtimestamp(os.path.getmtime(path))
                if mtime.date() == date.today():
                    return sym, pd.read_csv(path, index_col=0, parse_dates=True)
            except Exception:
                pass
        for attempt in range(3):
            try:
                df = yf.download(cand, period=period, interval="1d",
                                 auto_adjust=True, progress=False)
                if df is not None and len(df) > 5:
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    df.to_csv(path)
                    return sym, df
            except Exception:
                pass
            try:
                df = yf.Ticker(cand).history(period=period, auto_adjust=True)
                if df is not None and len(df) > 5:
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    df.to_csv(path)
                    return sym, df
            except Exception:
                pass
            time.sleep(1.0 + attempt * 1.5)
    return sym, None


def load_symbols(path, limit):
    syms, names = [], {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        import csv
        r = csv.DictReader(f)
        for row in r:
            s = (row.get("Symbol") or "").strip()
            if not s:
                continue
            s = s.upper()
            if s.endswith(".NS"):
                s = s[:-3]
            syms.append(s)
            names[s] = (row.get("Name") or "").strip()
    if limit:
        syms = syms[:limit]
    return syms, names


# --------------------------------------------------------------------------------------
#  Report builders
# --------------------------------------------------------------------------------------
def build_html(rows, report_date, days, total_scanned):
    status_color = {"IN_ZONE": "#27ae60", "NEAR": "#f39c12", "SWEPT": "#8e44ad"}
    trs = []
    for r in rows:
        st = r["Status"]
        c = status_color.get(st, "#666")
        prox, dist = r["Proximal"], r["Distal"]
        zt = r["Type"]
        z_hi, z_lo = (prox, dist) if zt == "DEMAND" else (dist, prox)
        trs.append(
            "<tr><td>{sym}</td><td>{name}</td><td>{zt}</td>"
            "<td style='color:{c};font-weight:700'>{st}</td>"
            "<td>{formed}</td><td>{age}</td>"
            "<td>{ltp}</td><td>{lo}</td>"
            "<td style='text-align:right'>{z_hi:.2f}</td>"
            "<td style='text-align:right'>{z_lo:.2f}</td>"
            "<td style='text-align:right'>{gap:.2f}%</td></tr>".format(
                sym=r["Symbol"], name=r["Name"], zt=zt, st=st, c=c,
                formed=r["ZoneFormed"], age=r["ZoneAgeM"], ltp=f"{r['LTP']:.2f}",
                lo=f"{r['TodayLow']:.2f}", z_hi=z_hi, z_lo=z_lo, gap=r["GapPct"]))
    html = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>Monthly Zone Scan - {d}</title>
<style>
 body{{font-family:Segoe UI,Arial,sans-serif;background:#0e1117;color:#e6e6e6;margin:0;padding:24px}}
 h1{{font-size:20px;margin:0 0 4px}} .sub{{color:#9aa4b2;font-size:13px;margin-bottom:16px}}
 .card{{background:#161b24;border:1px solid #232b38;border-radius:10px;padding:14px 18px;margin-bottom:16px}}
 .grid{{display:flex;gap:24px;flex-wrap:wrap;font-size:13px}}
 .grid b{{font-size:24px;display:block}}
 input{{background:#0e1117;border:1px solid #2a3444;color:#eee;padding:8px 12px;border-radius:6px;width:280px;font-size:13px}}
 table{{border-collapse:collapse;width:100%;font-size:12.5px}}
 th{{background:#1b2230;text-align:left;padding:7px 8px;position:sticky;top:0;color:#c8d2e0;font-size:11px;text-transform:uppercase;letter-spacing:.4px}}
 td{{padding:6px 8px;border-bottom:1px solid #1d2532}}
 tr:hover td{{background:#182031}}
 .pill{{padding:2px 9px;border-radius:20px;color:#fff;font-size:11px;font-weight:700}}
</style></head><body>
<h1>📊 Monthly Demand / Supply Zone Touch Report</h1>
<div class="sub">NSE India • Report date: {d} • Last {days} trading day(s) • {total} symbols scanned</div>
<div class="card"><div class="grid">
 <div><b style="color:#27ae60">{n_in}</b>IN ZONE (aaj touch)</div>
 <div><b style="color:#f39c12">{n_near}</b>NEAR (0.5% andar)</div>
 <div><b style="color:#8e44ad">{n_swept}</b>SWEPT (below distal)</div>
 <div><b style="color:#9aa4b2">{total}</b>Symbols scanned</div>
</div></div>
<div class="card"><input id="q" placeholder="Filter: symbol / name / status..." onkeyup="f()"></div>
<div style="overflow:auto;max-height:70vh"><table id="t">
<thead><tr><th>Symbol</th><th>Name</th><th>Type</th><th>Status</th><th>Zone Formed</th><th>Age (M)</th><th>LTP</th><th>Today Low</th><th style="text-align:right">Zone Top</th><th style="text-align:right">Zone Bottom</th><th style="text-align:right">Dist from Zone</th></tr></thead>
<tbody>{rows}</tbody></table></div>
<script>
function f(){{var q=document.getElementById('q').value.toLowerCase();
document.querySelectorAll('#t tbody tr').forEach(function(r){{r.style.display=r.innerText.toLowerCase().indexOf(q)>-1?'':'none'}})}}
</script>
</body></html>""".format(
        d=report_date, days=days, total=total_scanned,
        n_in=sum(1 for r in rows if r["Status"] == "IN_ZONE"),
        n_near=sum(1 for r in rows if r["Status"] == "NEAR"),
        n_swept=sum(1 for r in rows if r["Status"] == "SWEPT"),
        rows="\n".join(trs))
    return html


# --------------------------------------------------------------------------------------
#  Main
# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Monthly D/S zone scanner (NSE)")
    ap.add_argument("--symbols", default=NSE_LIST, help="CSV: Symbol,Name")
    ap.add_argument("--symbols-list", default=None, help="TXT: sirf symbols")
    ap.add_argument("--limit", type=int, default=0, help="pehle N symbols")
    ap.add_argument("--days", type=int, default=1, help="kitne trading dinon mein touch check karna hai")
    ap.add_argument("--months", type=int, default=24, help="zone lookback (months)")
    ap.add_argument("--type", default="DEMAND", choices=["DEMAND", "SUPPLY", "BOTH"])
    ap.add_argument("--period", default="5y", help="yfinance data period")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--fresh", action="store_true", help="cache ignore karke naya data")
    ap.add_argument("--min-price", type=float, default=20.0, help="penny stocks skip")
    args = ap.parse_args()

    if args.symbols_list:
        syms = [s.strip().upper() for s in open(args.symbols_list) if s.strip()]
        syms = [s[:-3] if s.endswith(".NS") else s for s in syms]
        names = {}
    else:
        syms, names = load_symbols(args.symbols, args.limit)
        if args.limit:
            syms = syms[:args.limit]
    if not syms:
        print("❌ koi symbol nahi mila. --symbols file check karein.")
        sys.exit(1)

    print(f"📥 Fetching data for {len(syms)} symbols (workers={args.workers}) ...")
    t0 = time.time()
    data = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_one, s, args.period, args.fresh): s for s in syms}
        done = 0
        for fut in as_completed(futs):
            done += 1
            s, df = fut.result()
            data[s] = df
            if done % 100 == 0 or done == len(futs):
                el = time.time() - t0
                print(f"   ... {done}/{len(futs)} fetched ({el:.0f}s)")
    print(f"✅ Data ready in {time.time()-t0:.0f}s ({sum(1 for v in data.values() if v is not None)} symbols with data)\n")

    report_date = date.today().isoformat()
    rows = []
    errors = 0
    for sym, df in data.items():
        if df is None or len(df) < 60:
            errors += 1
            continue
        last_close = float(df["Close"].iloc[-1])
        if last_close < args.min_price:
            continue
        try:
            mo = to_monthly(df)
            if len(mo) < 30:
                continue
            zones = find_zones(mo, args.months)
            if not zones:
                continue
            tail = df.tail(args.days)
            for z in zones:
                if z["type"] != args.type and args.type != "BOTH":
                    continue
                if not z["valid"]:
                    continue
                for _, bar in tail.iterrows():
                    st = classify_touch(z, float(bar["Low"]), float(bar["High"]),
                                        CFG["near_pct"])
                    if st == "FAR":
                        continue
                    if z["type"] == "DEMAND":
                        gap = max(0.0, (float(bar["Low"]) - z["prox"]) / float(bar["Low"]) * 100.0)
                    else:
                        gap = max(0.0, (z["prox"] - float(bar["High"])) / float(bar["High"]) * 100.0)
                    age = (mo.index[-1].year - z["date"].year) * 12 + \
                          (mo.index[-1].month - z["date"].month)
                    rows.append({
                        "Symbol": sym, "Name": names.get(sym, ""),
                        "Type": z["type"], "Status": st,
                        "ZoneFormed": z["date"].strftime("%Y-%m"),
                        "ZoneAgeM": age,
                        "LTP": last_close,
                        "TodayLow": float(bar["Low"]),
                        "Proximal": z["prox"], "Distal": z["dist"],
                        "GapPct": gap,
                    })
        except Exception:
            errors += 1
            continue

    # dedupe: same symbol+zone ke liye best status rakho (IN_ZONE > NEAR > SWEPT)
    prio = {"IN_ZONE": 0, "NEAR": 1, "SWEPT": 2}
    best = {}
    for r in rows:
        k = (r["Symbol"], r["Type"], r["ZoneFormed"])
        if k not in best or prio[r["Status"]] < prio[best[k]["Status"]]:
            best[k] = r
    uniq = sorted(best.values(), key=lambda x: (prio[x["Status"]], x["GapPct"]))

    csv_path = os.path.join(REPORT_DIR, f"zone_report_{report_date}.csv")
    html_path = os.path.join(REPORT_DIR, f"zone_report_{report_date}.html")
    wl_path = os.path.join(REPORT_DIR, f"zone_watchlist_{report_date}.txt")
    if uniq:
        pd.DataFrame(uniq).to_csv(csv_path, index=False)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(build_html(uniq, report_date, args.days, len(syms)))
        # TradingView watchlist format: NSE:SYMBOL (aaj ke hits)
        with open(wl_path, "w", encoding="utf-8") as f:
            for r in uniq:
                f.write("NSE:" + r["Symbol"] + "\n")
    else:
        with open(csv_path, "w") as f:
            f.write("Symbol,Name,Type,Status,ZoneFormed,ZoneAgeM,LTP,TodayLow,Proximal,Distal,GapPct\n")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(build_html([], report_date, args.days, len(syms)))
        with open(wl_path, "w", encoding="utf-8") as f:
            f.write("")

    print(f"📄 Report: {csv_path}")
    print(f"📄 Report: {html_path}")
    print(f"📄 Watchlist (TradingView NSE: format): {wl_path}")
    print(f"\nSummary ({report_date}, last {args.days} day(s)):")
    for st in ["IN_ZONE", "NEAR", "SWEPT"]:
        n = sum(1 for r in uniq if r["Status"] == st)
        print(f"   {st:8s}: {n}")
    if uniq:
        print("\nTop picks (IN ZONE - aaj monthly demand zone touch):")
        for r in [x for x in uniq if x["Status"] == "IN_ZONE"][:20]:
            nm = f" ({r['Name'][:28]})" if r["Name"] else ""
            print(f"   {r['Symbol']:12s} {r['Type']:7s} zone {r['ZoneFormed']}  "
                  f"zone {r['Proximal']:.2f}-{r['Distal']:.2f}  LTP {r['LTP']:.2f}{nm}")


if __name__ == "__main__":
    main()
