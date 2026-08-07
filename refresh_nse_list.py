#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NSE ki official symbol list refresh karta hai (EQUITY_L.csv -> nse_symbols.csv).
Fail ho to purani list chalti rahegi (bundled nse_symbols.csv)."""
import csv
import os
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
URL = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
OUT = os.path.join(HERE, "nse_symbols.csv")

try:
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    data = urllib.request.urlopen(req, timeout=30).read().decode("utf-8-sig")
    lines = data.splitlines()
    reader = csv.DictReader(lines)
    rows = []
    for row in reader:
        sym = (row.get("SYMBOL") or "").strip()
        series = (row.get(" SERIES") or row.get("SERIES") or "").strip()
        name = (row.get("NAME OF COMPANY") or "").strip()
        if series == "EQ" and sym:
            rows.append((sym, name))
    if len(rows) > 1500:
        with open(OUT, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Symbol", "Name"])
            w.writerows(rows)
        print(f"✅ NSE list refreshed: {len(rows)} symbols")
    else:
        print(f"⚠️ Sirf {len(rows)} symbols mile, purani list rakhi")
except Exception as e:
    print(f"⚠️ NSE list download fail ({e}), bundled list use hogi")
