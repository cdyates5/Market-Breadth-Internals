#!/usr/bin/env python3
"""
S&P 500 Breadth & Internals — weekly dashboard (Acheron Insights)
Regenerates sp500_internals_weekly.html from an empty directory.

Pipeline
  1. Wikipedia: current constituents + "Historical components of the S&P 500" change table
     (table moved off the main list page on 11-Aug-2026).
  2. Rebuild point-in-time membership backward from today's list (survivorship-bias-free membership).
  3. Yahoo Finance daily split-adjusted closes for every ever-member since 2015 (+ ^GSPC, ^VIX).
  4. Daily breadth on members only: % > 50DMA, % > 200DMA, ratio-adjusted McClellan Summation.
  5. Sample to weekly (last session of each W-FRI week); 14-wk Wilder RSI and 40-wk MA on weekly SPX.
  6. Forward-return stress test (quintiles, 13-wk fwd return and fwd max drawdown) + episode table.
  7. Inject JSON into the embedded HTML template.

Usage:  python3 sp500_internals_rebuild.py [--cache DIR]   (cache reuses downloaded prices if present)
Deps:   pandas numpy requests lxml yfinance
"""
import io, re, sys, json, time, argparse, datetime as dt
import numpy as np, pandas as pd, requests

UA = {'User-Agent': 'Mozilla/5.0 (Acheron Insights research; breadth dashboard)'}
URL_CUR = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
URL_HIST = 'https://en.wikipedia.org/wiki/Historical_components_of_the_S%26P_500'
PRICE_START = '2014-06-01'           # warm-up for 200DMA / 40wMA / RSI
MEMBER_START = pd.Timestamp('2015-01-01')
DISPLAY_START = pd.Timestamp('2016-01-01')
OUT = 'sp500_internals_weekly.html'

# Ticker hygiene -------------------------------------------------------------
ALIAS = {'SATS': 'ECHO', 'RE': 'EG'}            # change-table ticker -> ticker used in current list
MANUAL_CHANGES = [('2024-07-08', 'SW', 'WRK')]  # Smurfit Westrock for WestRock (missing from table)
# Renamed companies whose continuous history lives under a new Yahoo symbol
YAHOO_RENAME = {'GPS': 'GAP', 'FBHS': 'FBIN', 'FLT': 'CPAY', 'WLTW': 'WTW', 'HFC': 'DINO',
                'ADS': 'BFH', 'ARNC': 'HWM', 'DWDP': 'DD', 'DISCA': 'WBD'}


def clean_tk(s):
    if not isinstance(s, str):
        return None
    s = re.sub(r'[^A-Z\.]', '', s.upper())
    return ALIAS.get(s, s) or None


def fetch_wiki():
    cur = pd.read_html(io.StringIO(requests.get(URL_CUR, headers=UA, timeout=60).text))[0]
    hist = pd.read_html(io.StringIO(requests.get(URL_HIST, headers=UA, timeout=60).text))[0]
    hist = hist.iloc[:, :6]
    hist.columns = ['date', 'add_t', 'add_n', 'rem_t', 'rem_n', 'reason']
    changes = [(pd.Timestamp(d), clean_tk(a), clean_tk(r)) for d, a, r in zip(hist.date, hist.add_t, hist.rem_t)]
    changes += [(pd.Timestamp(d), a, r) for d, a, r in MANUAL_CHANGES]
    return [clean_tk(s) for s in cur['Symbol']], changes


def membership(cur_syms, changes, dates):
    ch = sorted([c for c in changes if c[0] >= MEMBER_START], key=lambda c: c[0], reverse=True)
    mem = set(cur_syms)
    for _, a, r in ch:              # walk back: undo each change
        if a: mem.discard(a)
        if r: mem.add(r)
    cols = sorted(mem | {c[1] for c in ch if c[1]} | set(cur_syms))
    ch.sort(key=lambda c: c[0])
    live, k, rows = set(mem), 0, []
    for d in dates:                 # walk forward: apply changes effective on/before d
        while k < len(ch) and ch[k][0] <= d:
            _, a, r = ch[k]
            if r: live.discard(r)
            if a: live.add(a)
            k += 1
        rows.append([c in live for c in cols])
    return pd.DataFrame(rows, index=dates, columns=cols)


def fetch_prices(tickers):
    import yfinance as yf
    ymap = {t: t.replace('.', '-') for t in tickers}
    syms = sorted(set(ymap.values())) + ['^GSPC', '^VIX']
    frames = []
    for i in range(0, len(syms), 100):
        chunk = syms[i:i + 100]
        for attempt in range(3):
            try:
                d = yf.download(chunk, start=PRICE_START, auto_adjust=False, actions=False,
                                progress=False, threads=True, group_by='column')
                frames.append(d['Close'])
                break
            except Exception as e:
                print('  retry', i, e); time.sleep(5)
    px = pd.concat(frames, axis=1, sort=True)
    px = px.loc[:, ~px.columns.duplicated()]
    for old, new in YAHOO_RENAME.items():
        if old in ymap and px.get(ymap[old], pd.Series(dtype=float)).notna().sum() == 0:
            try:
                d = yf.download(new, start=PRICE_START, auto_adjust=False, actions=False, progress=False)
                c = d['Close']; c = c.iloc[:, 0] if isinstance(c, pd.DataFrame) else c
                px[ymap[old]] = c.reindex(px.index)
            except Exception:
                pass
    inv = {v: k for k, v in ymap.items()}
    return px.rename(columns=lambda c: inv.get(c, c))


def rsi_wilder(s, n=14):
    d = s.diff(); up = d.clip(lower=0); dn = -d.clip(upper=0)
    au = up.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    ad = dn.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + au / ad)


def compute(px, cur_syms, changes):
    spx = px['^GSPC'].dropna()
    px = px.loc[spx.index]
    dates = px.index
    vix = px['^VIX'].ffill(limit=2)
    mask = membership(cur_syms, changes, dates)
    P = px.reindex(columns=mask.columns)
    Pf = P.ffill(limit=3)                                   # bridge isolated missing prints only
    M = mask.values
    have = Pf.notna().values & M
    out = {}
    for n in (50, 200):
        ma = Pf.rolling(n, min_periods=n).mean()
        valid = ma.notna().values & have
        above = (Pf > ma).values & valid
        with np.errstate(invalid='ignore', divide='ignore'):
            out[f'p{n}'] = pd.Series(above.sum(1) / valid.sum(1) * 100, index=dates)
    chg = P.diff()                                          # genuine consecutive prints only
    adv = ((chg > 0).values & M).sum(1); dec = ((chg < 0).values & M).sum(1)
    with np.errstate(invalid='ignore', divide='ignore'):
        rana = pd.Series(np.where(adv + dec > 0, (adv - dec) / (adv + dec) * 1000, 0.0), index=dates)
    r0 = pd.concat([pd.Series([0.0], index=[dates[0] - pd.Timedelta(days=1)]), rana])  # zero-seeded EMAs
    e19 = r0.ewm(alpha=0.10, adjust=False).mean().iloc[1:]
    e39 = r0.ewm(alpha=0.05, adjust=False).mean().iloc[1:]
    summ = (e19 - e39).cumsum()
    daily = pd.DataFrame({'spx': spx, 'vix': vix, 'p50': out['p50'], 'p200': out['p200'], 'summ': summ,
                          'cov': pd.Series(have.sum(1) / M.sum(1) * 100, index=dates),
                          'nmem': pd.Series(M.sum(1), index=dates)})
    wk = daily.groupby(daily.index.to_period('W-FRI')).tail(1).copy()
    wk['ma40'] = wk.spx.rolling(40, min_periods=40).mean()
    wk['rel40'] = (wk.spx / wk.ma40 - 1) * 100
    wk['rsi'] = rsi_wilder(wk.spx, 14)
    wk['dd52'] = (wk.spx / wk.spx.rolling(52, min_periods=40).max() - 1) * 100
    s = wk.spx.values; n = len(s)
    wk['f13'] = (wk.spx.shift(-13) / wk.spx - 1) * 100
    wk['f26'] = (wk.spx.shift(-26) / wk.spx - 1) * 100
    wk['mdd13'] = [min(0.0, (s[i + 1:i + 14].min() / s[i] - 1) * 100) if i + 13 < n else np.nan for i in range(n)]
    ident_err = float((summ - (20 * e39.shift(-1) - 10 * e19.shift(-1))).abs().loc['2016':].iloc[:-1].max())
    return daily, wk, mask, ident_err, int((P.notna().any()).sum()), int(len(mask.columns))


def stress(wk):
    w = wk.loc[DISPLAY_START:]
    gauges = [('p200', '% above 200-day'), ('p50', '% above 50-day'), ('summ', 'McClellan Summation'),
              ('vix', 'VIX'), ('rsi', '14-week RSI'), ('rel40', 'vs 40-week MA')]
    res = []
    for k, lab in gauges:
        x = w[[k, 'f13', 'f26', 'mdd13']].dropna()
        q = pd.qcut(x[k], 5, labels=False)
        rows = []
        for i in range(5):
            s = x[q == i]
            rows.append({'lo': round(float(s[k].min()), 1), 'hi': round(float(s[k].max()), 1), 'n': int(len(s)),
                         'f13': round(float(s.f13.mean()), 2), 'hit13': round(float((s.f13 > 0).mean() * 100)),
                         'f26': round(float(s.f26.mean()), 2), 'mdd13': round(float(s.mdd13.mean()), 2)})
        res.append({'key': k, 'label': lab, 'q': rows})
    x = w.dropna(subset=['f13'])
    unc = {'f13': round(float(x.f13.mean()), 2), 'hit13': round(float((x.f13 > 0).mean() * 100)),
           'f26': round(float(w.f26.mean()), 2), 'mdd13': round(float(w.mdd13.mean()), 2), 'n': int(len(x))}
    cond = w[(w.dd52 > -3) & (w.p50 < 40)]
    eps, last = [], None
    for d in cond.index:
        if last is None or (d - last).days > 91:
            r = w.loc[d]
            eps.append({'d': d.strftime('%Y-%m-%d'), 'spx': round(float(r.spx), 1), 'dd': round(float(r.dd52), 1),
                        'p50': round(float(r.p50), 1), 'p200': round(float(r.p200), 1), 'summ': round(float(r.summ)),
                        'f13': None if pd.isna(r.f13) else round(float(r.f13), 1),
                        'f26': None if pd.isna(r.f26) else round(float(r.f26), 1),
                        'mdd13': None if pd.isna(r.mdd13) else round(float(r.mdd13), 1)})
        last = d
    return {'quint': res, 'uncond': unc, 'episodes': eps}


def validation(daily, wk):
    out = []
    for lab, a, b in [('Feb-2016 low', '2016-01-15', '2016-02-29'), ('Dec-2018 low', '2018-12-10', '2018-12-31'),
                      ('COVID low', '2020-03-09', '2020-03-31'), ('Oct-2022 low', '2022-09-20', '2022-10-20'),
                      ('Oct-2023 low', '2023-10-16', '2023-11-03'), ('Apr-2025 tariff low', '2025-03-31', '2025-04-15')]:
        x = daily.loc[a:b]; y = wk.loc[a:b]
        out.append({'ep': lab, 'd': x.p200.idxmin().strftime('%Y-%m-%d'), 'p200d': round(float(x.p200.min()), 1),
                    'p200w': round(float(y.p200.min()), 1), 'p50d': round(float(x.p50.min()), 1),
                    'vix': round(float(x.vix.max()), 1), 'cov': round(float(x['cov'].mean()), 1)})
    return out


def r2(s, nd=2):
    return [None if pd.isna(v) else round(float(v), nd) for v in s]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--cache', default=None); a = ap.parse_args()
    print('1/5 Wikipedia membership')
    cur_syms, changes = fetch_wiki()
    ever = sorted(set(cur_syms) | {t for c in changes if c[0] >= MEMBER_START for t in c[1:] if t})
    print('   current', len(cur_syms), 'ever-members since 2015', len(ever))
    print('2/5 Yahoo prices')
    px = None
    if a.cache:
        import os; os.makedirs(a.cache, exist_ok=True)
        fp = os.path.join(a.cache, 'close.pkl')
        if os.path.exists(fp):
            px = pd.read_pickle(fp)
            px.columns = [c if c.startswith('^') else c.replace('-', '.') for c in px.columns]
    if px is None:
        px = fetch_prices(ever)
        if a.cache: px.to_pickle(fp)
    print('3/5 Breadth + weekly panel')
    daily, wk, mask, ident_err, n_found, n_ever = compute(px, cur_syms, changes)
    print('   price series found', n_found, '/', n_ever, '| summation identity max err', round(ident_err, 3))
    print('4/5 Stress test')
    st = stress(wk); val = validation(daily, wk)
    w = wk.loc[DISPLAY_START:]
    last = w.index[-1]
    partial = last.weekday() < 4 and (pd.Timestamp.today().normalize() - last).days < 7
    data = {
        'meta': {'generated': dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M UTC'),
                 'last': last.strftime('%Y-%m-%d'), 'partial': bool(partial),
                 'n_ever': n_ever, 'n_found': n_found, 'n_changes': int(sum(1 for c in changes if c[0] >= MEMBER_START)),
                 'cov_now': round(float(w['cov'].iloc[-1]), 1), 'cov_min': round(float(w['cov'].min()), 1),
                 'cov_min_d': w['cov'].idxmin().strftime('%Y-%m-%d'), 'nmem': int(w.nmem.iloc[-1]),
                 'ident_err': round(ident_err, 3), 'weeks': int(len(w))},
        'w': {'d': [d.strftime('%Y-%m-%d') for d in w.index], 'spx': r2(w.spx), 'ma40': r2(w.ma40),
              'rel40': r2(w.rel40), 'rsi': r2(w.rsi), 'p200': r2(w.p200), 'p50': r2(w.p50),
              'summ': r2(w.summ, 1), 'vix': r2(w.vix), 'cov': r2(w['cov'], 1), 'dd52': r2(w.dd52)},
        'stress': st, 'validation': val,
    }
    print('5/5 Inject')
    payload = json.dumps(data, separators=(',', ':'))
    html, n = re.subn(r'/\*__DATA__\*/.*?/\*__END__\*/', lambda m: '/*__DATA__*/' + payload + '/*__END__*/',
                      TEMPLATE, flags=re.S)
    assert n == 1, 'data placeholder not found'
    open(OUT, 'w', encoding='utf-8').write(html)
    print('   wrote', OUT, f'{len(html)/1024:.0f} KB | last week {data["meta"]["last"]} partial={partial}')


TEMPLATE = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>S&amp;P 500 Breadth &amp; Internals | Acheron Insights</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
:root{
  --paper:#F7F1E6; --card:#FDFAF3; --ink:#221C14; --muted:#6B6153; --rule:#E4DACA; --rule-strong:#CFC2AD;
  --orange:#D2622A; --teal:#0E756C; --ochre:#A67A22; --plum:#7A4A66; --slate:#4E6577;
  --teal-soft:#0E756C1F; --orange-soft:#D2622A1F;
  --f-head:'Space Grotesk', 'Helvetica Neue', Arial, sans-serif;
  --f-body:'IBM Plex Sans', 'Helvetica Neue', Arial, sans-serif;
  --f-mono:'IBM Plex Mono', ui-monospace, 'SFMono-Regular', Menlo, Consolas, monospace;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --paper:#17130E; --card:#201B15; --ink:#EDE4D3; --muted:#A89C89; --rule:#352D23; --rule-strong:#4A4033;
    --orange:#E8804D; --teal:#3FAE9F; --ochre:#CFA04A; --plum:#B888A6; --slate:#91A7BA;
    --teal-soft:#3FAE9F26; --orange-soft:#E8804D26;
  }
}
:root[data-theme="dark"]{
  --paper:#17130E; --card:#201B15; --ink:#EDE4D3; --muted:#A89C89; --rule:#352D23; --rule-strong:#4A4033;
  --orange:#E8804D; --teal:#3FAE9F; --ochre:#CFA04A; --plum:#B888A6; --slate:#91A7BA;
  --teal-soft:#3FAE9F26; --orange-soft:#E8804D26;
}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.55 var(--f-body)}
.wrap{max-width:1180px;margin:0 auto;padding:28px 24px 64px}
a{color:inherit}
h1,h2,h3{font-family:var(--f-head);font-weight:600;letter-spacing:-0.01em;margin:0}
h1{font-size:clamp(28px,4vw,40px);line-height:1.08;font-weight:700}
h2{font-size:22px;line-height:1.2}
h3{font-size:15px;line-height:1.3}
p{margin:0 0 12px;max-width:72ch}
.num,.mono{font-family:var(--f-mono);font-variant-numeric:tabular-nums}
.muted{color:var(--muted)}
button{font:inherit;color:inherit}
:focus-visible{outline:2px solid var(--orange);outline-offset:2px}

.top{display:grid;grid-template-columns:1fr auto;gap:12px 32px;align-items:end;padding-bottom:18px;border-bottom:2px solid var(--ink)}
.top .firm{font-family:var(--f-head);font-weight:500;font-size:14px;color:var(--muted);margin-bottom:6px}
.asof{text-align:right;font-size:13px;color:var(--muted);line-height:1.5}
.asof b{color:var(--ink);font-weight:500}

.summary{display:grid;grid-template-columns:minmax(0,5fr) minmax(0,7fr);gap:32px;margin:26px 0 34px}
.read h2{margin-bottom:10px}
.read p{font-size:15.5px}
.tbl-scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
table{border-collapse:collapse;width:100%}
.readings th{font:500 12px/1.3 var(--f-body);color:var(--muted);text-align:right;padding:0 10px 8px;border-bottom:1px solid var(--rule-strong);white-space:nowrap}
.readings th:first-child{text-align:left;padding-left:0}
.readings td{padding:9px 10px;border-bottom:1px solid var(--rule);text-align:right;white-space:nowrap;font-size:14px}
.readings td:first-child{text-align:left;padding-left:0;white-space:normal;min-width:150px}
.readings td:first-child a{text-decoration:none;border-bottom:1px solid var(--rule-strong)}
.readings td:first-child a:hover{border-color:var(--ink)}
.swatch{display:inline-block;width:10px;height:3px;border-radius:1px;vertical-align:middle;margin-right:8px;position:relative;top:-1px}
.zone{font-size:13px;font-weight:500}
.z-stress{color:var(--orange)} .z-weak{color:var(--ochre)} .z-ok{color:var(--muted)} .z-strong{color:var(--teal)} .z-hot{color:var(--plum)}
.pbar{display:inline-block;width:54px;height:6px;background:var(--rule);border-radius:3px;vertical-align:middle;margin-left:8px;position:relative;overflow:hidden}
.pbar i{position:absolute;left:0;top:0;bottom:0;background:var(--ink);opacity:.55}

.stack{background:var(--card);border:1px solid var(--rule-strong);border-radius:6px}
.stack-head{position:sticky;top:0;z-index:5;background:var(--card);border-radius:6px 6px 0 0;border-bottom:1px solid var(--rule-strong)}
.tabs{display:flex;gap:26px;padding:0 16px;border-bottom:1px solid var(--rule);overflow-x:auto;scrollbar-width:none}
.tabs::-webkit-scrollbar{display:none}
.tabs button{background:none;border:0;border-bottom:2px solid transparent;margin-bottom:-1px;padding:13px 0 10px;font:500 15px/1.2 var(--f-head);color:var(--muted);cursor:pointer;white-space:nowrap}
.tabs button:hover{color:var(--ink)}
.tabs button[aria-selected="true"]{color:var(--ink);border-bottom-color:var(--orange)}
.tabs .sm{display:none}
.tabs button .ct{font:400 12px var(--f-mono);color:var(--muted);margin-left:6px}
.bar{display:flex;flex-wrap:wrap;gap:10px 22px;align-items:center;justify-content:space-between;padding:12px 16px}
.tab-note{padding:14px 16px 10px 16px;border-bottom:1px solid var(--rule)}
.tab-note p{margin:0;font-size:14.5px;max-width:92ch}
.tab-note b{font-weight:600}
.bar .when{font-family:var(--f-head);font-weight:600;font-size:17px;min-width:15ch}
.bar .when small{font-family:var(--f-body);font-weight:400;font-size:12.5px;color:var(--muted);margin-left:6px}
.ctrls{display:flex;flex-wrap:wrap;gap:8px 18px;align-items:center}
.seg{display:inline-flex;border:1px solid var(--rule-strong);border-radius:4px;overflow:hidden}
.seg button{background:transparent;border:0;padding:5px 11px;font-size:13px;cursor:pointer;color:var(--muted);border-right:1px solid var(--rule)}
.seg button:last-child{border-right:0}
.seg button[aria-pressed="true"]{background:var(--ink);color:var(--card)}
.ctrl-label{font-size:12.5px;color:var(--muted);margin-right:6px}
.panel{padding:10px 16px 2px 8px;border-bottom:1px solid var(--rule);scroll-margin-top:64px}
.panel:last-of-type{border-bottom:0}
.phead{display:flex;justify-content:space-between;align-items:baseline;gap:12px;padding-left:8px}
.phead h3{display:flex;align-items:baseline;gap:8px;flex-wrap:wrap}
.phead h3 .sub{font:400 12.5px var(--f-body);color:var(--muted)}
.pval{font-family:var(--f-mono);font-size:15px;font-weight:500;white-space:nowrap;text-align:right}
.pval .sec{font-size:12.5px;color:var(--muted);font-weight:400;margin-left:10px}
.cbox{position:relative;width:100%;touch-action:pan-y}
.stack-hint{font-size:12.5px;color:var(--muted);padding:8px 16px 12px;border-top:1px solid var(--rule)}

section{margin-top:46px}
.sec-head{display:grid;grid-template-columns:minmax(0,5fr) minmax(0,7fr);gap:32px;margin-bottom:18px;align-items:start}
.sec-head p{font-size:15px}
.heat th{font:500 12px/1.3 var(--f-body);color:var(--muted);padding:0 6px 8px;text-align:center;white-space:nowrap;border-bottom:1px solid var(--rule-strong)}
.heat th:first-child{text-align:left;padding-left:0}
.heat td{padding:3px;text-align:center}
.heat td:first-child{text-align:left;padding:8px 10px 8px 0;font-size:14px;white-space:nowrap}
.cell{border-radius:3px;padding:8px 4px 6px;font-family:var(--f-mono);font-size:14px;min-width:84px}
.cell small{display:block;font-size:10.5px;color:var(--muted);margin-top:2px;font-family:var(--f-mono)}
.heat tr.unc td{border-top:1px solid var(--rule-strong);padding-top:10px}
.card{background:var(--card);border:1px solid var(--rule-strong);border-radius:6px;padding:18px 18px 12px}
.eps th{font:500 12px/1.3 var(--f-body);color:var(--muted);text-align:right;padding:0 10px 8px;border-bottom:1px solid var(--rule-strong);white-space:nowrap}
.eps td{padding:8px 10px;text-align:right;border-bottom:1px solid var(--rule);white-space:nowrap;font-size:14px}
.eps th:first-child,.eps td:first-child{text-align:left;padding-left:0}
.eps tr.now td{background:var(--orange-soft)}
.pos{color:var(--teal)} .neg{color:var(--orange)}
.two{display:grid;grid-template-columns:minmax(0,5fr) minmax(0,7fr);gap:32px}
#valid td,#valid th{padding-left:7px;padding-right:7px;font-size:13px}
#valid th{font-size:11.5px}
.notes p{font-size:14.5px}
.notes h3{margin:18px 0 6px}
.notes h3:first-child{margin-top:0}
footer{margin-top:40px;padding-top:14px;border-top:1px solid var(--rule-strong);font-size:12.5px;color:var(--muted)}

@media (max-width:900px){
  .summary,.sec-head,.two{grid-template-columns:minmax(0,1fr);gap:20px}
  .summary>*,.sec-head>*,.two>*{min-width:0}
  .top{grid-template-columns:1fr}
  .asof{text-align:left}
}
@media (max-width:560px){
  .wrap{padding:18px 12px 48px}
  .panel{padding:8px 8px 2px 2px}
  .bar{padding:10px}
  .stack-head{position:static}
  .tabs{padding:0 10px;gap:18px}
  .tabs .lg{display:none} .tabs .sm{display:inline}
  .tabs button .ct{display:none}
  .tab-note{padding:12px 10px 8px}
  .phead h3 .sub{display:none}
  .readings td,.readings th{padding-left:6px;padding-right:6px}
  .pbar{display:none}
}
@media (prefers-reduced-motion: reduce){*{scroll-behavior:auto!important}}
</style>
</head>
<body>
<div class="wrap">

<header class="top">
  <div>
    <div class="firm">Acheron Insights, systematic macro</div>
    <h1>S&amp;P 500 breadth and internals</h1>
  </div>
  <div class="asof" id="asof"></div>
</header>

<div class="summary">
  <div class="read">
    <h2>Where internals stand</h2>
    <div id="readText"></div>
  </div>
  <div>
    <div class="tbl-scroll">
      <table class="readings" id="readings" aria-label="Latest weekly readings">
        <thead><tr><th>Gauge</th><th>Latest</th><th>4-wk change</th><th>Percentile since 2016</th><th>Zone</th></tr></thead>
        <tbody></tbody>
      </table>
    </div>
  </div>
</div>

<div class="stack" id="stack" tabindex="0" aria-label="Weekly chart stack. Use left and right arrow keys to move the week marker.">
  <div class="stack-head">
  <div class="tabs" id="tabs" role="tablist" aria-label="Chart views"></div>
  <div class="bar">
    <div class="when" id="when"></div>
    <div class="ctrls">
      <span><span class="ctrl-label">Range</span><span class="seg" id="segRange">
        <button data-v="52">1Y</button><button data-v="156">3Y</button><button data-v="260">5Y</button><button data-v="all" aria-pressed="true">All</button>
      </span></span>
      <span><span class="ctrl-label">S&amp;P scale</span><span class="seg" id="segScale">
        <button data-v="log" aria-pressed="true">Log</button><button data-v="linear">Linear</button>
      </span></span>
      <span><span class="ctrl-label">Shade weeks</span><span class="seg" id="segShade">
        <button data-v="below40" aria-pressed="true">Below 40-wk MA</button><button data-v="thin">Under 30% above 200-day</button><button data-v="none">Off</button>
      </span></span>
    </div>
  </div>
  </div>
  <div class="tab-note" id="tabNote" hidden></div>
  <div id="panels" role="tabpanel"></div>
  <div class="stack-hint">Hover or drag across any panel to move the week marker through every chart on the tab. Arrow keys step a week (Shift for a quarter) when the stack has focus.</div>
</div>

<section id="stress">
  <div class="sec-head">
    <div><h2>Do these gauges forecast anything?</h2></div>
    <div id="stressText"></div>
  </div>
  <div class="card">
    <div style="display:flex;flex-wrap:wrap;justify-content:space-between;gap:10px;align-items:center;margin-bottom:12px">
      <h3>S&amp;P 500 outcome after each weekly reading, by quintile of the gauge</h3>
      <span class="seg" id="segMetric">
        <button data-v="f13" aria-pressed="true">13-wk return</button><button data-v="f26">26-wk return</button><button data-v="hit13">13-wk hit rate</button><button data-v="mdd13">13-wk max drawdown</button>
      </span>
    </div>
    <div class="tbl-scroll">
      <table class="heat" id="heat"><thead></thead><tbody></tbody></table>
    </div>
    <p class="muted" style="font-size:12.5px;margin:12px 0 4px;max-width:none">Quintile 1 is the weakest internal reading for every gauge (for VIX, the highest level). Cells show the mean outcome; the small line gives the gauge's range in that quintile. Shading is the gap to the all-weeks average, teal better and orange worse. Quintile edges are set on the full sample, so this is descriptive, not a tradable rule.</p>
  </div>

  <div class="card" style="margin-top:22px">
    <h3 style="margin-bottom:4px">The current setup: index within 3% of its 52-week high, fewer than 40% of members above their 50-day MA</h3>
    <p class="muted" style="font-size:13px;max-width:none">First week of each episode since 2016 (episodes separated by more than 13 weeks).</p>
    <div class="tbl-scroll">
      <table class="eps" id="eps"><thead><tr><th>Week</th><th>S&amp;P 500</th><th>From 52-wk high</th><th>% &gt; 50-day</th><th>% &gt; 200-day</th><th>McClellan</th><th>13-wk return</th><th>13-wk max drawdown</th><th>26-wk return</th></tr></thead><tbody></tbody></table>
    </div>
  </div>
</section>

<section id="method">
  <div class="sec-head"><div><h2>How this is built</h2></div><div></div></div>
  <div class="two">
    <div class="notes" id="notes"></div>
    <div>
      <div class="card">
        <h3 style="margin-bottom:8px">Reconstruction check at known washouts</h3>
        <div class="tbl-scroll">
          <table class="eps" id="valid"><thead><tr><th>Episode</th><th>Daily low</th><th>% &gt; 200-day, daily</th><th>Weekly close</th><th>% &gt; 50-day, daily</th><th>VIX peak</th><th>Price coverage</th></tr></thead><tbody></tbody></table>
        </div>
        <p class="muted" style="font-size:12.5px;margin:10px 0 0;max-width:none">Weekly sampling misses intra-week extremes: the Dec-2018 trough printed on a Monday and the week-end reading is several points higher.</p>
      </div>
      <div class="card" style="margin-top:18px">
        <h3 style="margin-bottom:6px">Share of index members with Yahoo price history</h3>
        <div class="cbox" style="height:150px"><canvas id="covChart"></canvas></div>
      </div>
    </div>
  </div>
</section>

<footer id="foot"></footer>
</div>

<script>
const D = /*__DATA__*/null/*__END__*/;

(function(){
'use strict';
const W = D.w, N = W.d.length, M = D.meta;
const MONTHS=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
const fmtDate = s => { const [y,m,d]=s.split('-'); return `${+d} ${MONTHS[+m-1]} ${y}`; };
const fmtShort = s => { const [y,m]=s.split('-'); return `${MONTHS[+m-1]} ${y.slice(2)}`; };
const fmtMY = s => { const [y,m]=s.split('-'); return `${MONTHS[+m-1]} ${y}`; };
const f1 = v => v==null?'n/a':v.toFixed(1);
const sgn = (v,d=1) => v==null?'n/a':(v>0?'+':v<0?'\u2212':'')+Math.abs(v).toFixed(d);
const fmtPx = v => v==null?'n/a':v.toLocaleString('en-US',{minimumFractionDigits:1,maximumFractionDigits:1});

const PAL = {
  light:{ink:'#221C14',muted:'#6B6153',grid:'#221C1414',zero:'#221C1466',card:'#FDFAF3',orange:'#D2622A',teal:'#0E756C',ochre:'#A67A22',plum:'#7A4A66',slate:'#4E6577',tealA:'#0E756C33',orangeA:'#D2622A38',shade:'#D2622A17',cross:'#221C149E',ref:'#221C1440',clear:'#FDFAF300'},
  dark:{ink:'#EDE4D3',muted:'#A89C89',grid:'#EDE4D314',zero:'#EDE4D366',card:'#201B15',orange:'#E8804D',teal:'#3FAE9F',ochre:'#CFA04A',plum:'#B888A6',slate:'#91A7BA',tealA:'#3FAE9F38',orangeA:'#E8804D3D',shade:'#E8804D1F',cross:'#EDE4D3A6',ref:'#EDE4D340',clear:'#201B1500'}
};
const isDark = () => { const t=document.documentElement.getAttribute('data-theme'); if(t==='dark') return true; if(t==='light') return false; return !!(window.matchMedia && matchMedia('(prefers-color-scheme: dark)').matches); };
let C = isDark()?PAL.dark:PAL.light;

const PANELS = [
  {id:'spx', title:'S&amp;P 500', sub:'weekly close, 40-week MA dashed', h:250, log:true,
   series:[{k:'spx',c:'ink',w:1.6},{k:'ma40',c:'ochre',w:1.2,dash:[5,4]}],
   val:i=>fmtPx(W.spx[i]), sec:i=>W.ma40[i]==null?'':`40-wk MA ${fmtPx(W.ma40[i])}`},
  {id:'p200', title:'% of members above their 200-day MA', sub:'', h:140, fixed:[0,100], refs:[20,50,80],
   series:[{k:'p200',c:'teal',w:1.5}], val:i=>f1(W.p200[i])+'%'},
  {id:'p50', title:'% of members above their 50-day MA', sub:'', h:140, fixed:[0,100], refs:[20,50,80],
   series:[{k:'p50',c:'slate',w:1.5}], val:i=>f1(W.p50[i])+'%'},
  {id:'summ', title:'McClellan Summation Index', sub:'ratio-adjusted, members only', h:140, refs:[0], incZero:true, fillZero:true,
   series:[{k:'summ',c:'plum',w:1.5}], val:i=>sgn(W.summ[i],0)},
  {id:'vix', title:'VIX, inverted', sub:'axis reversed, stress plots low', h:140, reverse:true, refs:[20,30],
   series:[{k:'vix',c:'orange',w:1.4}], val:i=>f1(W.vix[i])},
  {id:'rsi', title:'14-week RSI', sub:'Wilder smoothing on weekly closes', h:140, refs:[30,50,70],
   series:[{k:'rsi',c:'ochre',w:1.5}], val:i=>f1(W.rsi[i])},
  {id:'rel40', title:'S&amp;P 500 vs its 40-week MA', sub:'% above or below', h:142, refs:[0], incZero:true, fillZero:true,
   series:[{k:'rel40',c:'ink',w:1.3}], val:i=>sgn(W.rel40[i],1)+'%'}
];

const PBYID = Object.fromEntries(PANELS.map(p=>[p.id,p]));
const TABS = [
  {id:'all', label:'All gauges', short:'All', panels:[['spx'],['p200'],['p50'],['summ'],['vix'],['rsi'],['rel40']]},
  {id:'breadth', label:'Price vs breadth', short:'Breadth', panels:[['spx',300],['p200',170],['p50',170],['summ',175]]},
  {id:'vix', label:'Price vs VIX', short:'VIX', panels:[['spx',330],['vix',280]]},
  {id:'mom', label:'Price vs momentum', short:'Momentum', panels:[['spx',310],['rsi',205],['rel40',215]]}
];
const HOME = {p200:'breadth', p50:'breadth', summ:'breadth', vix:'vix', rsi:'mom', rel40:'mom'};
const TAB_KEY = 'acheron-sp500-internals-tab';
let savedTab = 'all';
try { const v = localStorage.getItem(TAB_KEY); if(v && TABS.some(x=>x.id===v)) savedTab = v; } catch(e) {}
const st = { lo:0, hi:N-1, hover:null, scale:'log', shade:'below40', metric:'f13', tab:savedTab };
const shadeMask = {
  below40: W.rel40.map(v=>v!=null && v<0),
  thin: W.p200.map(v=>v!=null && v<30),
  none: W.d.map(()=>false)
};

const bands = {
  id:'bands',
  beforeDatasetsDraw(ch){
    const m = shadeMask[st.shade]; const {ctx, chartArea:a, scales:{x}} = ch;
    ctx.save(); ctx.fillStyle = C.shade;
    let i = Math.max(0, Math.floor(x.min));
    const end = Math.min(N-1, Math.ceil(x.max));
    while(i<=end){
      if(m[i]){ let j=i; while(j+1<=end && m[j+1]) j++;
        const x0=Math.max(a.left, x.getPixelForValue(i-0.5)), x1=Math.min(a.right, x.getPixelForValue(j+0.5));
        if(x1>x0) ctx.fillRect(x0, a.top, x1-x0, a.bottom-a.top); i=j+1; }
      else i++;
    }
    ctx.restore();
    const p = ch.$panel; if(!p || !p.refs) return;
    const y = ch.scales.y; ctx.save(); ctx.lineWidth=1;
    p.refs.forEach(r=>{
      const py = y.getPixelForValue(r); if(py<a.top-1 || py>a.bottom+1) return;
      ctx.strokeStyle = r===0?C.zero:C.ref; ctx.setLineDash(r===0?[]:[3,4]);
      ctx.beginPath(); ctx.moveTo(a.left,py); ctx.lineTo(a.right,py); ctx.stroke();
    });
    ctx.restore();
  },
  afterDraw(ch){
    if(st.hover==null) return;
    const {ctx, chartArea:a, scales:{x,y}} = ch; const px = x.getPixelForValue(st.hover);
    if(px<a.left-0.5||px>a.right+0.5) return;
    ctx.save(); ctx.strokeStyle=C.cross; ctx.lineWidth=1; ctx.setLineDash([]);
    ctx.beginPath(); ctx.moveTo(px,a.top); ctx.lineTo(px,a.bottom); ctx.stroke();
    const p = ch.$panel;
    if(p){ const v = W[p.series[0].k][st.hover];
      if(v!=null){ const py=y.getPixelForValue(v); ctx.fillStyle=C[p.series[0].c]; ctx.strokeStyle=C.card; ctx.lineWidth=2;
        ctx.beginPath(); ctx.arc(px,py,3.6,0,Math.PI*2); ctx.fill(); ctx.stroke(); } }
    ctx.restore();
  }
};

function xTicks(lo,hi){
  const span = hi-lo, out=[], narrow = window.innerWidth<640;
  const quarterly = span<=160;
  const stepY = (!quarterly && narrow && span>300) ? 2 : 1;
  for(let i=Math.max(1,Math.ceil(lo)); i<=Math.floor(hi); i++){
    const [y0,m0]=W.d[i-1].split('-'), [y1,m1]=W.d[i].split('-');
    if(quarterly){
      const q0=Math.floor((+m0-1)/3), q1=Math.floor((+m1-1)/3);
      if(y0!==y1 || q0!==q1){ if(!(narrow && span>60 && q1%2)) out.push({value:i}); }
    } else if(y0!==y1 && (+y1)%stepY===0) out.push({value:i});
  }
  return out;
}
function xLabel(v,lo,hi){
  const i=Math.round(v); if(i<0||i>=N) return '';
  return (hi-lo)<=160 ? fmtShort(W.d[i]) : W.d[i].slice(0,4);
}
function yRange(p){
  if(p.fixed) return {min:p.fixed[0], max:p.fixed[1]};
  let mn=Infinity,mx=-Infinity;
  const lo=Math.max(0,Math.floor(st.lo)), hi=Math.min(N-1,Math.ceil(st.hi));
  p.series.forEach(s=>{ const a=W[s.k]; for(let i=lo;i<=hi;i++){ const v=a[i]; if(v!=null){ if(v<mn)mn=v; if(v>mx)mx=v; } } });
  if(p.incZero){ mn=Math.min(mn,0); mx=Math.max(mx,0); }
  if(p.log && st.scale==='log'){ return {min:mn/1.04, max:mx*1.04}; }
  const pad=(mx-mn)*0.07||1; return {min:mn-pad, max:mx+pad};
}
function niceTicks(min,max,count){
  const raw=(max-min)/count; if(!(raw>0)) return [{value:min}];
  const mag=Math.pow(10,Math.floor(Math.log10(raw))), norm=raw/mag;
  const step=(norm<1.5?1:norm<3?2:norm<7?5:10)*mag, out=[];
  for(let v=Math.ceil(min/step)*step; v<=max+step*1e-6; v+=step) out.push({value:+v.toFixed(8)});
  return out;
}
function logTicks(min,max){
  const cand=[1000,1500,2000,2500,3000,4000,5000,6000,7000,8000,10000,12000];
  let t=cand.filter(v=>v>=min&&v<=max);
  while(t.length>6) t=t.filter((_,k)=>k%2===0);
  return t.map(v=>({value:v}));
}

const charts=[];
const monoFont = {family:"'IBM Plex Mono', ui-monospace, monospace", size:10.5};
function yTickFmt(p){
  return v => {
    if(p.log) return v>=1000 ? (v/1000).toLocaleString('en-US',{maximumFractionDigits:1})+'k' : v;
    if(p.fixed) return v+'%';
    if(Math.abs(v)>=1000) return (v/1000).toFixed(1)+'k';
    return +(+v).toFixed(1);
  };
}
function buildPanels(){
  charts.forEach(c=>c.destroy()); charts.length=0;
  const host=document.getElementById('panels'); host.innerHTML='';
  const narrow = window.innerWidth<560;
  const list = TABS.find(x=>x.id===st.tab).panels;
  list.forEach(([pid,hOverride],idx)=>{
    const p = PBYID[pid];
    const last = idx===list.length-1;
    const baseH = (hOverride||p.h) + (last?16:0);
    const el=document.createElement('div'); el.className='panel'; el.id='panel-'+p.id;
    el.innerHTML=`<div class="phead"><h3>${p.title}${p.sub?`<span class="sub">${p.sub}</span>`:''}</h3><div class="pval"><span data-v></span><span class="sec" data-s></span></div></div><div class="cbox" style="height:${Math.round(baseH*(narrow?0.82:1))}px"><canvas role="img" aria-label="${p.title.replace('&amp;','&')} weekly chart"></canvas></div>`;
    host.appendChild(el);
    const cv=el.querySelector('canvas');
    const ds=p.series.map(s=>({
      data:W[s.k].map((v,i)=>({x:i,y:v})), borderColor:C[s.c], borderWidth:s.w||1.4, borderDash:s.dash||[],
      pointRadius:0, pointHoverRadius:0, tension:0, spanGaps:false,
      fill: p.fillZero ? {target:{value:0}, above:C.tealA, below:C.orangeA} : false
    }));
    const r=yRange(p);
    const ch=new Chart(cv,{
      type:'line', data:{datasets:ds}, plugins:[bands],
      options:{
        animation:false, responsive:true, maintainAspectRatio:false, parsing:false, normalized:true, events:[],
        layout:{padding:{top:6,right:6,bottom:last?0:4}},
        plugins:{legend:{display:false}, tooltip:{enabled:false}},
        scales:{
          x:{type:'linear', min:st.lo, max:st.hi, border:{display:false},
             grid:{color:C.grid, drawTicks:last, tickLength:4},
             ticks:{display:last, color:C.muted, font:monoFont, autoSkip:false, maxRotation:0, callback:v=>xLabel(v,st.lo,st.hi)},
             afterBuildTicks:sc=>{ sc.ticks = xTicks(sc.min,sc.max); }},
          y:{type:(p.log&&st.scale==='log')?'logarithmic':'linear', reverse:!!p.reverse, min:r.min, max:r.max,
             border:{display:false}, grid:{color:C.grid, drawTicks:false},
             ticks:{color:C.muted, font:monoFont, padding:6, maxTicksLimit:12, includeBounds:false, callback:yTickFmt(p)},
             afterBuildTicks:sc=>{ if(p.log && st.scale==='log') sc.ticks=logTicks(sc.min,sc.max); else if(p.fixed) sc.ticks=[0,20,50,80,100].map(v=>({value:v})); else sc.ticks=niceTicks(sc.min,sc.max,p.log?5:4); },
             afterFit:sc=>{ sc.width = narrow?42:56; }}
        }
      }
    });
    ch.$panel=p; ch.$el=el;
    charts.push(ch);
    attachPointer(cv,ch);
  });
  refreshHeads();
}
function attachPointer(cv,ch){
  let raf=null;
  const move=e=>{
    const rect=cv.getBoundingClientRect(); const x=e.clientX-rect.left;
    const a=ch.chartArea; if(!a || x<a.left-4||x>a.right+4) return;
    let i=Math.round(ch.scales.x.getValueForPixel(Math.min(Math.max(x,a.left),a.right)));
    i=Math.max(Math.ceil(st.lo),Math.min(Math.floor(st.hi),i));
    if(i===st.hover) return; st.hover=i;
    if(!raf) raf=requestAnimationFrame(()=>{ raf=null; redrawHover(); });
  };
  cv.addEventListener('pointermove',move);
  cv.addEventListener('pointerdown',move);
  cv.addEventListener('pointerleave',e=>{ if(e.pointerType==='mouse'){ st.hover=null; redrawHover(); } });
}
function redrawHover(){ charts.forEach(c=>c.draw()); refreshHeads(); }
function refreshHeads(){
  const i = st.hover==null ? N-1 : st.hover;
  const isLast = i===N-1;
  document.getElementById('when').innerHTML = `Week to ${fmtDate(W.d[i])}${isLast&&M.partial?'<small>partial week</small>':(st.hover==null?'<small>latest</small>':'')}`;
  charts.forEach(c=>{ const p=c.$panel; c.$el.querySelector('[data-v]').textContent=p.val(i); c.$el.querySelector('[data-s]').textContent=p.sec?p.sec(i):''; });
}
function applyRange(){
  charts.forEach(c=>{
    const p=c.$panel; c.options.scales.x.min=st.lo; c.options.scales.x.max=st.hi;
    c.options.scales.y.type=(p.log&&st.scale==='log')?'logarithmic':'linear';
    const r=yRange(p); c.options.scales.y.min=r.min; c.options.scales.y.max=r.max;
    c.update('none');
  });
  refreshHeads();
}

function seg(id,fn){
  const el=document.getElementById(id);
  el.querySelectorAll('button').forEach(b=>{ if(!b.hasAttribute('aria-pressed')) b.setAttribute('aria-pressed','false'); });
  el.addEventListener('click',e=>{ const b=e.target.closest('button'); if(!b) return;
    el.querySelectorAll('button').forEach(x=>x.setAttribute('aria-pressed',x===b?'true':'false'));
    fn(b.dataset.v); });
}
seg('segRange',v=>{ st.lo = v==='all'?0:Math.max(0,N-1-(+v)); st.hi=N-1; if(st.hover!=null&&st.hover<st.lo) st.hover=null; applyRange(); });
seg('segScale',v=>{ st.scale=v; applyRange(); });
seg('segShade',v=>{ st.shade=v; charts.forEach(c=>c.draw()); });
seg('segMetric',v=>{ st.metric=v; renderHeat(); });
document.getElementById('stack').addEventListener('keydown',e=>{
  if(e.key!=='ArrowLeft'&&e.key!=='ArrowRight') return;
  if(e.target.closest && e.target.closest('button')) return;
  e.preventDefault();
  const cur = st.hover==null? N-1 : st.hover; const step = e.shiftKey?13:1;
  st.hover = Math.max(Math.ceil(st.lo), Math.min(N-1, cur + (e.key==='ArrowLeft'?-step:step)));
  redrawHover();
});

const pctile = (arr,v) => { let n=0,k=0; arr.forEach(x=>{ if(x!=null){ n++; if(x<=v) k++; } }); return n?k/n*100:null; };
const ZONES = {
  p200: v=> v<20?['Washout','z-stress']: v<50?['Narrow','z-weak']: v<80?['Healthy','z-ok']:['Very broad','z-strong'],
  p50:  v=> v<20?['Washout','z-stress']: v<50?['Narrow','z-weak']: v<80?['Healthy','z-ok']:['Very broad','z-strong'],
  summ: v=> v<-500?['Oversold','z-stress']: v<0?['Negative','z-weak']: v<1000?['Positive','z-ok']:['Overbought','z-hot'],
  vix:  v=> v>=30?['Stress','z-stress']: v>=20?['Elevated','z-weak']: v>=13?['Normal','z-ok']:['Complacent','z-hot'],
  rsi:  v=> v<30?['Oversold','z-stress']: v<50?['Weak','z-weak']: v<70?['Positive','z-ok']:['Overbought','z-hot'],
  rel40:v=> v<-10?['Deep below trend','z-stress']: v<0?['Below trend','z-weak']: v<10?['Above trend','z-ok']:['Extended','z-hot'],
  spx:  v=> { const d=W.dd52[N-1]; return d!=null && d>-0.01 ? ['At 52-wk high','z-strong'] : [`${sgn(d,1)}% from 52-wk high`, d<-10?'z-stress':'z-ok']; }
};
function renderReadings(){
  const i=N-1, j=Math.max(0,N-5);
  const rows = [
    ['spx','S&amp;P 500','ink', fmtPx(W.spx[i]), sgn((W.spx[i]/W.spx[j]-1)*100,1)+'%'],
    ['p200','% above 200-day MA','teal', f1(W.p200[i])+'%', sgn(W.p200[i]-W.p200[j],1)+' pp'],
    ['p50','% above 50-day MA','slate', f1(W.p50[i])+'%', sgn(W.p50[i]-W.p50[j],1)+' pp'],
    ['summ','McClellan Summation','plum', sgn(W.summ[i],0), sgn(W.summ[i]-W.summ[j],0)],
    ['vix','VIX','orange', f1(W.vix[i]), sgn(W.vix[i]-W.vix[j],1)],
    ['rsi','14-week RSI','ochre', f1(W.rsi[i]), sgn(W.rsi[i]-W.rsi[j],1)],
    ['rel40','vs 40-week MA','ink', sgn(W.rel40[i],1)+'%', sgn(W.rel40[i]-W.rel40[j],1)+' pp']
  ];
  document.querySelector('#readings tbody').innerHTML = rows.map(([k,lab,c,v,d])=>{
    const p=pctile(W[k],W[k][i]); const [zt,zc]=ZONES[k](W[k][i]);
    return `<tr><td><span class="swatch" style="background:var(--${c})"></span><a href="#panel-${k}" data-k="${k}">${lab}</a></td><td class="num">${v}</td><td class="num">${d}</td><td class="num">${p==null?'':Math.round(p)}<span class="pbar" aria-hidden="true"><i style="width:${p}%"></i></span></td><td><span class="zone ${zc}">${zt}</span></td></tr>`;
  }).join('');
}

function renderRead(){
  const i=N-1; const lb=Math.max(0,N-26);
  let pk=lb; for(let k=lb;k<=i;k++) if(W.spx[k]>W.spx[pk]) pk=k;
  const ddPk=(W.spx[i]/W.spx[pk]-1)*100;
  const allMax = Math.max(...W.spx.filter(v=>v!=null));
  const isRecord = W.spx[pk]===allMax;
  const d50=W.p50[i]-W.p50[pk];
  const wkLabel = `the week to ${fmtDate(W.d[i])}${M.partial?' (partial week)':''}`;
  const out=[];
  if(pk===i){
    out.push(`The S&amp;P 500 closed at ${fmtPx(W.spx[i])} in ${wkLabel}, its highest weekly close of the past six months${isRecord?' and a record':''}. ${f1(W.p50[i])}% of members are above their 50-day MA and ${f1(W.p200[i])}% above their 200-day; the McClellan Summation Index reads ${sgn(W.summ[i],0)}.`);
  } else {
    out.push(`The S&amp;P 500 closed at ${fmtPx(W.spx[i])} in ${wkLabel}, ${Math.abs(ddPk).toFixed(1)}% below its ${fmtDate(W.d[pk])} ${isRecord?'record weekly close':'six-month high'}.`);
    const verdict = d50<-15 ? 'Internals have weakened far faster than price.' : d50>15 ? 'Internals have improved even as price has slipped.' : 'Internals have broadly tracked price.';
    out.push(`${verdict} Since that high the share of members above their 50-day MA has gone from ${f1(W.p50[pk])}% to ${f1(W.p50[i])}%, above their 200-day from ${f1(W.p200[pk])}% to ${f1(W.p200[i])}%, and the McClellan Summation Index from ${sgn(W.summ[pk],0)} to ${sgn(W.summ[i],0)}.`);
  }
  const trend = W.rel40[i]>=0 ? `Index-level trend is still intact: price sits ${W.rel40[i].toFixed(1)}% above its 40-week MA` : `Index-level trend has broken: price sits ${Math.abs(W.rel40[i]).toFixed(1)}% below its 40-week MA`;
  out.push(`${trend}, 14-week RSI is ${f1(W.rsi[i])}, and VIX at ${f1(W.vix[i])} is at its ${Math.round(pctile(W.vix,W.vix[i]))}th percentile since 2016.`);
  const E=D.stress.episodes, done=E.filter(e=>e.f13!=null);
  const nowSetup = W.dd52[i]>-3 && W.p50[i]<40;
  if(done.length){
    const pos=done.filter(e=>e.f13>0).length, mean=done.reduce((a,e)=>a+e.f13,0)/done.length;
    const worst=done.reduce((a,e)=>e.mdd13<a.mdd13?e:a,done[0]);
    out.push(`${nowSetup?'This combination, a near-high index with thin 50-day participation,':'A near-high index with thin 50-day participation (not the case this week)'} has occurred ${done.length} times before since 2016. The S&amp;P 500 was higher 13 weeks later in ${pos} of ${done.length}, by ${sgn(mean,1)}% on average; the worst drawdown inside those 13 weeks was ${sgn(worst.mdd13,1)}%, after ${fmtMY(worst.d)}. <a href="#stress">Stress test below.</a>`);
  }
  document.getElementById('readText').innerHTML = out.map(s=>`<p>${s}</p>`).join('');
}

const METRIC = {
  f13:{fmt:v=>sgn(v,1)+'%'}, f26:{fmt:v=>sgn(v,1)+'%'}, hit13:{fmt:v=>Math.round(v)+'%'}, mdd13:{fmt:v=>sgn(v,1)+'%'}
};
function hexA(hex,a){ return hex.slice(0,7)+Math.round(Math.max(0,Math.min(1,a))*255).toString(16).padStart(2,'0'); }
function renderHeat(){
  const m=st.metric, U=D.stress.uncond[m];
  const G=D.stress.quint.map(g=>({...g, q: g.key==='vix'? g.q.slice().reverse() : g.q}));
  let maxd=0; G.forEach(g=>g.q.forEach(q=>{ maxd=Math.max(maxd,Math.abs(q[m]-U)); }));
  const H=document.getElementById('heat');
  H.querySelector('thead').innerHTML = `<tr><th>Gauge</th><th>Q1, weakest</th><th>Q2</th><th>Q3</th><th>Q4</th><th>Q5, strongest</th></tr>`;
  const gFmt = (k,v)=> k==='summ'? Math.round(v) : (k==='p200'||k==='p50'? v.toFixed(0)+'%' : (k==='rel40'? v.toFixed(1)+'%' : v.toFixed(1)));
  H.querySelector('tbody').innerHTML = G.map(g=>`<tr><td>${g.label}</td>${g.q.map(q=>{
      const dlt=q[m]-U, a=maxd? Math.abs(dlt)/maxd*0.5 : 0;
      const bg = dlt>=0? hexA(C.teal,a) : hexA(C.orange,a);
      const lo = g.key==='vix'? q.hi : q.lo, hi = g.key==='vix'? q.lo : q.hi;
      return `<td><div class="cell" style="background:${bg}">${METRIC[m].fmt(q[m])}<small>${gFmt(g.key,lo)} to ${gFmt(g.key,hi)}</small></div></td>`;
    }).join('')}</tr>`).join('') +
    `<tr class="unc"><td>All weeks</td><td colspan="5" style="text-align:left;padding-left:10px" class="num">${METRIC[m].fmt(U)} <span class="muted" style="font-family:var(--f-body);font-size:13px">average across ${D.stress.uncond.n} weeks with a full 13-week window</span></td></tr>`;
}
function renderStressText(){
  const U=D.stress.uncond;
  const G=D.stress.quint.map(g=>({label:g.label, weak: g.key==='vix'? g.q[4] : g.q[0], strong: g.key==='vix'? g.q[0] : g.q[4]}));
  const wk=G.map(g=>g.weak.f13), hitW=G.map(g=>g.weak.hit13), ddW=G.map(g=>g.weak.mdd13);
  const nBeat = wk.filter(v=>v>U.f13).length;
  const p1 = nBeat===6
    ? `For all six gauges the weakest quintile was followed by the strongest average 13-week return, ${Math.min(...wk).toFixed(1)}% to ${Math.max(...wk).toFixed(1)}% against ${U.f13.toFixed(1)}% for all weeks. Over this sample weak internals marked buying opportunities, not warnings.`
    : `Weak-reading quintiles beat the all-weeks 13-week return (${U.f13.toFixed(1)}%) for ${nBeat} of 6 gauges, ranging ${Math.min(...wk).toFixed(1)}% to ${Math.max(...wk).toFixed(1)}%.`;
  const p2 = `The edge is size, not frequency. Hit rates in the weakest quintile run ${Math.min(...hitW)}% to ${Math.max(...hitW)}% against ${U.hit13}% overall, and the average 13-week max drawdown that followed (${sgn(Math.min(...ddW),1)}% to ${sgn(Math.max(...ddW),1)}%, against ${sgn(U.mdd13,1)}%) was no shallower than normal. What weak readings caught was the rebound leg of sharp, V-shaped washouts: 2016, 2018, 2020, 2022, 2025.`;
  let worstCell=null; D.stress.quint.forEach(g=>g.q.forEach((q,k)=>{ if(!worstCell||q.mdd13<worstCell.v) worstCell={v:q.mdd13,g:g.label,k:(g.key==='vix'?5-k:k+1)}; }));
  const p3 = `No gauge isolates drawdown risk convincingly. The deepest average follow-on 13-week drawdown, ${sgn(worstCell.v,1)}% in quintile ${worstCell.k} of ${worstCell.g}, is ${Math.abs(worstCell.v-U.mdd13).toFixed(1)} points worse than all weeks, on overlapping windows. The sample is one secular uptrend without a grinding bear market, which is the regime where buying weak breadth works, and each quintile holds about ${Math.round(U.n/5)} weeks but only a handful of independent 13-week periods. Treat all of this as regime-conditional evidence, not a structural rule.`;
  document.getElementById('stressText').innerHTML = [p1,p2,p3].map(s=>`<p>${s}</p>`).join('');
}
function renderEpisodes(){
  const cls=v=>v==null?'':(v>0?'pos':'neg');
  document.querySelector('#eps tbody').innerHTML = D.stress.episodes.map(e=>{
    const now=e.f13==null;
    return `<tr class="${now?'now':''}"><td>${fmtDate(e.d)}${now?' <span class="muted" style="font-size:12px">in progress</span>':''}</td><td class="num">${fmtPx(e.spx)}</td><td class="num">${sgn(e.dd,1)}%</td><td class="num">${f1(e.p50)}%</td><td class="num">${f1(e.p200)}%</td><td class="num">${sgn(e.summ,0)}</td><td class="num ${cls(e.f13)}">${e.f13==null?'':sgn(e.f13,1)+'%'}</td><td class="num ${cls(e.mdd13)}">${e.mdd13==null?'':sgn(e.mdd13,1)+'%'}</td><td class="num ${cls(e.f26)}">${e.f26==null?'':sgn(e.f26,1)+'%'}</td></tr>`;
  }).join('');
}

function renderNotes(){
  document.getElementById('notes').innerHTML = `
  <h3>Universe and membership</h3>
  <p>Breadth is computed bottom-up from S&amp;P 500 members, not the NYSE composite. Point-in-time membership is rebuilt backward from today's constituent list using Wikipedia's historical change table (${M.n_changes} changes since 2015), so each day counts only stocks in the index that day. Two ticker aliases (SATS to ECHO, RE to EG) and one missing change (Smurfit Westrock for WestRock, Jul-2024) are patched by hand.</p>
  <h3>Prices and coverage</h3>
  <p>Daily split-adjusted closes from Yahoo Finance, not dividend-adjusted. Yahoo drops most acquired and delisted names: ${M.n_found} of ${M.n_ever} ever-members returned history, including renamed companies recovered under their current symbols. Coverage of live membership bottoms at ${f1(M.cov_min)}% (${fmtMY(M.cov_min_d)}) and is ${f1(M.cov_now)}% now. Early readings are mildly survivorship-tinted toward stocks that lasted; the coverage chart shows by how much.</p>
  <h3>Definitions</h3>
  <p>% above MA counts members closing above their 50- or 200-day simple moving average, over members with a full window (recent IPOs stay out of the denominator until they have one). McClellan Summation uses daily ratio-adjusted net advances, (A&minus;D)/(A+D)&times;1000, a 19-day minus 39-day EMA oscillator (10% and 5% smoothing), cumulated. With both EMAs seeded at zero the sum equals 20&times;EMA39 &minus; 10&times;EMA19 exactly (max check error ${M.ident_err}), so its level does not depend on the start date. VIX is the CBOE index on a reversed axis. RSI is 14-week Wilder on weekly closes; the 40-week MA is a simple average of weekly closes.</p>
  <h3>Weekly sampling</h3>
  <p>Every daily series is sampled at the last session of each Friday-ending week, ${M.weeks} weeks from January 2016.${M.partial?` The final bar (${fmtDate(M.last)}) is a partial week and will move until Friday's close.`:''}</p>
  <h3>Stress test</h3>
  <p>Forward returns use weekly S&amp;P 500 price closes, excluding dividends. The 13-week max drawdown is the lowest weekly close over the next 13 weeks relative to the signal week. Quintile edges use the whole sample, which puts look-ahead into the thresholds.</p>`;
  document.querySelector('#valid tbody').innerHTML = D.validation.map(v=>`<tr><td>${v.ep}</td><td class="num">${fmtDate(v.d)}</td><td class="num">${f1(v.p200d)}%</td><td class="num">${f1(v.p200w)}%</td><td class="num">${f1(v.p50d)}%</td><td class="num">${f1(v.vix)}</td><td class="num">${f1(v.cov)}%</td></tr>`).join('');
  document.getElementById('asof').innerHTML = `Weekly data to <b>${fmtDate(M.last)}</b>${M.partial?' (partial week)':''}<br>${M.nmem} members, ${f1(M.cov_now)}% with prices<br>Built ${M.generated}`;
  document.getElementById('foot').innerHTML = `Sources: Yahoo Finance (constituent closes, ^GSPC, ^VIX); Wikipedia, List of S&amp;P 500 companies and Historical components of the S&amp;P 500. Regenerate with sp500_internals_rebuild.py. Not investment advice.`;
}
let covChart=null;
function buildCov(){
  if(covChart) covChart.destroy();
  covChart = new Chart(document.getElementById('covChart'),{
    type:'line', data:{datasets:[{data:W.cov.map((v,i)=>({x:i,y:v})), borderColor:C.slate, borderWidth:1.4, pointRadius:0, fill:{target:{value:100}, above:C.clear, below:C.orangeA}}]},
    options:{animation:false, responsive:true, maintainAspectRatio:false, parsing:false, events:[], plugins:{legend:{display:false},tooltip:{enabled:false}},
      layout:{padding:{top:4,right:6}},
      scales:{x:{type:'linear',min:0,max:N-1,border:{display:false},grid:{color:C.grid},ticks:{color:C.muted,font:monoFont,autoSkip:false,maxRotation:0,callback:v=>{ const d=W.d[Math.round(v)]; return d?d.slice(0,4):''; }},afterBuildTicks:sc=>{ sc.ticks=xTicks(0,N-1).filter((_,k)=>k%2===0); }},
              y:{min:70,max:100,border:{display:false},grid:{color:C.grid},ticks:{color:C.muted,font:monoFont,stepSize:10,callback:v=>v+'%'}}}}
  });
}

const ord = n => { const s=['th','st','nd','rd'], v=n%100; return n+(s[(v-20)%10]||s[v]||s[0]); };
function renderTabs(){
  const el=document.getElementById('tabs');
  el.innerHTML = TABS.map(x=>`<button role="tab" id="tab-${x.id}" data-tab="${x.id}" aria-selected="${x.id===st.tab}" tabindex="${x.id===st.tab?0:-1}"><span class="lg">${x.label}</span><span class="sm">${x.short}</span><span class="ct">${x.panels.length}</span></button>`).join('');
  el.addEventListener('click',e=>{ const b=e.target.closest('button[data-tab]'); if(b) setTab(b.dataset.tab); });
  el.addEventListener('keydown',e=>{
    if(e.key!=='ArrowLeft'&&e.key!=='ArrowRight'&&e.key!=='Home'&&e.key!=='End') return;
    e.preventDefault(); e.stopPropagation();
    const k=TABS.findIndex(x=>x.id===st.tab);
    const n = e.key==='Home'?0 : e.key==='End'?TABS.length-1 : (k+(e.key==='ArrowRight'?1:-1)+TABS.length)%TABS.length;
    setTab(TABS[n].id); document.getElementById('tab-'+TABS[n].id).focus();
  });
}
function setTab(id, focusPanel){
  if(!TABS.some(x=>x.id===id)) return;
  st.tab=id;
  document.querySelectorAll('#tabs button').forEach(b=>{ const on=b.dataset.tab===id; b.setAttribute('aria-selected',on); b.tabIndex=on?0:-1; });
  document.getElementById('panels').setAttribute('aria-labelledby','tab-'+id);
  try { localStorage.setItem(TAB_KEY,id); } catch(e) {}
  renderTabNote();
  if(window.Chart){ buildPanels(); applyRange(); }
  if(focusPanel){ const p=document.getElementById('panel-'+focusPanel); if(p) p.scrollIntoView({block:'start'}); }
}
function renderTabNote(){
  const el=document.getElementById('tabNote'); const i=N-1, j=Math.max(0,N-5);
  if(st.tab==='all'){ el.hidden=true; el.innerHTML=''; return; }
  const lb=Math.max(0,N-26); let pk=lb; for(let k=lb;k<=i;k++) if(W.spx[k]>W.spx[pk]) pk=k;
  const pc = k => Math.round(pctile(W[k],W[k][i]));
  const dd = W.dd52[i];
  const ddTxt = dd>-0.05 ? 'at its 52-week high' : `${Math.abs(dd).toFixed(1)}% below its 52-week high`;
  let html='';
  if(st.tab==='breadth'){
    const avg = (pc('p50')+pc('p200')+pc('summ'))/3;
    const detail = `% above 50-day at ${f1(W.p50[i])}% (${ord(pc('p50'))} percentile), % above 200-day at ${f1(W.p200[i])}% (${ord(pc('p200'))}) and McClellan Summation at ${sgn(W.summ[i],0)} (${ord(pc('summ'))})`;
    if(dd>-5 && avg<30) html = `<b>Negative divergence.</b> The S&amp;P 500 is ${ddTxt}, but breadth readings average the ${ord(Math.round(avg))} percentile since 2016: ${detail}.`;
    else if(dd<-10 && avg>60) html = `<b>Positive divergence.</b> The S&amp;P 500 is ${ddTxt}, but breadth readings average the ${ord(Math.round(avg))} percentile: ${detail}.`;
    else html = `<b>Price and breadth broadly aligned.</b> The S&amp;P 500 is ${ddTxt}, with ${detail}.`;
  } else if(st.tab==='vix'){
    const lo52=Math.max(0,N-52); let vmax=lo52, vmin=lo52;
    for(let k=lo52;k<=i;k++){ if(W.vix[k]>W.vix[vmax]) vmax=k; if(W.vix[k]<W.vix[vmin]) vmin=k; }
    const vp=pc('vix'), bAvg=(pc('p50')+pc('p200')+pc('summ'))/3;
    let lead;
    if(vp>=80) lead = '<b>Volatility is pricing stress.</b>';
    else if(bAvg<30 && vp<70) lead = '<b>Volatility has not confirmed the breadth deterioration.</b>';
    else if(vp<=20) lead = '<b>Volatility is subdued.</b>';
    else lead = '<b>Volatility is unremarkable.</b>';
    html = `${lead} VIX reads ${f1(W.vix[i])}, its ${ord(vp)} percentile since 2016, ${sgn(W.vix[i]-W.vix[j],1)} over four weeks. Its 52-week range of weekly closes runs ${f1(W.vix[vmin])} to ${f1(W.vix[vmax])} (${fmtDate(W.d[vmax])}). The S&amp;P 500 is ${ddTxt}.`;
  } else if(st.tab==='mom'){
    const r=W.rsi[i], g=W.rel40[i], rp=W.rsi[pk], gp=W.rel40[pk];
    let lead;
    if(g>=0 && r>=50) lead = (pk!==i && r<rp-5) ? '<b>Trend and momentum still positive, but fading.</b>' : '<b>Trend and momentum both positive.</b>';
    else if(g>=0 && r<50) lead = '<b>Trend holds, momentum has turned negative.</b>';
    else if(g<0 && r>=50) lead = '<b>Momentum recovering, price still below trend.</b>';
    else lead = '<b>Trend and momentum both negative.</b>';
    const since = pk===i ? ' The index is at its six-month high this week.' : ` At the ${fmtDate(W.d[pk])} high they read ${f1(rp)} and ${sgn(gp,1)}%.`;
    html = `${lead} 14-week RSI is ${f1(r)} (${ord(pc('rsi'))} percentile since 2016) and price is ${sgn(g,1)}% versus its 40-week MA (${ord(pc('rel40'))}).${since}`;
  }
  el.hidden=false; el.innerHTML=`<p>${html}</p>`;
}
document.getElementById('readings').addEventListener('click',e=>{
  const a=e.target.closest('a[data-k]'); if(!a) return;
  const k=a.dataset.k; const inTab = TABS.find(x=>x.id===st.tab).panels.some(([id])=>id===k);
  if(!inTab && HOME[k]){ e.preventDefault(); setTab(HOME[k], k); }
});

function renderAll(){
  C = isDark()?PAL.dark:PAL.light;
  buildPanels(); applyRange(); buildCov(); renderHeat();
}
renderReadings(); renderRead(); renderStressText(); renderEpisodes(); renderNotes(); renderTabs(); renderTabNote();
document.getElementById('panels').setAttribute('aria-labelledby','tab-'+st.tab);
if(!window.Chart){
  document.getElementById('panels').innerHTML='<p style="padding:16px">Charts did not load because Chart.js could not be downloaded. Check the connection and reload the page.</p>';
  renderHeat();
} else {
  Chart.defaults.font.family="'IBM Plex Sans', 'Helvetica Neue', Arial, sans-serif";
  renderAll();
  let rt=null, lastNarrow=window.innerWidth<560;
  window.addEventListener('resize',()=>{ clearTimeout(rt); rt=setTimeout(()=>{ const n=window.innerWidth<560; if(n!==lastNarrow){ lastNarrow=n; renderAll(); } else { applyRange(); } },180); });
  if(window.matchMedia){ const mq=matchMedia('(prefers-color-scheme: dark)'); if(mq.addEventListener) mq.addEventListener('change',renderAll); }
  new MutationObserver(renderAll).observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});
  if(document.fonts && document.fonts.ready) document.fonts.ready.then(()=>charts.forEach(c=>c.update('none')));
}
})();
</script>
</body>
</html>
'''


if __name__ == '__main__':
    main()
