#!/usr/bin/env python3
"""Cross-validation: run V2 strategy on new stock group and compare"""
import sys,io
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
import numpy as np, pandas as pd

# Load new data
df = pd.read_parquet('data/backtest_new.parquet')
df = df.sort_values(['symbol','date']).reset_index(drop=True)
symbols = sorted(df['symbol'].unique())
print(f'Group 2: {len(df)} rows, {len(symbols)} stocks, {df["date"].nunique()} days')

MIN_HISTORY = 60; TOP_N = 3

# Copy indicator computation (avoids importing backtest_v2 with side effects)
def compute_indicators(group):
    g = group.sort_values('date').copy()
    close = g['close'].values.astype(float)
    high = g['high'].values.astype(float)
    low = g['low'].values.astype(float)
    volume = g['volume'].values.astype(float)
    n = len(close)
    g['ret_1d'] = np.append(np.diff(close) / close[:-1], np.nan)
    g['ret_5d'] = np.append(np.diff(close, 5) / close[:-5], [np.nan]*5)
    g['ret_10d'] = np.append(np.diff(close, 10) / close[:-10], [np.nan]*10)
    g['ret_20d'] = np.append(np.diff(close, 20) / close[:-20], [np.nan]*20)
    for p in [5, 10, 20, 60]:
        g[f'ma_{p}'] = pd.Series(close).rolling(p, min_periods=p//2).mean().values
    g['ma20_dev'] = (close - g['ma_20'].values) / g['ma_20'].values
    ma5v = g['ma_5'].values
    slope = np.diff(ma5v) / np.maximum(np.abs(ma5v[:-1]), 1e-9)
    g['ma5_slope'] = np.append([np.nan], slope)
    delta = np.diff(close, prepend=close[0])
    gain = np.maximum(delta, 0); loss = np.maximum(-delta, 0)
    avg_gain = pd.Series(gain).rolling(14).mean().values
    avg_loss = pd.Series(loss).rolling(14).mean().values
    rs_ = avg_gain / np.maximum(avg_loss, 1e-9)
    g['rsi'] = 100 - 100 / (1 + rs_)
    ma20 = g['ma_20'].values
    std20 = pd.Series(close).rolling(20).std().values
    g['bb_upper'] = ma20 + 2 * std20
    g['bb_lower'] = ma20 - 2 * std20
    g['bb_position'] = (close - ma20) / np.maximum(2 * std20, 1e-9)
    tr = np.maximum(high - low, np.maximum(abs(high - np.append([close[0]], close[:-1])), abs(low - np.append([close[0]], close[:-1]))))
    g['atr'] = pd.Series(tr).rolling(14).mean().values
    g['atr_pct'] = g['atr'].values / close
    g['vol_ma5'] = pd.Series(volume).rolling(5).mean().values
    g['vol_ma20'] = pd.Series(volume).rolling(20).mean().values
    g['vol_ratio'] = volume / np.maximum(g['vol_ma20'].values, 1)
    g['vol_surge'] = (g['vol_ratio'] > 1.5).astype(int)
    g['vol_surge_days'] = 0
    streak = 0
    for i in range(n):
        if g['vol_ratio'].iloc[i] > 1.3: streak += 1
        else: streak = 0
        g.loc[g.index[i], 'vol_surge_days'] = streak
    return g

print('Computing indicators...')
frames = []
for sym in symbols:
    g = df[df['symbol'] == sym].copy()
    if len(g) < MIN_HISTORY: continue
    g = compute_indicators(g)
    frames.append(g)
df = pd.concat(frames, ignore_index=True)
symbols = sorted(df['symbol'].unique())
dates = sorted(df['date'].unique())
print(f'{len(df)} rows, {len(symbols)} stocks, {len(dates)} days')

print('Running backtest...')
balanced_rets = {'1d':[], '5d':[]}
aggressive_rets = {'1d':[], '5d':[]}
baseline_rets = {'1d':[], '5d':[]}
scored = 0

by_sym = {sym: df[df['symbol']==sym].set_index('date') for sym in symbols}

for di, dt in enumerate(dates):
    if (di+1)%200==0: print(f'  [{di+1}/{len(dates)}] {dt}')
    day_data = df[df['date'] == dt]
    if len(day_data) < 5: continue

    up_r = np.mean(day_data['ma20_dev'].dropna().values > 0) if 'ma20_dev' in day_data.columns else 0.5
    mkt_20 = np.mean(day_data['ret_20d'].dropna().values) if 'ret_20d' in day_data.columns else 0
    if mkt_20 > 0.05 and up_r > 0.6: regime = 'trending_up'
    elif mkt_20 < -0.05 or up_r < 0.3: regime = 'trending_down'
    else: regime = 'ranging'

    if regime == 'trending_up':
        w = {'m5':0.18,'m20':0.12,'tma':0.18,'ms':0.10,'vb':0.12,'rsi':0.08,'bb':0.02,'lv':0.05,'vs':0.10,'ra':0.05}
    elif regime == 'trending_down':
        w = {'m5':0.05,'m20':0.05,'tma':0.05,'ms':0.05,'vb':0.08,'rsi':0.15,'bb':0.22,'lv':0.20,'vs':0.05,'ra':0.10}
    else:
        w = {'m5':0.12,'m20':0.08,'tma':0.10,'ms':0.08,'vb':0.18,'rsi':0.10,'bb':0.14,'lv':0.08,'vs':0.08,'ra':0.04}

    scores = []
    for sym in symbols:
        if sym not in by_sym: continue
        sdf = by_sym[sym]
        if dt not in sdf.index: continue
        row = sdf.loc[dt]

        r5=float(row['ret_5d']) if pd.notna(row['ret_5d']) else 0
        r20=float(row['ret_20d']) if pd.notna(row['ret_20d']) else 0
        ap=float(row['atr_pct']) if pd.notna(row['atr_pct']) else 0.03
        md=float(row['ma20_dev']) if pd.notna(row['ma20_dev']) else 0
        ms=float(row['ma5_slope']) if pd.notna(row['ma5_slope']) else 0
        vr=float(row['vol_ratio']) if pd.notna(row['vol_ratio']) else 1.0
        rs=float(row['rsi']) if pd.notna(row['rsi']) else 50
        bp=float(row['bb_position']) if pd.notna(row['bb_position']) else 0
        vs=int(row['vol_surge_days']) if pd.notna(row['vol_surge_days']) else 0
        r1=float(row['ret_1d']) if pd.notna(row['ret_1d']) else 0

        f_m5=min(100,max(0,50+(r5/max(ap,0.005))*80))
        f_m20=min(100,max(0,50+r20*200))
        f_tma=min(100,max(0,55+md*300-abs(md)*200))
        f_ms=min(100,max(0,50+ms*2000))
        f_vb=min(100,50+vr*25) if vr>1.3 and r1>0 else min(100,40+vr*10)
        f_rsi=min(100,50+(rs-45)*2) if 45<=rs<=70 else (70 if rs<30 else (10 if rs>80 else 35))
        f_bb=85 if -1<=bp<=-0.5 else (70 if -0.5<bp<=0 else (55 if 0<bp<=0.5 else 25))
        f_lv=min(100,max(5,90-ap*250))
        f_vs=min(100,20+vs*20)
        f_ra=min(100,max(0,50+(r20/max(ap,0.005))*20))

        bt=f_m5*w['m5']+f_m20*w['m20']+f_tma*w['tma']+f_ms*w['ms']+f_vb*w['vb']+f_rsi*w['rsi']+f_bb*w['bb']+f_lv*w['lv']+f_vs*w['vs']+f_ra*w['ra']
        at=f_m5*w['m5']*1.5+f_m20*w['m20']*1.5+f_tma*w['tma']+f_ms*w['ms']+f_vb*w['vb']*1.5+f_rsi*w['rsi']+f_bb*w['bb']+f_lv*w['lv']+f_vs*w['vs']*1.5+f_ra*w['ra']
        facs=[f_m5,f_m20,f_tma,f_ms,f_vb,f_rsi,f_bb,f_lv,f_vs,f_ra]
        hc=sum(1 for f in facs if f>60)
        if hc>=7: bt*=1.15; at*=1.15
        elif hc>=5: bt*=1.08; at*=1.08
        if r1>0.095: bt*=0.3; at*=0.3

        f1=float(row['ret_1d']) if pd.notna(row['ret_1d']) else np.nan
        f5=float(row['ret_5d']) if pd.notna(row['ret_5d']) else np.nan
        scores.append({'b':bt,'a':at,'f1':f1,'f5':f5})

    if len(scores) < TOP_N: continue
    scored += 1

    scores.sort(key=lambda x:x['b'],reverse=True)
    for rk,target in [('f1',balanced_rets['1d']),('f5',balanced_rets['5d'])]:
        v=[s[rk] for s in scores[:TOP_N] if pd.notna(s[rk])]
        if v: target.append(np.mean(v))
    scores.sort(key=lambda x:x['a'],reverse=True)
    for rk,target in [('f1',aggressive_rets['1d']),('f5',aggressive_rets['5d'])]:
        v=[s[rk] for s in scores[:TOP_N] if pd.notna(s[rk])]
        if v: target.append(np.mean(v))
    for rk,target in [('f1',baseline_rets['1d']),('f5',baseline_rets['5d'])]:
        v=[s[rk] for s in scores if pd.notna(s[rk])]
        if v: target.append(np.mean(v))

# Results
print(f'\nScored {scored} days')
print()
print('='*75)
print('  CROSS-VALIDATION: Group 1 (original 30) vs Group 2 (new 30)')
print('='*75)
print(f'  {"Metric":22s} {"Group1 Balanced":>14s} {"Group2 Balanced":>14s} {"Group2 Aggressive":>14s}')
print('  '+'-'*66)

G1_DR = 0.00137; G1_5DR = 0.06557; G1_CUM = 1.3936; G1_SH = 1.60; G1_MDD = -0.1915; G1_WR = 52.4

for label,rk in [('Daily Return','1d'),('5-Day Return','5d')]:
    g1 = G1_DR if rk=='1d' else G1_5DR
    g2b = np.mean(balanced_rets[rk]) if balanced_rets[rk] else 0
    g2a = np.mean(aggressive_rets[rk]) if aggressive_rets[rk] else 0
    print(f'  {label:22s} {g1:>+13.3%} {g2b:>+13.3%} {g2a:>+13.3%}')

g2bc = np.prod([1+x for x in balanced_rets['1d']])-1 if balanced_rets['1d'] else 0
g2ac = np.prod([1+x for x in aggressive_rets['1d']])-1 if aggressive_rets['1d'] else 0
print(f'  {"Cumulative Return":22s} {G1_CUM:>+13.2%} {g2bc:>+13.2%} {g2ac:>+13.2%}')

def sh(r): return np.mean(r)/np.std(r)*np.sqrt(252) if r and np.std(r)>0 else 0
g2bs=sh(balanced_rets['1d']); g2as=sh(aggressive_rets['1d'])
print(f'  {"Sharpe Ratio":22s} {G1_SH:>13.2f} {g2bs:>13.2f} {g2as:>13.2f}')

def mdd(r):
    if not r: return 0
    v=np.cumprod([1+x for x in r])
    return np.min((v-np.maximum.accumulate(v))/np.maximum.accumulate(v))
g2bm=mdd(balanced_rets['1d']); g2am=mdd(aggressive_rets['1d'])
print(f'  {"Max Drawdown":22s} {G1_MDD:>+13.2%} {g2bm:>+13.2%} {g2am:>+13.2%}')

g2bw=sum(1 for x in balanced_rets['1d'] if x>0)/len(balanced_rets['1d'])*100 if balanced_rets['1d'] else 0
g2aw=sum(1 for x in aggressive_rets['1d'] if x>0)/len(aggressive_rets['1d'])*100 if aggressive_rets['1d'] else 0
print(f'  {"Win Rate":22s} {G1_WR:>13.1f}% {g2bw:>13.1f}% {g2aw:>13.1f}%')

print('='*75)
ds = abs(g2bs - G1_SH)
if ds < 0.8:
    print(f'  VERDICT: Strategy IS STABLE (Sharpe diff {ds:.2f})')
else:
    print(f'  VERDICT: Strategy shows variance (Sharpe diff {ds:.2f} > 0.8)')
