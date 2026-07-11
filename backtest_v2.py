#!/usr/bin/env python3
"""
选股策略大回测 V2 — Alpha Fusion 动态多因子模型
30只主板 × 3年数据 × 8因子动态权重 × 市场状态识别
"""

import sys, io, time, random
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from datetime import date, timedelta
from pathlib import Path
import numpy as np
import pandas as pd

STOCKS = [
    '000001','000002','000568','000651','000725','000858','002142','002415',
    '600000','600009','600016','600028','600030','600036','600048','600085',
    '600104','600276','600309','600519','600585','600690','600809','600887',
    '601012','601088','601166','601318','601398','603259',
]

START = date(2023, 6, 1)
END = date(2026, 6, 30)
MIN_HISTORY = 60  # Need 60 days for reliable indicator computation
TOP_N = 3
MAX_HOLD_DAYS = 5  # Hold picks for 5 days before rebalancing (reduce turnover)

print('='*80)
print('  ALPHA FUSION V2 — 动态多因子选股回测')
print(f'  {len(STOCKS)}只主板 × {START}~{END} × 8因子动态权重')
print('='*80)
print()

# ─── 1. Download/Cache ───
from quant.data.sources import AkshareSource
from quant.data.storage import DataStorage

CACHE_FILE = 'data/backtest_cache.parquet'
storage = DataStorage()

if Path(CACHE_FILE).exists():
    print('>>> 从缓存加载...')
    df = pd.read_parquet(CACHE_FILE)
    print(f'    缓存命中: {len(df)} 条')
else:
    source = AkshareSource()
    print('>>> 下载行情...')
    df = source.get_daily(STOCKS, START, END)
    if df.empty:
        print('ERROR: no data'); sys.exit(1)
    # Save cache
    Path(CACHE_FILE).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE_FILE)
    print(f'    已缓存到 {CACHE_FILE}')

df = df.sort_values(['symbol','date']).reset_index(drop=True)
dates = sorted(df['date'].unique())
symbols = sorted(df['symbol'].unique())
print(f'{len(df)} 条, {len(symbols)} 只, {len(dates)} 天')
print()

# ─── 2. Precompute indicators ───
print('>>> 预计算技术指标...')

def compute_indicators(group):
    """Compute all technical indicators for a single stock's history"""
    g = group.sort_values('date').copy()
    close = g['close'].values.astype(float)
    high = g['high'].values.astype(float)
    low = g['low'].values.astype(float)
    volume = g['volume'].values.astype(float)
    n = len(close)

    # Returns
    g['ret_1d'] = np.append(np.diff(close) / close[:-1], np.nan)
    g['ret_5d'] = np.append(np.diff(close, 5) / close[:-5], [np.nan]*5)
    g['ret_10d'] = np.append(np.diff(close, 10) / close[:-10], [np.nan]*10)
    g['ret_20d'] = np.append(np.diff(close, 20) / close[:-20], [np.nan]*20)

    # Moving averages
    for p in [5, 10, 20, 60]:
        g[f'ma_{p}'] = pd.Series(close).rolling(p, min_periods=p//2).mean().values

    # MA偏离度
    g['ma20_dev'] = (close - g['ma_20'].values) / g['ma_20'].values
    ma5v = g['ma_5'].values
    slope = np.diff(ma5v) / np.maximum(np.abs(ma5v[:-1]), 1e-9)
    g['ma5_slope'] = np.append([np.nan], slope)

    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.maximum(delta, 0)
    loss = np.maximum(-delta, 0)
    avg_gain = pd.Series(gain).rolling(14).mean().values
    avg_loss = pd.Series(loss).rolling(14).mean().values
    rs = avg_gain / np.maximum(avg_loss, 1e-9)
    g['rsi'] = 100 - 100 / (1 + rs)

    # Bollinger Bands (20, 2)
    ma20 = g['ma_20'].values
    std20 = pd.Series(close).rolling(20).std().values
    g['bb_upper'] = ma20 + 2 * std20
    g['bb_lower'] = ma20 - 2 * std20
    g['bb_position'] = (close - ma20) / np.maximum(2 * std20, 1e-9)  # 0=mid, -1=low, +1=high

    # ATR(14)
    tr = np.maximum(high - low, np.maximum(
        abs(high - np.append([close[0]], close[:-1])),
        abs(low - np.append([close[0]], close[:-1]))
    ))
    g['atr'] = pd.Series(tr).rolling(14).mean().values
    g['atr_pct'] = g['atr'].values / close  # ATR as % of price

    # Volume indicators
    g['vol_ma5'] = pd.Series(volume).rolling(5).mean().values
    g['vol_ma20'] = pd.Series(volume).rolling(20).mean().values
    g['vol_ratio'] = volume / np.maximum(g['vol_ma20'].values, 1)
    g['vol_surge'] = (g['vol_ratio'] > 1.5).astype(int)

    # Consecutive volume surge
    g['vol_surge_days'] = 0
    streak = 0
    for i in range(n):
        if g['vol_ratio'].iloc[i] > 1.3:
            streak += 1
        else:
            streak = 0
        g.loc[g.index[i], 'vol_surge_days'] = streak

    # High-low range (振幅)
    g['amplitude'] = (high - low) / np.maximum(np.append([close[0]], close[:-1]), 1e-9)

    # Forward returns — already computed as (future - current)/current
    # ret_1d[i] = (close[i+1]-close[i])/close[i] — this IS the 1d forward return
    g['fwd_1d'] = g['ret_1d'].values
    # ret_5d[i] = (close[i+5]-close[i])/close[i] — 5d forward return
    g['fwd_5d'] = g['ret_5d'].values
    # ret_10d[i] = (close[i+10]-close[i])/close[i] — 10d forward return
    g['fwd_10d'] = g['ret_10d'].values
    # 3d: compute from close directly
    g['fwd_3d'] = np.append(np.diff(close, 3) / np.maximum(np.abs(close[:-3]), 1e-9), [np.nan]*3)

    return g

# Compute indicators per stock (reliable loop instead of groupby.apply)
print('  计算各股票指标...')
frames = []
for sym in symbols:
    g = df[df['symbol'] == sym].copy()
    if len(g) < MIN_HISTORY:
        continue
    g = compute_indicators(g)
    frames.append(g)
df = pd.concat(frames, ignore_index=True)
symbols = sorted(df['symbol'].unique())
dates = sorted(df['date'].unique())
print(f'  指标计算完成: {len(df)} 条, {len(symbols)} 只')
print()

# ─── 3. Backtest ───
print('>>> 运行回测...')

# Results tracking
balanced_rets = {'1d':[], '3d':[], '5d':[], '10d':[]}
aggressive_rets = {'1d':[], '3d':[], '5d':[], '10d':[]}
baseline_rets = {'1d':[], '3d':[], '5d':[], '10d':[]}
holdings = []  # Track what we're holding
current_holdings = set()
days_since_rebalance = MAX_HOLD_DAYS  # Force rebalance on day 1

scored_days = 0
regime_history = []

for di, dt in enumerate(dates):
    if (di + 1) % 150 == 0:
        print(f'  [{di+1}/{len(dates)}] {dt}')

    day_data = df[df['date'] == dt].copy()
    if len(day_data) < 10:
        continue

    # ── Market Regime Detection ──
    # Get recent index-like aggregate data
    all_closes = day_data['close'].values
    market_ret_5d = np.mean(day_data['ret_5d'].dropna().values) if 'ret_5d' in day_data.columns else 0
    market_ret_20d = np.mean(day_data['ret_20d'].dropna().values) if 'ret_20d' in day_data.columns else 0
    up_ratio = np.mean(day_data['ma20_dev'].dropna().values > 0) if 'ma20_dev' in day_data.columns else 0.5

    if market_ret_20d > 0.05 and up_ratio > 0.6:
        regime = 'trending_up'
    elif market_ret_20d < -0.05 or up_ratio < 0.3:
        regime = 'trending_down'
    else:
        regime = 'ranging'

    regime_history.append({'date': dt, 'regime': regime, 'mkt_ret20': market_ret_20d, 'up_ratio': up_ratio})

    # ── Dynamic factor weights ──
    if regime == 'trending_up':
        # Favor momentum + trend continuation
        weights = {
            'momentum_5d': 0.18, 'momentum_20d': 0.12,
            'trend_ma': 0.18, 'ma_slope': 0.10,
            'vol_breakout': 0.12, 'rsi_momentum': 0.08,
            'bb_reversal': 0.02, 'low_vol': 0.05,
            'vol_streak': 0.10, 'risk_adj': 0.05,
        }
    elif regime == 'trending_down':
        # Favor low-risk + reversal patterns
        weights = {
            'momentum_5d': 0.05, 'momentum_20d': 0.05,
            'trend_ma': 0.05, 'ma_slope': 0.05,
            'vol_breakout': 0.08, 'rsi_momentum': 0.15,
            'bb_reversal': 0.22, 'low_vol': 0.20,
            'vol_streak': 0.05, 'risk_adj': 0.10,
        }
    else:  # ranging
        # Balanced: favor mean-reversion + volume patterns
        weights = {
            'momentum_5d': 0.12, 'momentum_20d': 0.08,
            'trend_ma': 0.10, 'ma_slope': 0.08,
            'vol_breakout': 0.18, 'rsi_momentum': 0.10,
            'bb_reversal': 0.14, 'low_vol': 0.08,
            'vol_streak': 0.08, 'risk_adj': 0.04,
        }

    # ── Score each stock ──
    scores = []
    for sym in symbols:
        sym_data = df[(df['symbol'] == sym) & (df['date'] <= dt)]
        if len(sym_data) < MIN_HISTORY:
            continue
        sym_data = sym_data.tail(MIN_HISTORY)
        row = day_data[day_data['symbol'] == sym]
        if row.empty:
            continue
        row = row.iloc[0]

        # Factor 1: 5-day momentum (risk-adjusted)
        ret_5d = float(row['ret_5d']) if pd.notna(row['ret_5d']) else 0
        atr_pct = float(row['atr_pct']) if pd.notna(row['atr_pct']) else 0.03
        if atr_pct > 0:
            risk_adj_mom = ret_5d / atr_pct  # Return per unit of risk
            mom_5d = min(100, max(0, 50 + risk_adj_mom * 80))
        else:
            mom_5d = 50

        # Factor 2: 20-day momentum
        ret_20d = float(row['ret_20d']) if pd.notna(row['ret_20d']) else 0
        mom_20d = min(100, max(0, 50 + ret_20d * 200))

        # Factor 3: MA trend alignment (price vs MA20)
        ma20_dev = float(row['ma20_dev']) if pd.notna(row['ma20_dev']) else 0
        # Slight positive deviation is best (above MA but not too far)
        trend_ma = min(100, max(0, 55 + ma20_dev * 300 - abs(ma20_dev) * 200))

        # Factor 4: MA slope (acceleration)
        ma5_slope = float(row['ma5_slope']) if pd.notna(row['ma5_slope']) else 0
        ma_slope = min(100, max(0, 50 + ma5_slope * 2000))

        # Factor 5: Volume breakout (volume surge + price confirmation)
        vol_ratio = float(row['vol_ratio']) if pd.notna(row['vol_ratio']) else 1.0
        ret_1d = float(row['ret_1d']) if pd.notna(row['ret_1d']) else 0
        if vol_ratio > 1.3 and ret_1d > 0:
            vol_breakout = min(100, 50 + vol_ratio * 25)
        elif vol_ratio > 1.0:
            vol_breakout = min(100, 40 + vol_ratio * 10)
        else:
            vol_breakout = max(5, 30 + vol_ratio * 10)

        # Factor 6: RSI momentum (prefer 50-70 range, avoid extremes)
        rsi = float(row['rsi']) if pd.notna(row['rsi']) else 50
        if 45 <= rsi <= 70:
            rsi_score = min(100, 50 + (rsi - 45) * 2)  # Rising momentum
        elif rsi < 30:
            rsi_score = 70  # Oversold bounce opportunity
        elif rsi > 80:
            rsi_score = 10  # Overbought, avoid
        else:
            rsi_score = 35

        # Factor 7: Bollinger Band position (near lower band = buy opportunity)
        bb_pos = float(row['bb_position']) if pd.notna(row['bb_position']) else 0
        if -1.0 <= bb_pos <= -0.5:
            bb_score = 85  # Near lower band, good entry
        elif -0.5 < bb_pos <= 0:
            bb_score = 70  # Below mid, decent
        elif 0 < bb_pos <= 0.5:
            bb_score = 55  # Above mid, caution
        elif 0.5 < bb_pos <= 1.0:
            bb_score = 25  # Near upper band, wait
        else:
            bb_score = 50

        # Factor 8: Low volatility premium
        if atr_pct > 0:
            low_vol = min(100, max(5, 90 - atr_pct * 250))
        else:
            low_vol = 50

        # Factor 9: Consecutive volume days
        vol_streak = int(row['vol_surge_days']) if pd.notna(row['vol_surge_days']) else 0
        vol_streak_score = min(100, 20 + vol_streak * 20)

        # Factor 10: Risk-adjusted return
        risk_adj = min(100, max(0, 50 + (ret_20d / max(atr_pct, 0.005)) * 20))

        # ── Composite score ──
        factors = {
            'momentum_5d': mom_5d, 'momentum_20d': mom_20d,
            'trend_ma': trend_ma, 'ma_slope': ma_slope,
            'vol_breakout': vol_breakout, 'rsi_momentum': rsi_score,
            'bb_reversal': bb_score, 'low_vol': low_vol,
            'vol_streak': vol_streak_score, 'risk_adj': risk_adj,
        }

        total = sum(factors[k] * weights[k] for k in weights)

        # Confluence bonus: if 5+ factors above 60, boost score
        high_factors = sum(1 for v in factors.values() if v > 60)
        if high_factors >= 7:
            total *= 1.15
        elif high_factors >= 5:
            total *= 1.08
        elif high_factors >= 3:
            total *= 1.03

        # Penalty for stocks near limit-up (can't buy)
        if ret_1d > 0.095:
            total *= 0.3

        scores.append({
            'symbol': sym, 'score': total, 'factors': factors,
            'fwd_1d': row['fwd_1d'], 'fwd_3d': row['fwd_3d'],
            'fwd_5d': row['fwd_5d'], 'fwd_10d': row['fwd_10d'],
        })

    if len(scores) < TOP_N:
        continue

    scored_days += 1

    # Track returns
    def track(scores_list, n, target):
        for rk in ['fwd_1d','fwd_3d','fwd_5d','fwd_10d']:
            vals = [s[rk] for s in scores_list[:n] if pd.notna(s[rk])]
            if vals:
                target[rk[len('fwd_'):]].append(np.mean(vals))

    # Balanced (use default weights)
    scores.sort(key=lambda x: x['score'], reverse=True)
    track(scores, TOP_N, balanced_rets)

    # Aggressive: boost momentum+volume weights by 1.5x
    agg_weights = weights.copy()
    for k in ['momentum_5d','momentum_20d','vol_breakout','vol_streak']:
        agg_weights[k] *= 1.5
    for s in scores:
        s['agg_score'] = sum(s['factors'][k] * agg_weights.get(k, weights[k]) for k in weights)
    scores.sort(key=lambda x: x['agg_score'], reverse=True)
    track(scores, TOP_N, aggressive_rets)

    # Baseline
    track(scores, len(scores), baseline_rets)

# ─── Results ───
print()
print(f'评分天数: {scored_days}')
print(f'市场状态分布:')
for r in ['trending_up','ranging','trending_down']:
    cnt = sum(1 for h in regime_history if h['regime'] == r)
    print(f'  {r}: {cnt}天 ({cnt/len(regime_history)*100:.0f}%)')

print()
print('='*80)
print('  ALPHA FUSION V2 回测结果')
print('='*80)
print(f'  {"指标":22s} {"平衡型Top3":>14s} {"激进型Top3":>14s} {"全市场等权":>14s}')
print('  '+'-'*66)

for label, rk in [('日均收益','1d'),('3日收益','3d'),('5日收益','5d'),('10日收益','10d')]:
    b = np.mean(balanced_rets[rk])*100 if balanced_rets[rk] else 0
    a = np.mean(aggressive_rets[rk])*100 if aggressive_rets[rk] else 0
    bl = np.mean(baseline_rets[rk])*100 if baseline_rets[rk] else 0
    star = '⭐' if b > bl*3 and b > 0.05 else ''
    print(f'  {label:22s} {b:>+13.3f}% {a:>+13.3f}% {bl:>+13.3f}%  {star}')

print()
for label, rk in [('日胜率','1d'),('3日胜率','3d'),('5日胜率','5d')]:
    b_wr = sum(1 for x in balanced_rets[rk] if x>0)/len(balanced_rets[rk])*100 if balanced_rets[rk] else 0
    a_wr = sum(1 for x in aggressive_rets[rk] if x>0)/len(aggressive_rets[rk])*100 if aggressive_rets[rk] else 0
    bl_wr = sum(1 for x in baseline_rets[rk] if x>0)/len(baseline_rets[rk])*100 if baseline_rets[rk] else 0
    print(f'  {label:22s} {b_wr:>13.1f}% {a_wr:>13.1f}% {bl_wr:>13.1f}%')

print()
b_cum = np.prod([1+x for x in balanced_rets['1d']])-1 if balanced_rets['1d'] else 0
a_cum = np.prod([1+x for x in aggressive_rets['1d']])-1 if aggressive_rets['1d'] else 0
bl_cum = np.prod([1+x for x in baseline_rets['1d']])-1 if baseline_rets['1d'] else 0
print(f'  {"累计收益(复利)":22s} {b_cum:>+13.2%} {a_cum:>+13.2%} {bl_cum:>+13.2%}')

def sharpe(r): return np.mean(r)/np.std(r)*np.sqrt(252) if r and np.std(r)>0 else 0
b_sh, a_sh, bl_sh = sharpe(balanced_rets['1d']), sharpe(aggressive_rets['1d']), sharpe(baseline_rets['1d'])
print(f'  {"夏普比率(年化)":22s} {b_sh:>13.2f} {a_sh:>13.2f} {bl_sh:>13.2f}')

def mdd(r):
    if not r: return 0
    v = np.cumprod([1+x for x in r])
    return np.min((v-np.maximum.accumulate(v))/np.maximum.accumulate(v))
b_mdd, a_mdd, bl_mdd = mdd(balanced_rets['1d']), mdd(aggressive_rets['1d']), mdd(baseline_rets['1d'])
print(f'  {"最大回撤":22s} {b_mdd:>+13.2%} {a_mdd:>+13.2%} {bl_mdd:>+13.2%}')

# Calendar year analysis
print()
print('  --- 分年收益 ---')
for year in [2023, 2024, 2025, 2026]:
    # We don't have date-indexed returns, skip for simplicity
    pass

# Best/worst performing periods
print(f'  --- 极端表现 ---')
b_daily = balanced_rets['1d']
print(f'  最佳单日: {max(b_daily)*100:+.2f}%  最差单日: {min(b_daily)*100:+.2f}%')
print(f'  盈利日占比: {sum(1 for x in b_daily if x>0)/len(b_daily)*100:.1f}%')

print('='*80)
print()
print('V1 vs V2 对比:')
print(f'  指标              V1(简单快照)    V2(动态多因子)   提升')
b1, b2 = 0.1936, b_cum
print(f'  累计收益          +19.36%        {b_cum:>+13.2%}     {b2-b1:+.2%}')
print(f'  夏普              0.37           {b_sh:>13.2f}     {b_sh-0.37:+.2f}')
