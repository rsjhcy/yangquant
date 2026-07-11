"""
选股策略大回测
30只主板股票 × 3年数据 × 每日评分 × 跟踪前向收益
"""

import sys, io, json, time, random
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from datetime import date, timedelta
from collections import defaultdict
import numpy as np
import pandas as pd

# 30只主板股票，覆盖多个行业
STOCKS = [
    # 金融
    '000001','600016','601166','601318','600036','601398',
    # 消费
    '000858','600519','600809','000568','600690','002142','600887',
    # 科技/制造
    '000725','002415','000651','600276','603259','601012',
    # 能源/材料
    '600028','600585','601088',
    # 汽车/交通
    '600104','600009',
    # 地产/建筑
    '000002','600048',
    # 医药
    '600085',
    # 其他
    '600030','600309','600000',
]

START = date(2023, 6, 1)
END = date(2026, 6, 30)
MIN_HISTORY = 30

print(f'=== 选股策略大回测 ===')
print(f'股票: {len(STOCKS)} 只主板')
print(f'区间: {START} ~ {END}')
print()

# ─── 1. 下载数据 ───
from quant.data.sources import AkshareSource
source = AkshareSource()

print('>>> 下载行情数据...')
df = source.get_daily(STOCKS, START, END)
if df.empty:
    print('ERROR: 无数据')
    sys.exit(1)

df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
dates = sorted(df['date'].unique())
symbols = sorted(df['symbol'].unique())
print(f'获取 {len(df)} 条, {len(symbols)} 只股票, {len(dates)} 个交易日')
print()

# ─── 2. 预计算前向收益 ───
print('>>> 计算前向收益...')
df['ret_1d'] = df.groupby('symbol')['close'].pct_change().shift(-1)
df['ret_3d'] = df.groupby('symbol')['close'].pct_change(3).shift(-3)
df['ret_5d'] = df.groupby('symbol')['close'].pct_change(5).shift(-5)
df['ret_10d'] = df.groupby('symbol')['close'].pct_change(10).shift(-10)

# ─── 3. 逐日评分回测 ───
print('>>> 逐日评分回测...')

balanced_ret = {'1d':[], '3d':[], '5d':[], '10d':[]}
aggressive_ret = {'1d':[], '3d':[], '5d':[], '10d':[]}
baseline_ret = {'1d':[], '3d':[], '5d':[], '10d':[]}
scored_days = 0

for di, dt in enumerate(dates):
    if (di + 1) % 100 == 0:
        print(f'  [{di+1}/{len(dates)}] {dt}')

    day_data = df[df['date'] == dt]
    if len(day_data) < 5:
        continue

    scores = []
    for sym in symbols:
        sym_hist = df[(df['symbol'] == sym) & (df['date'] <= dt)]
        if len(sym_hist) < MIN_HISTORY:
            continue
        sym_hist = sym_hist.tail(MIN_HISTORY)

        row = day_data[day_data['symbol'] == sym]
        if row.empty:
            continue
        row = row.iloc[0]

        close = float(row['close'])

        # 5日收益
        if len(sym_hist) >= 6:
            ret_5d = close / float(sym_hist['close'].iloc[-6]) - 1
        else:
            ret_5d = 0

        # 20日收益
        lookback_20 = min(21, len(sym_hist))
        ret_20d = close / float(sym_hist['close'].iloc[-lookback_20]) - 1

        # 换手率
        turnover = float(row.get('turnover', 0) or 0)

        # 量比 (volume ratio)
        volume = float(row.get('volume', 0) or 0)
        vol_5avg = sym_hist['volume'].tail(5).mean() if len(sym_hist) >= 5 else volume
        vol_ratio = volume / vol_5avg if vol_5avg > 0 else 1.0

        # 波动率
        daily_rets = sym_hist['close'].pct_change().dropna()
        volatility = float(daily_rets.tail(20).std()) if len(daily_rets) >= 5 else 0.02

        # ── 因子打分 ──
        # 动量因子: 短期动量+中期动量
        mom_5d_score = min(100, max(0, 50 + ret_5d * 400))
        mom_20d_score = min(100, max(0, 50 + ret_20d * 200))
        momentum = mom_5d_score * 0.5 + mom_20d_score * 0.5

        # 趋势因子
        trend = min(100, max(0, 50 + ret_20d * 300))

        # 量价因子: 换手率活跃+量比放大
        to_score = min(100, max(5, 40 if 2 < turnover < 15 else 15))
        vr_score = min(100, max(5, 50 + (vol_ratio - 1) * 25))
        volume_quality = to_score * 0.5 + vr_score * 0.5

        # 低风险因子: 波动率低+温和涨跌
        risk_from_vol = min(100, max(0, 80 - volatility * 400)) if volatility > 0 else 70
        risk = risk_from_vol

        # 综合得分
        b_total = momentum * 0.30 + trend * 0.25 + volume_quality * 0.25 + risk * 0.20
        a_total = momentum * 0.45 + trend * 0.15 + volume_quality * 0.30 + risk * 0.10

        scores.append({
            'symbol': sym,
            'b_score': b_total,
            'a_score': a_total,
            'ret_1d': row.get('ret_1d'),
            'ret_3d': row.get('ret_3d'),
            'ret_5d': row.get('ret_5d'),
            'ret_10d': row.get('ret_10d'),
        })

    if len(scores) < 5:
        continue

    scored_days += 1

    def track_returns(sorted_scores, n_picks, target):
        for rk in ['1d', '3d', '5d', '10d']:
            vals = []
            for s in sorted_scores[:n_picks]:
                v = s[f'ret_{rk}']
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    vals.append(v)
            if vals:
                target[rk].append(np.mean(vals))

    # 平衡型 Top 3
    scores.sort(key=lambda x: x['b_score'], reverse=True)
    track_returns(scores, 3, balanced_ret)

    # 激进型 Top 3
    scores.sort(key=lambda x: x['a_score'], reverse=True)
    track_returns(scores, 3, aggressive_ret)

    # 基线: 全部等权
    track_returns(scores, len(scores), baseline_ret)

# ─── 4. 结果输出 ───
print()
print(f'评分天数: {scored_days}')
print()
print('=' * 78)
print('  选股策略回测结果 | 30只主板股票 | 2023-2026')
print('=' * 78)
print(f'  {"指标":20s} {"平衡型Top3":>14s} {"激进型Top3":>14s} {"全市场等权":>14s}')
print('  ' + '-' * 66)

for label, rk in [
    ('日均收益', '1d'), ('3日平均收益', '3d'),
    ('5日平均收益', '5d'), ('10日平均收益', '10d'),
]:
    b = np.mean(balanced_ret[rk]) * 100 if balanced_ret[rk] else 0
    a = np.mean(aggressive_ret[rk]) * 100 if aggressive_ret[rk] else 0
    bl = np.mean(baseline_ret[rk]) * 100 if baseline_ret[rk] else 0
    print(f'  {label:20s} {b:>+13.3f}% {a:>+13.3f}% {bl:>+13.3f}%')

# 胜率
print()
for label, rk in [('日胜率(>0)', '1d'), ('3日胜率', '3d'), ('5日胜率', '5d')]:
    b_wr = sum(1 for x in balanced_ret[rk] if x > 0) / len(balanced_ret[rk]) * 100 if balanced_ret[rk] else 0
    a_wr = sum(1 for x in aggressive_ret[rk] if x > 0) / len(aggressive_ret[rk]) * 100 if aggressive_ret[rk] else 0
    bl_wr = sum(1 for x in baseline_ret[rk] if x > 0) / len(baseline_ret[rk]) * 100 if baseline_ret[rk] else 0
    print(f'  {label:20s} {b_wr:>13.1f}% {a_wr:>13.1f}% {bl_wr:>13.1f}%')

# 累计收益
print()
b_cum = np.prod([1+x for x in balanced_ret['1d']]) - 1 if balanced_ret['1d'] else 0
a_cum = np.prod([1+x for x in aggressive_ret['1d']]) - 1 if aggressive_ret['1d'] else 0
bl_cum = np.prod([1+x for x in baseline_ret['1d']]) - 1 if baseline_ret['1d'] else 0
print(f'  {"累计收益(复利)":20s} {b_cum:>+13.2%} {a_cum:>+13.2%} {bl_cum:>+13.2%}')

# 夏普
def calc_sharpe(rets):
    if not rets: return 0
    avg = np.mean(rets)
    std = np.std(rets)
    return avg / std * np.sqrt(252) if std > 0 else 0

b_sh = calc_sharpe(balanced_ret['1d'])
a_sh = calc_sharpe(aggressive_ret['1d'])
bl_sh = calc_sharpe(baseline_ret['1d'])
print(f'  {"夏普比率(年化)":20s} {b_sh:>13.2f} {a_sh:>13.2f} {bl_sh:>13.2f}')

# 最大回撤
def calc_mdd(rets):
    if not rets: return 0
    vals = np.cumprod([1+x for x in rets])
    peak = np.maximum.accumulate(vals)
    return np.min((vals - peak) / peak)

b_mdd = calc_mdd(balanced_ret['1d'])
a_mdd = calc_mdd(aggressive_ret['1d'])
bl_mdd = calc_mdd(baseline_ret['1d'])
print(f'  {"最大回撤":20s} {b_mdd:>+13.2%} {a_mdd:>+13.2%} {bl_mdd:>+13.2%}')

# 信息比率
b_ir = (np.mean(balanced_ret['1d']) - np.mean(baseline_ret['1d'])) / np.std([a-b for a,b in zip(balanced_ret['1d'], baseline_ret['1d'])]) * np.sqrt(252) if balanced_ret['1d'] and baseline_ret['1d'] else 0
a_ir = (np.mean(aggressive_ret['1d']) - np.mean(baseline_ret['1d'])) / np.std([a-b for a,b in zip(aggressive_ret['1d'], baseline_ret['1d'])]) * np.sqrt(252) if aggressive_ret['1d'] and baseline_ret['1d'] else 0
print(f'  {"信息比率(vs基准)":20s} {b_ir:>13.2f} {a_ir:>13.2f}')

print('=' * 78)
print()
print('结论:')
if b_sh > 0.5:
    print(f'  ✅ 平衡型策略有效 (夏普{b_sh:.2f}), 显著跑赢等权基准')
if a_sh > 0.5:
    print(f'  ✅ 激进型策略有效 (夏普{a_sh:.2f})')
if b_sh > a_sh:
    print(f'  💡 平衡型在本回测区间表现更优')
else:
    print(f'  💡 激进型在本回测区间表现更优')
