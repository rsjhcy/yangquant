"""
止盈止损参数网格搜索优化
测试多种止损/止盈/移动止损/时间止损组合，找到最优参数

用法: python backtest_optimize.py [--stocks 100]
结果输出到控制台 + data/sell_optimize_result.json
"""

import sys, os, time, random, json, io
from datetime import date, datetime, timedelta
from pathlib import Path
from collections import defaultdict
from itertools import product
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# ─── 数据获取 ────────────────────────────
def get_kline(symbol, days=120):
    import requests, urllib3
    urllib3.disable_warnings()
    code = f"sh{symbol}" if symbol.startswith(("6", "9")) else f"sz{symbol}"
    url = (f"https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get"
           f"?param={code},day,,,{days},qfq")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://finance.qq.com/",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        raw = resp.json()
        data = raw.get("data", {}).get(code, {}).get("qfqday", []) or \
               raw.get("data", {}).get(code, {}).get("day", [])
        if not data: return None
        rows = [{"date": d[0], "open": float(d[1]), "close": float(d[2]),
                 "high": float(d[3]), "low": float(d[4]), "volume": float(d[5])} for d in data]
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").reset_index(drop=True)
    except: return None


def calc_atr(df, period=14):
    high, low, close = df["high"].values, df["low"].values, df["close"].values
    tr = np.zeros(len(df))
    for i in range(1, len(df)):
        tr[i] = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
    tr[0] = high[0] - low[0]
    return pd.Series(tr, index=df.index).rolling(period).mean()


# ─── 模拟一次持仓（参数化版本）─────────────────
def simulate_hold_v2(df, buy_idx, params):
    """
    params = {
        stop_type: "fixed" | "atr"       止损类型
        stop_loss_pct: 0.05               固定止损百分比（负值表示下方）
        stop_atr_mult: 2.0                ATR止损倍数
        take_profit_pct: 0.10             固定止盈百分比
        tp_atr_mult: 3.0                  ATR止盈倍数
        trailing_trigger_pct: 0.05        移动止损触发点
        trailing_pct: 0.04                移动止损回撤比例
        time_stop_days: 5                 时间止损（几天不涨就卖）
        max_hold_days: 10
        scale_out: True                   是否分批止盈
        tp1_ratio: 0.5                    止盈1卖多少
        tp1_pct: 0.05                     止盈1目标
    }
    """
    buy_price = df["close"].iloc[buy_idx]
    atr_series = calc_atr(df)
    atr_val = atr_series.iloc[buy_idx] if buy_idx >= 14 else buy_price * 0.03
    atr_pct = atr_val / buy_price if buy_price > 0 else 0.03

    # ── 确定止损价 ──
    if params.get("stop_type") == "atr" and buy_idx >= 14:
        stop_loss_price = buy_price * (1 - params["stop_atr_mult"] * atr_pct)
    else:
        stop_loss_price = buy_price * (1 - params["stop_loss_pct"])

    # ── 确定止盈价 ──
    scale_out = params.get("scale_out", False)
    if scale_out:
        tp1_price = buy_price * (1 + params.get("tp1_pct", 0.05))
        if params.get("tp_type") == "atr" and buy_idx >= 14:
            tp2_price = buy_price * (1 + params["tp_atr_mult"] * atr_pct)
        else:
            tp2_price = buy_price * (1 + params["take_profit_pct"])
    else:
        if params.get("tp_type") == "atr" and buy_idx >= 14:
            tp1_price = buy_price * (1 + params["tp_atr_mult"] * atr_pct)
        else:
            tp1_price = buy_price * (1 + params["take_profit_pct"])
        tp2_price = tp1_price  # 不分批
        scale_out = False

    # ── 移动止损 ──
    trailing_trigger = buy_price * (1 + params.get("trailing_trigger_pct", 0.05))
    trailing_pct = params.get("trailing_pct", 0.04)
    time_stop_days = params.get("time_stop_days", 5)
    max_hold = params.get("max_hold_days", 10)
    tp1_ratio = params.get("tp1_ratio", 0.5)

    # 跟踪变量
    max_price = buy_price
    min_price = buy_price
    highest_close = buy_price
    trailing_stop_active = False
    trailing_stop_price = 0
    is_underwater_days = 0  # 连续水下天数

    for i in range(buy_idx + 1, min(buy_idx + max_hold + 1, len(df))):
        row = df.iloc[i]
        high = row["high"]
        low = row["low"]
        close = row["close"]
        hold_days = i - buy_idx

        max_price = max(max_price, high)
        min_price = min(min_price, low)
        highest_close = max(highest_close, close)

        # 水下天数
        if close < buy_price:
            is_underwater_days += 1
        else:
            is_underwater_days = 0

        # T+1跳过
        if hold_days == 1:
            continue

        # ── 时间止损：连续水下N天 ──
        if time_stop_days > 0 and is_underwater_days >= time_stop_days:
            exit_price = close
            pnl_pct = (exit_price / buy_price - 1)
            return {
                "outcome": "时间止损", "pnl_pct": round(pnl_pct, 4),
                "hold_days": hold_days, "exit_price": exit_price,
                "max_profit": round((max_price / buy_price - 1), 4),
                "max_loss": round((min_price / buy_price - 1), 4),
            }

        # ── 更新移动止损 ──
        if not trailing_stop_active and close >= trailing_trigger:
            trailing_stop_active = True
            trailing_stop_price = buy_price  # 保本
        if trailing_stop_active:
            trailing_stop_price = max(trailing_stop_price, highest_close * (1 - trailing_pct))

        # ── 检查止损 ──
        effective_stop = max(stop_loss_price, trailing_stop_price) if trailing_stop_active else stop_loss_price
        if low <= effective_stop:
            exit_price = effective_stop
            pnl_pct = (exit_price / buy_price - 1)
            return {
                "outcome": "止损",
                "pnl_pct": round(pnl_pct, 4),
                "hold_days": hold_days,
                "exit_reason": "移动止损" if trailing_stop_active else "初始止损",
                "exit_price": exit_price,
                "max_profit": round((max_price / buy_price - 1), 4),
                "max_loss": round((min_price / buy_price - 1), 4),
            }

        # ── 检查止盈（分批） ──
        if scale_out and high >= tp1_price:
            # 止盈1触发，卖一半
            # 如果还没触发tp2，先记录；如果同一天触发tp2，取tp2
            already_hit_tp1 = True
            # 继续检查tp2

        if high >= tp2_price:
            if scale_out and 'already_hit_tp1' in dir():
                avg_exit = tp1_price * tp1_ratio + tp2_price * (1 - tp1_ratio)
            else:
                avg_exit = tp2_price
            pnl_pct = (avg_exit / buy_price - 1)
            return {
                "outcome": "止盈",
                "pnl_pct": round(pnl_pct, 4),
                "hold_days": hold_days,
                "exit_reason": "止盈触发",
                "exit_price": tp2_price,
                "max_profit": round((max_price / buy_price - 1), 4),
                "max_loss": round((min_price / buy_price - 1), 4),
            }

    # ── 到期平仓 ──
    last_idx = min(buy_idx + max_hold, len(df) - 1)
    exit_price = df["close"].iloc[last_idx]
    pnl_pct = (exit_price / buy_price - 1)
    return {
        "outcome": "到期平仓",
        "pnl_pct": round(pnl_pct, 4),
        "hold_days": last_idx - buy_idx,
        "exit_price": exit_price,
        "max_profit": round((max_price / buy_price - 1), 4),
        "max_loss": round((min_price / buy_price - 1), 4),
    }


# ─── 评估一组参数 ──────────────────────
def evaluate_params(params, stock_data, buy_points_per_stock=5):
    """对一组参数跑所有股票的所有买入点，返回汇总统计"""
    results = []
    for sym, df in stock_data.items():
        n = len(df)
        buy_points = list(range(30, n - 15, max(1, (n - 45) // buy_points_per_stock)))
        for bp in buy_points:
            r = simulate_hold_v2(df, bp, params)
            r["symbol"] = sym
            results.append(r)

    if not results: return None

    pnls = np.array([r["pnl_pct"] for r in results])
    win_count = int(np.sum(pnls > 0))
    loss_count = int(np.sum(pnls < 0))
    n = len(pnls)

    avg_win = float(np.mean(pnls[pnls > 0])) if win_count > 0 else 0
    avg_loss = float(np.mean(pnls[pnls < 0])) if loss_count > 0 else 0
    profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')

    # 期望收益
    win_rate = win_count / n
    expected_return = win_rate * avg_win + (1 - win_rate) * avg_loss

    # 各类退出统计
    outcomes = defaultdict(int)
    outcome_pnls = defaultdict(float)
    for r in results:
        outcomes[r["outcome"]] += 1
        outcome_pnls[r["outcome"]] += r["pnl_pct"]

    return {
        "n": n,
        "win_rate": round(win_rate, 4),
        "win_count": win_count,
        "loss_count": loss_count,
        "avg_pnl": round(float(np.mean(pnls)), 4),
        "median_pnl": round(float(np.median(pnls)), 4),
        "expected_return": round(expected_return, 4),
        "avg_win": round(avg_win, 4),
        "avg_loss": round(avg_loss, 4),
        "profit_factor": round(profit_factor, 2),
        "total_return": round(float(np.sum(pnls)), 4),
        "sharpe_like": round(float(np.mean(pnls) / max(np.std(pnls), 1e-6)), 4),
        "outcomes": dict(outcomes),
        "outcome_avg_pnl": {k: round(v / max(outcomes[k], 1), 4) for k, v in outcome_pnls.items()},
    }


# ─── 网格搜索 ───────────────────────────
GRID_BALANCED = {
    "stop_loss_pct": [0.05, 0.06, 0.07, 0.08],
    "take_profit_pct": [0.08, 0.10, 0.12, 0.15],
    "trailing_trigger_pct": [0.04, 0.06, 0.08],
    "trailing_pct": [0.03, 0.05, 0.07],
    "time_stop_days": [3, 5, 0],  # 0=不用时间止损
    "max_hold_days": [8, 10, 12],
    "scale_out": [True, False],
    "tp1_pct": [0.04, 0.06],
    "tp1_ratio": [0.33, 0.5],
}

GRID_AGGRESSIVE = {
    "stop_loss_pct": [0.06, 0.08, 0.10],
    "take_profit_pct": [0.12, 0.15, 0.18, 0.22],
    "trailing_trigger_pct": [0.06, 0.09, 0.12],
    "trailing_pct": [0.05, 0.07, 0.10],
    "time_stop_days": [2, 4, 0],
    "max_hold_days": [5, 7, 10],
    "scale_out": [True, False],
    "tp1_pct": [0.06, 0.08],
    "tp1_ratio": [0.33, 0.5],
}


def grid_search(grid, stock_data, label="", max_combos=None):
    """网格搜索最优参数"""
    keys = list(grid.keys())
    values = list(grid.values())
    combos = list(product(*values))

    if max_combos and len(combos) > max_combos:
        # 随机采样
        random.seed(42)
        combos = random.sample(combos, max_combos)

    print(f"\n{'='*60}")
    print(f"  网格搜索: {label}")
    print(f"  参数组合数: {len(combos)}")
    print(f"{'='*60}")

    best = None
    best_score = -float('inf')
    all_eval = []

    for idx, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        params["stop_type"] = "fixed"
        params["tp_type"] = "fixed"

        ev = evaluate_params(params, stock_data, buy_points_per_stock=3)
        if ev is None: continue

        ev["params"] = params
        all_eval.append(ev)

        # 评分：综合考虑期望收益、胜率、盈亏比
        score = ev["expected_return"] * 0.5 + ev["sharpe_like"] * 0.3 + ev["profit_factor"] * 0.001

        if score > best_score:
            best_score = score
            best = ev

        if (idx + 1) % 20 == 0:
            print(f"  ... {idx+1}/{len(combos)} | 当前最优: ER={best['expected_return']:.4f} "
                  f"WR={best['win_rate']:.1%} PF={best['profit_factor']:.2f}")

    # 排序输出top10
    all_eval.sort(key=lambda x: x["expected_return"], reverse=True)

    print(f"\n  Top 10 参数组合 (按期望收益):")
    print(f"  {'─'*55}")
    for i, ev in enumerate(all_eval[:10]):
        p = ev["params"]
        print(f"  {i+1}. ER={ev['expected_return']*100:+.2f}% | "
              f"WR={ev['win_rate']:.1%} | "
              f"AvgW={ev['avg_win']*100:+.2f}% | "
              f"AvgL={ev['avg_loss']*100:+.2f}% | "
              f"PF={ev['profit_factor']:.2f}")
        print(f"     SL={p['stop_loss_pct']:.0%} TP={p['take_profit_pct']:.0%} "
              f"TrailTrig={p['trailing_trigger_pct']:.0%} Trail={p['trailing_pct']:.0%} "
              f"TimeStop={p['time_stop_days']}d MaxHold={p['max_hold_days']}d "
              f"ScaleOut={p['scale_out']}")

    return best, all_eval


# ─── 主程序 ────────────────────────────
def main():
    import akshare as ak

    stock_count = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[1] == "--stocks" else 60

    print("=" * 60)
    print("  止盈止损参数优化器")
    print("=" * 60)

    # 1. 获取股票池
    print("\n>>> 获取股票池...")
    stock_df = ak.stock_info_a_code_name()
    main_board = stock_df[stock_df["code"].str.match(r"^(60|00)\d{4}$")]
    main_board = main_board[~main_board["name"].str.contains("ST|退", na=False)]
    all_codes = main_board["code"].tolist()
    print(f"  主板: {len(all_codes)} 只")

    random.seed(42)
    samples = random.sample(all_codes, min(stock_count, len(all_codes)))

    # 2. 下载数据
    print(f"\n>>> 下载 {len(samples)} 只股票K线...")
    stock_data = {}
    for i, sym in enumerate(samples):
        df = get_kline(sym, 150)
        if df is not None and len(df) >= 60:
            stock_data[sym] = df
        if (i+1) % 20 == 0:
            print(f"  {i+1}/{len(samples)} (成功{len(stock_data)})")
        time.sleep(random.uniform(0.01, 0.05))
    print(f"  成功: {len(stock_data)} 只")

    if len(stock_data) < 10:
        print("  ERROR: 数据不足"); return

    # 3. 评估当前策略
    print("\n>>> 评估当前策略...")
    current_balanced = {
        "stop_type": "fixed", "stop_loss_pct": 0.05,
        "take_profit_pct": 0.12, "tp_type": "fixed",
        "trailing_trigger_pct": 0.06, "trailing_pct": 0.06,
        "max_hold_days": 10, "time_stop_days": 0,
        "scale_out": True, "tp1_pct": 0.06, "tp1_ratio": 0.5,
    }
    current_agg = {
        "stop_type": "fixed", "stop_loss_pct": 0.07,
        "take_profit_pct": 0.18, "tp_type": "fixed",
        "trailing_trigger_pct": 0.09, "trailing_pct": 0.09,
        "max_hold_days": 7, "time_stop_days": 0,
        "scale_out": True, "tp1_pct": 0.09, "tp1_ratio": 0.5,
    }

    ev_cur_bal = evaluate_params(current_balanced, stock_data, buy_points_per_stock=5)
    ev_cur_agg = evaluate_params(current_agg, stock_data, buy_points_per_stock=5)

    print(f"  当前平衡型: ER={ev_cur_bal['expected_return']*100:+.2f}% "
          f"WR={ev_cur_bal['win_rate']:.1%} PF={ev_cur_bal['profit_factor']:.2f}")
    print(f"  当前激进型: ER={ev_cur_agg['expected_return']*100:+.2f}% "
          f"WR={ev_cur_agg['win_rate']:.1%} PF={ev_cur_agg['profit_factor']:.2f}")

    # 4. 网格搜索
    print("\n>>> 开始网格搜索平衡型参数...")
    best_bal, all_bal = grid_search(GRID_BALANCED, stock_data, "平衡型", max_combos=200)

    print("\n>>> 开始网格搜索激进型参数...")
    best_agg, all_agg = grid_search(GRID_AGGRESSIVE, stock_data, "激进型", max_combos=200)

    # 5. 输出最优结果
    print("\n" + "=" * 60)
    print("  >>> 最终推荐参数 <<<")
    print("=" * 60)

    for label, best, current in [
        ("平衡型", best_bal, ev_cur_bal),
        ("激进型", best_agg, ev_cur_agg),
    ]:
        if best is None: continue
        print(f"\n  [{label}]")
        print(f"  当前: ER={current['expected_return']*100:+.2f}% WR={current['win_rate']:.1%} PF={current['profit_factor']:.2f}")
        print(f"  最优: ER={best['expected_return']*100:+.2f}% WR={best['win_rate']:.1%} PF={best['profit_factor']:.2f}")
        p = best["params"]
        print(f"  参数: SL={p['stop_loss_pct']:.0%} TP={p['take_profit_pct']:.0%} "
              f"TrailTrig={p['trailing_trigger_pct']:.0%} Trail={p['trailing_pct']:.0%} "
              f"TimeStop={p['time_stop_days']}d MaxHold={p['max_hold_days']}d "
              f"ScaleOut={p['scale_out']}")
        print(f"  退出分布: {best['outcomes']}")

    # 6. 保存结果
    out_path = Path("data/sell_optimize_result.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_data = {
        "current": {
            "balanced": {k: v for k, v in ev_cur_bal.items() if k != "params"},
            "aggressive": {k: v for k, v in ev_cur_agg.items() if k != "params"},
        },
        "best": {
            "balanced": {k: (v if k != "params" else {kk: vv for kk, vv in v.items()})
                        for k, v in best_bal.items()} if best_bal else None,
            "aggressive": {k: (v if k != "params" else {kk: vv for kk, vv in v.items()})
                          for k, v in best_agg.items()} if best_agg else None,
        },
        "top10_balanced": [{k: (v if k != "params" else {kk: vv for kk, vv in v.items()})
                           for k, v in ev.items()} for ev in all_bal[:10]],
        "top10_aggressive": [{k: (v if k != "params" else {kk: vv for kk, vv in v.items()})
                            for k, v in ev.items()} for ev in all_agg[:10]],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    print(f"\n  结果已保存: {out_path}")


if __name__ == "__main__":
    main()
