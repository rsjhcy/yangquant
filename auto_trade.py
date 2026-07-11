#!/usr/bin/env python3
"""
自动交易执行脚本 — 本地运行，Windows 定时任务触发

首次使用:
    python auto_trade.py --setup
    → 打开浏览器 → 手动扫码登录同花顺 → 登录状态保存

每日自动:
    python auto_trade.py
    → 从 GitHub 拉取最新交易计划 → 自动下单

定时任务:
    Windows 任务计划程序 → 每天 9:25 → 运行此脚本
"""

import json
import sys
from datetime import date
from pathlib import Path


def main():
    # 自动判断模式
    if "--setup" in sys.argv or "--login" in sys.argv:
        do_setup()
    elif "--dry-run" in sys.argv:
        do_trade(dry_run=True)
    else:
        do_trade(dry_run=False)


def do_setup():
    """首次设置：打开浏览器登录同花顺，保存登录状态"""
    print("=" * 50)
    print("  同花顺首次登录设置")
    print("=" * 50)
    print()
    print("1. 会弹出浏览器窗口")
    print("2. 请用同花顺APP扫码登录（或手机验证码）")
    print("3. 登录成功后回到这里按 Enter")
    print("4. 登录状态会自动保存，以后不需要再登录")
    print()

    from quant.execution.ths_trader import THSTrader

    trader = THSTrader(headless=False)
    if trader.login():
        print()
        print("✅ 登录成功！以后可以自动交易了")
        # 验证账户
        info = trader.get_account_info()
        if "error" not in info:
            print(f"   账户信息: {json.dumps(info, ensure_ascii=False)}")
    else:
        print("❌ 登录失败，请重试")

    trader.close()


def do_trade(dry_run: bool = False):
    """执行交易"""
    import subprocess

    print(f"[{date.today()}] 开始自动交易...")

    # 1. 从 GitHub 拉取最新交易计划
    plan_path = Path("data/trade_plan.json")

    # 尝试 git pull
    try:
        result = subprocess.run(
            ["git", "pull", "--rebase"],
            capture_output=True, text=True, timeout=30,
        )
        print(f"  git pull: {result.stdout.strip() or '已是最新'}")
    except Exception as e:
        print(f"  git pull 失败: {e}（使用本地交易计划）")

    # 2. 检查交易计划是否存在
    if not plan_path.exists():
        print("❌ 交易计划不存在，请确保收盘筛选已运行")
        return

    with open(plan_path, "r", encoding="utf-8") as f:
        plan = json.load(f)

    plan_date = plan.get("date", "unknown")
    print(f"  交易计划日期: {plan_date}")

    # 如果交易计划不是今天，跳过
    if plan_date != str(date.today()):
        print(f"⚠ 交易计划日期({plan_date})不是今天({date.today()})，跳过")
        return

    # 3. 执行交易
    print()
    print("--- 平衡型买入 ---")
    for p in plan.get("balanced", [])[:3]:
        qty = _calc_qty(float(p["close"]), 100000, 3)
        print(f"  {p['symbol']} {p.get('name','')} {qty}股 @ {p['close']}")

    print()
    print("--- 激进型买入 ---")
    for p in plan.get("aggressive", [])[:3]:
        qty = _calc_qty(float(p["close"]), 100000, 3)
        print(f"  {p['symbol']} {p.get('name','')} {qty}股 @ {p['close']}")

    print()
    if dry_run:
        print("[模拟模式] 未实际下单")
        return

    confirm = input("确认执行以上交易? (y/N): ").strip().lower()
    if confirm != "y":
        print("已取消")
        return

    # 4. Playwright 自动下单
    from quant.execution.ths_trader import THSTrader

    trader = THSTrader(headless=False)  # 非 headless 方便调试

    if not trader.login():
        print("❌ 登录失败（可能需要重新扫码），请运行 python auto_trade.py --setup")
        trader.close()
        return

    # 执行激进型（更高收益）
    style = plan.get("style", "aggressive")
    orders = []
    for pick in plan.get(style, [])[:3]:
        orders.append({
            "symbol": pick["symbol"],
            "price": float(pick["close"]),
            "quantity": _calc_qty(float(pick["close"]), 100000, 3),
        })

    results = trader.execute_buy(orders, dry_run=False)
    trader.close()

    # 输出结果
    ok = sum(1 for r in results if r["status"] == "SUBMITTED")
    print(f"\n✅ 完成: {ok}/{len(results)} 笔下单成功")
    for r in results:
        print(f"  {r['symbol']}: {r['status']} - {r['message']}")


def _calc_qty(price: float, capital: int, n: int) -> int:
    """整手计算"""
    return max(100, int(capital / n / price / 100) * 100)


if __name__ == "__main__":
    main()
