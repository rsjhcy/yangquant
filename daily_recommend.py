#!/usr/bin/env python3
"""
羊量每日选股推荐系统

用法:
    # 收盘后运行 (15:30): 筛选全主板股票
    python daily_recommend.py --phase close

    # 次日盘前运行 (9:20): 分析竞价 + 发送邮件
    python daily_recommend.py --phase auction

    # 一键两阶段 (在同一天内不会同时做两件事)
    python daily_recommend.py --phase all
"""

import sys
from datetime import date, datetime
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).parent))

from quant.utils.logger import setup_logger
from loguru import logger


@click.command()
@click.option(
    "--phase", "-p",
    type=click.Choice(["close", "auction", "all"]),
    default="all",
    help="close=收盘筛选 | auction=竞价验证+发邮件 | all=自动判断",
)
@click.option(
    "--style", "-s",
    type=click.Choice(["balanced", "aggressive", "both"]),
    default="both",
    help="选股风格",
)
@click.option(
    "--top-n", "-n",
    default=3,
    help="每种风格推荐数量",
)
@click.option(
    "--send/--no-send",
    default=True,
    help="是否发送邮件",
)
def main(phase, style, top_n, send):
    """🐑 羊量每日选股推荐系统"""
    setup_logger()

    # Use Beijing time (UTC+8) for all time decisions
    from datetime import timezone, timedelta
    beijing_tz = timezone(timedelta(hours=8))
    now = datetime.now(beijing_tz)
    hour = now.hour
    today = now.date()  # Beijing date

    # 自动判断阶段
    if phase == "all":
        if hour < 12 and hour >= 8:
            phase = "auction"
        else:
            phase = "close"

    # 检查是否为交易日
    from quant.data.calendar import cal
    cal.load()
    if not cal.is_trading_day(today):
        logger.info(f"{today} 非交易日，跳过")
        return

    if phase == "close":
        run_close_phase(style, top_n)
    elif phase == "auction":
        run_auction_phase(style, top_n, send)


def run_close_phase(style: str, top_n: int):
    """收盘后: 筛选主板股票，保存候选"""
    logger.info("=" * 50)
    logger.info("  收盘多因子筛选")
    logger.info("=" * 50)

    from quant.screener import CloseScreener

    screener = CloseScreener()
    result = screener.run(style=style, top_n=top_n)

    # 打印结果
    for st in (["balanced", "aggressive"] if style == "both" else [style]):
        picks = result.get(st, [])
        if not picks:
            continue

        style_name = "平衡型" if st == "balanced" else "激进型"
        print(f"\n{'─' * 50}")
        print(f"  【{style_name}推荐】")
        print(f"{'─' * 50}")

        from quant.screener.sell_advisor import SellAdvisor
        advisor = SellAdvisor()

        for i, pick in enumerate(picks):
            medal = ["🥇", "🥈", "🥉"][i]
            # Generate sell plan
            atr = advisor.get_atr_pct(pick['symbol'])
            sell_plan = advisor.generate(
                entry_price=float(pick['close']),
                atr_pct=atr,
                style=st,
                market_regime=market.get('regime', 'ranging'),
            )
            # Attach sell plan to pick
            pick['sell_plan'] = sell_plan
            pick['sell_plan_str'] = (
                f"止损: {sell_plan['stop_loss']}({sell_plan['stop_loss_pct']}) | "
                f"止盈1: {sell_plan['take_profit_1']}({sell_plan['take_profit_1_pct']}) | "
                f"止盈2: {sell_plan['take_profit_2']}({sell_plan['take_profit_2_pct']}) | "
                f"持有≤{sell_plan['max_hold_days']}天"
            )

            print(f"  {medal} {pick['symbol']} {pick['name']}")
            print(f"     综合得分: {pick['score']:.1f}  昨收: {pick['close']}")
            print(f"     动量: {pick['momentum_score']:.1f}  趋势: {pick['trend_score']:.1f}  量价: {pick['volume_score']:.1f}")
            print(f"     理由: {pick['reason']}")
            print(f"     📉 卖出计划: {pick['sell_plan_str']}")
            print(f"     💡 {sell_plan.get('note','')}")
            print()

    market = result.get("market", {})
    print(f"  市场概况: {market.get('total','-')}只主板 | "
          f"{market.get('up_count','-')}↑/{market.get('down_count','-')}↓ | "
          f"平均涨跌: {market.get('avg_pct_chg','-')}")

    logger.info(f"\n候选已保存到 data/candidates.json")

    # 也保存交易计划 (给 auto_trade.py 使用)
    trade_plan = {
        "date": str(date.today()),
        "style": style if style != "both" else "aggressive",
        "capital": 100000,
        "balanced": result.get("balanced", []),
        "aggressive": result.get("aggressive", []),
    }
    tp_path = Path("data/trade_plan.json")
    with open(tp_path, "w", encoding="utf-8") as f:
        json.dump(trade_plan, f, ensure_ascii=False, indent=2)
    logger.info(f"交易计划已保存 → {tp_path}")

    logger.info(f"明天 9:20 运行: python daily_recommend.py --phase auction")


def run_auction_phase(style: str, top_n: int, send_email: bool):
    """盘前: 分析竞价，验证候选，发送邮件"""
    logger.info("=" * 50)
    logger.info("  集合竞价分析")
    logger.info("=" * 50)

    from quant.screener import CloseScreener, AuctionAnalyzer

    # 加载昨日候选
    screener = CloseScreener()
    candidates_data = screener.load_candidates()
    if not candidates_data:
        logger.error("未找到候选数据！请先运行收盘筛选: python daily_recommend.py --phase close")
        return

    market = candidates_data.get("market", {})

    # 分析竞价
    analyzer = AuctionAnalyzer()

    final = {}

    for st in (["balanced", "aggressive"] if style == "both" else [style]):
        candidates = candidates_data.get(st, [])
        if not candidates:
            logger.warning(f"无{st}候选")
            continue

        # 用实时报价补充前收盘价
        symbols = [c["symbol"] for c in candidates]
        prev_close_map = analyzer.get_prev_close_map(symbols)

        # 验证
        validated = analyzer.validate(candidates, prev_close_map)
        final[st] = validated

        style_name = "平衡型" if st == "balanced" else "激进型"
        print(f"\n{'─' * 50}")
        print(f"  【{style_name}竞价验证结果】")
        print(f"{'─' * 50}")
        for i, pick in enumerate(validated):
            status = "✅" if pick.get("auction_score", 0) >= 55 else "⚠️"
            print(f"  {status} {pick['symbol']} {pick['name']}")
            print(f"     竞价得分: {pick.get('auction_score', '-')}  昨收: {pick.get('close', '-')}  {pick.get('open_pct', '-')}")
            print(f"     明细: {pick.get('auction_detail', '-')}")
            print(f"     建议: {pick.get('recommendation', '-')}")
            print()

    # 发送邮件
    if send_email:
        from quant.notify import EmailSender

        print("\n发送邮件...")
        sender = EmailSender()
        ok = sender.send_recommendation(
            balanced_picks=final.get("balanced", []),
            aggressive_picks=final.get("aggressive", []),
            market_summary=market,
        )
        if ok:
            print("✅ 邮件发送成功！")
        else:
            print("❌ 邮件发送失败，请检查 config.yaml 中的邮箱配置")
            print("   QQ邮箱需要授权码(不是登录密码):")
            print("   QQ邮箱→设置→账户→POP3/SMTP服务→生成授权码")


if __name__ == "__main__":
    main()
