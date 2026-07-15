#!/usr/bin/env python3
"""
稳定版每日选股推荐 — 本地运行，两阶段流程

阶段1 - 收盘筛选 (15:30):
    python stable_runner.py --phase close
    筛选主板股票 → 保存候选到 data/stable_result.json → 不发邮件

阶段2 - 竞价验证+推送 (次日 9:20):
    python stable_runner.py --phase auction
    加载昨日候选 → 获取竞价数据 → 综合评分 → 发送邮件到两个邮箱

自动判断:
    python stable_runner.py              # 根据当前时间自动选阶段
    python stable_runner.py --now        # 立即跑收盘筛选（测试用）

Windows 定时任务:
    任务1: 每天 15:30 → python stable_runner.py --phase close
    任务2: 每天 09:20 → python stable_runner.py --phase auction
"""

import sys, io, json, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger

# ─── 股票池 + 名称映射 ──────────────────────────────
STOCK_POOL = [
    "000001","000002","000333","000651","000725","000858","002142","002415",
    "002281","002304","600000","600009","600016","600028","600030","600036",
    "600048","600085","600104","600276","600309","600406","600519","600585",
    "600690","600809","600887","601012","601088","601166","601318","601398",
    "601668","601857","601899","603259","603288","603501","603986",
]

NAME_MAP = {
    "000001":"平安银行","000002":"万科A","000333":"美的集团","000651":"格力电器",
    "000725":"京东方A","000858":"五粮液","002142":"宁波银行","002415":"海康威视",
    "002281":"光迅科技","002304":"洋河股份","600000":"浦发银行","600009":"上海机场",
    "600016":"民生银行","600028":"中国石化","600030":"中信证券","600036":"招商银行",
    "600048":"保利发展","600085":"同仁堂","600104":"上汽集团","600276":"恒瑞医药",
    "600309":"万华化学","600406":"国电南瑞","600519":"贵州茅台","600585":"海螺水泥",
    "600690":"海尔智家","600809":"山西汾酒","600887":"伊利股份","601012":"隆基绿能",
    "601088":"中国神华","601166":"兴业银行","601318":"中国平安","601398":"工商银行",
    "601668":"中国建筑","601857":"中国石油","601899":"紫金矿业","603259":"药明康德",
    "603288":"海天味业","603501":"韦尔股份","603986":"兆易创新",
}


def main():
    """入口：自动判断阶段或按参数执行"""
    beijing = timezone(timedelta(hours=8))
    now = datetime.now(beijing)
    hour = now.hour

    # 解析参数
    args = set(sys.argv[1:])
    if "--phase" in args:
        idx = sys.argv.index("--phase")
        phase = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else "auto"
    elif "--now" in args:
        phase = "close"  # 测试模式：强制收盘筛选
    elif 8 <= hour < 12:
        phase = "auction"
    else:
        phase = "close"

    logger.info(f"当前时间: {now.strftime('%Y-%m-%d %H:%M')} (北京) → 执行阶段: {phase}")

    if phase == "close":
        run_close_phase()
    elif phase == "auction":
        run_auction_phase()
    else:
        logger.error(f"未知阶段: {phase}")


# ═══════════════════════════════════════════════════════════════
# 阶段1: 收盘多因子筛选 (15:30 执行)
# ═══════════════════════════════════════════════════════════════

def run_close_phase():
    """收盘筛选 → 保存候选 → 不发邮件"""
    today = date.today()
    beijing_now = datetime.now(timezone(timedelta(hours=8)))
    logger.info(f"{'='*60}")
    logger.info(f"  📊 阶段1: 收盘多因子筛选 — {today}")
    logger.info(f"{'='*60}")

    # 1. 下载近10个交易日数据
    from quant.data.sources import AkshareSource
    source = AkshareSource()
    start = today - timedelta(days=10)
    df = source.get_daily(STOCK_POOL, start, today)

    if df.empty:
        logger.error("❌ 数据下载失败，请检查网络")
        return

    df = df.sort_values(["symbol", "date"])
    latest_date = df["date"].max()
    logger.info(f"✅ 数据就绪: {len(df)}行, 最新日期: {latest_date}")

    # 2. 逐只打分
    results = []
    for sym in STOCK_POOL:
        sym_data = df[df["symbol"] == sym].sort_values("date")
        if len(sym_data) < 3:
            continue

        close_v = sym_data["close"].values.astype(float)
        close = close_v[-1]
        prev = close_v[-2] if len(close_v) >= 2 else close
        daily_pct = (close / prev - 1) * 100

        # 收益率
        r5 = close / close_v[max(0, len(close_v)-6)] - 1
        r20 = close / close_v[max(0, len(close_v)-21)] - 1

        # 换手率 & 成交额
        turnover = float(sym_data.iloc[-1].get("turnover", 3) or 3)
        amount = float(sym_data.iloc[-1].get("amount", 1e8) or 1e8)

        # 趋势强度: 过去20天有多少天收盘 > MA20
        if len(close_v) >= 20:
            ma20 = pd.Series(close_v).rolling(20, min_periods=10).mean().values
            trend_strength = np.sum(close_v[-20:] > ma20[-20:]) / 20
        else:
            trend_strength = 0.5

        # ── 多因子打分 ──
        # 动量因子 (30%)
        mom = min(100, max(0, 50 + r5 * 400)) * 0.5 + \
              min(100, max(0, 50 + r20 * 200)) * 0.5

        # 趋势因子 (25%)
        trend = min(100, max(0, 50 + r20 * 300))

        # 量价因子 (25%)
        to_s = min(100, max(5, 40 if 2 < turnover < 15 else 15))
        vol_q = to_s * 0.6 + 40

        # 风险因子 (20%)
        risk = min(100, max(0, 70 - abs(daily_pct) * 10))

        # 综合得分
        score = mom * 0.30 + trend * 0.25 + vol_q * 0.25 + risk * 0.20

        # 强趋势加分
        is_trending = trend_strength > 0.7 and r20 > 0.05
        if is_trending:
            score *= 1.15

        name = NAME_MAP.get(sym, sym)
        results.append({
            "symbol": sym,
            "name": name,
            "close": f"{close:.2f}",
            "close_val": float(close),
            "score": round(score, 1),
            "momentum_score": round(mom, 1),
            "trend_score": round(trend, 1),
            "volume_score": round(vol_q, 1),
            "risk_score": round(risk, 1),
            "pct_chg": f"{daily_pct:+.2f}%",
            "r5": f"{r5:+.1%}",
            "r20": f"{r20:+.1%}",
            "turnover": round(turnover, 1),
            "amount": amount,
            "is_trending": is_trending,
            "trend_strength": round(trend_strength * 100),
            "reason": _build_reason(r5, r20, turnover, daily_pct, is_trending),
        })

    # 3. 排序选股
    results.sort(key=lambda x: x["score"], reverse=True)

    # 平衡型: 直接取综合分前3
    balanced = _copy_top(results, 3)

    # 激进型: 对强趋势股给额外加分后重排
    for r in results:
        r["agg_score"] = r["score"] * 1.10 if r["is_trending"] else \
            r["momentum_score"] * 0.45 + r["trend_score"] * 0.15 + \
            r["volume_score"] * 0.30 + r["score"] * 0.10
    results.sort(key=lambda x: x.get("agg_score", 0), reverse=True)
    aggressive = _copy_top(results, 3)

    # 4. 生成卖出计划
    from quant.screener.sell_advisor import SellAdvisor
    advisor = SellAdvisor()
    for picks, style in [(balanced, "balanced"), (aggressive, "aggressive")]:
        for p in picks:
            plan = advisor.generate(float(p["close"]), style=style)
            p["sell_plan"] = plan
            p["sell_plan_str"] = (
                f"止损:{plan['stop_loss']}({plan['stop_loss_pct']}) | "
                f"止盈:{plan['take_profit_1']}({plan['take_profit_1_pct']}) | "
                f"移动止损:{plan['trailing_start']} | "
                f"水下{plan['time_stop_days']}天平仓 | "
                f"持有≤{plan['max_hold_days']}天"
            )

    # 5. 打印结果
    print()
    for label, picks in [("🥢 平衡型", balanced), ("🔥 激进型", aggressive)]:
        print(f"  ┌─ {label} ─────────────────────────────")
        for i, p in enumerate(picks):
            medal = ["🥇","🥈","🥉"][i]
            flag = " 🔥强趋势" if p["is_trending"] else ""
            print(f"  │ {medal} {p['symbol']} {p['name']}{flag}")
            print(f"  │   综合:{p['score']}  动量:{p['momentum_score']}  趋势:{p['trend_score']}  量价:{p['volume_score']}")
            print(f"  │   收盘:{p['close']}  {p['pct_chg']}  5日:{p['r5']}  20日:{p['r20']}")
            print(f"  │   {p['reason']}")
            print(f"  │   📉 {p['sell_plan_str']}")
        print(f"  └{'─'*50}")

    # 6. 保存到 JSON（供明早竞价阶段使用）
    output = {
        "balanced": balanced,
        "aggressive": aggressive,
        "date": str(today),
        "saved_at": datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M"),
        "market_summary": {
            "total_screened": len(results),
            "latest_data_date": str(latest_date),
        },
    }
    Path("data").mkdir(exist_ok=True)
    with open("data/stable_result.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info(f"✅ 候选已保存 → data/stable_result.json")
    logger.info(f"⏰ 明早 9:20 自动运行竞价验证 + 发送邮件")


# ═══════════════════════════════════════════════════════════════
# 阶段2: 竞价验证 + 发送邮件 (次日 9:20 执行)
# ═══════════════════════════════════════════════════════════════

def run_auction_phase():
    """加载昨日候选 → 竞价分析 → 发送最终推荐邮件"""
    beijing_now = datetime.now(timezone(timedelta(hours=8)))
    logger.info(f"{'='*60}")
    logger.info(f"  📈 阶段2: 竞价验证 + 推送 — {beijing_now.strftime('%Y-%m-%d %H:%M')}")
    logger.info(f"{'='*60}")

    # 1. 加载昨日候选
    path = Path("data/stable_result.json")
    if not path.exists():
        logger.error("❌ 未找到 data/stable_result.json，请先运行收盘筛选")
        return

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    saved_date = data.get("date", "未知")
    logger.info(f"📂 加载昨日候选 (日期: {saved_date})")

    # 2. 对每只候选获取竞价数据
    import akshare as ak
    all_picks = data.get("balanced", []) + data.get("aggressive", [])

    # 去重
    seen = set()
    unique_picks = []
    for p in all_picks:
        if p["symbol"] not in seen:
            seen.add(p["symbol"])
            unique_picks.append(p)

    logger.info(f"🔍 分析 {len(unique_picks)} 只候选竞价数据...")

    for p in unique_picks:
        sym = p["symbol"]
        try:
            auc = ak.stock_zh_a_hist_pre_min_em(sym, "09:15:00", "09:25:00")
            if auc is not None and not auc.empty:
                # 分析竞价数据
                prices = auc["收盘"].values.astype(float) if "收盘" in auc.columns else None
                volumes = auc["成交量"].values.astype(float) if "成交量" in auc.columns else None

                if prices is not None and len(prices) >= 2:
                    prev_close = float(p["close_val"])

                    # 竞价末期价格 (9:20-9:25)
                    late_prices = prices[-5:] if len(prices) >= 5 else prices
                    early_prices = prices[:5] if len(prices) >= 5 else prices

                    # 价格趋势分 (35%)
                    if len(late_prices) >= 2:
                        price_slope = (late_prices[-1] - late_prices[0]) / late_prices[0] * 100
                        price_score = min(100, max(0, 50 + price_slope * 30))
                    else:
                        price_score = 50

                    # 量能分 (30%)
                    if volumes is not None and len(volumes) >= 5:
                        late_vol = volumes[-5:].sum() if len(volumes) >= 5 else volumes.sum()
                        total_vol = volumes.sum()
                        vol_ratio = late_vol / total_vol if total_vol > 0 else 0.5
                        vol_score = min(100, max(0, vol_ratio * 200))
                    else:
                        vol_score = 50

                    # 高开幅度分 (35%)
                    auction_price = late_prices[-1]
                    gap_pct = (auction_price / prev_close - 1) * 100 if prev_close > 0 else 0
                    if 2 <= gap_pct <= 5:
                        gap_score = 90
                    elif 0 <= gap_pct < 2:
                        gap_score = 65
                    elif 5 < gap_pct <= 7:
                        gap_score = 50
                    elif gap_pct > 7:
                        gap_score = 20  # 高开太多追高风险大
                    else:
                        gap_score = 30  # 低开

                    auction_score = price_score * 0.35 + vol_score * 0.30 + gap_score * 0.35

                    p["auction_price"] = f"{auction_price:.2f}"
                    p["auction_gap"] = f"{gap_pct:+.2f}%"
                    p["auction_score"] = round(auction_score, 1)
                    p["auction_detail"] = (
                        f"竞价:{auction_price:.2f}({gap_pct:+.2f}%) | "
                        f"价格趋势:{price_score:.0f} | 量能:{vol_score:.0f} | 高开:{gap_score:.0f}"
                    )

                    # 综合判定
                    if auction_score >= 70:
                        p["auction_verdict"] = "✅ 竞价强势，建议关注"
                    elif auction_score >= 55:
                        p["auction_verdict"] = "⚠️ 竞价一般，谨慎参与"
                    else:
                        p["auction_verdict"] = "❌ 竞价偏弱，建议观望"
                else:
                    p["auction_gap"] = "N/A"
                    p["auction_score"] = 0
                    p["auction_detail"] = "竞价数据不足"
                    p["auction_verdict"] = "⚠️ 数据不足"
            else:
                p["auction_gap"] = "N/A"
                p["auction_score"] = 0
                p["auction_detail"] = "暂无竞价数据"
                p["auction_verdict"] = "⏳ 数据未出"
        except Exception as e:
            logger.warning(f"  {sym} 竞价获取失败: {e}")
            p["auction_gap"] = "N/A"
            p["auction_score"] = 0
            p["auction_detail"] = f"获取失败"
            p["auction_verdict"] = "⚠️ 获取失败"

        time.sleep(0.3)  # 避免请求过快

    # 3. 重新分组（保持 balanced/aggressive 分组，但附加竞价数据）
    balanced_final = _merge_auction(data.get("balanced", []), unique_picks)
    aggressive_final = _merge_auction(data.get("aggressive", []), unique_picks)

    # 4. 打印最终推荐
    print()
    print(f"  ╔══════════════════════════════════════════════════════╗")
    print(f"  ║  🐑 羊量每日精选 — {beijing_now.strftime('%Y年%m月%d日')} 盘前推荐  ║")
    print(f"  ╚══════════════════════════════════════════════════════╝")

    for label, picks in [("🥢 平衡型（稳健持仓）", balanced_final), ("🔥 激进型（追求收益）", aggressive_final)]:
        print(f"\n  ┌─ {label} ─────────────────────────────")
        for i, p in enumerate(picks):
            medal = ["🥇","🥈","🥉"][i]
            auc_score = p.get("auction_score", 0)
            verdict = p.get("auction_verdict", "⏳")

            print(f"  │ {medal} {p['symbol']} {p['name']}")
            print(f"  │   收盘得分:{p['score']}  竞价得分:{auc_score}")
            print(f"  │   昨收:{p['close']}  竞价:{p.get('auction_price','N/A')}({p.get('auction_gap','N/A')})")
            print(f"  │   {p.get('auction_detail', '')}")
            print(f"  │   {verdict}")
            if p.get("sell_plan_str"):
                print(f"  │   📉 {p['sell_plan_str']}")
        print(f"  └{'─'*50}")

    # 5. 发送邮件
    from quant.notify import EmailSender
    sender = EmailSender()

    # 构建邮件所需的市场概况
    market = data.get("market_summary", {})

    print(f"\n  📧 发送邮件...")
    ok = sender.send_recommendation(
        balanced_picks=balanced_final,
        aggressive_picks=aggressive_final,
        market_summary={
            "total": market.get("total_screened", "-"),
            "date": saved_date,
            "auction_time": beijing_now.strftime("%Y-%m-%d %H:%M"),
        },
    )

    if ok:
        print(f"  ✅ 邮件已发送 → {sender.receivers}")
        logger.info(f"✅ 邮件推送成功 → {sender.receivers}")
    else:
        print(f"  ❌ 邮件发送失败")
        logger.error("❌ 邮件发送失败")


# ═══════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════

def _build_reason(r5, r20, turnover, daily_pct, is_trending):
    """生成选股理由"""
    parts = []
    if is_trending:
        parts.append("🔥强趋势")
    if r5 > 0.03:
        parts.append("短期动量强")
    elif r5 < -0.02:
        parts.append("短期回调(机会)")
    if r20 > 0.05:
        parts.append("中期上行")
    if 2 < turnover < 15:
        parts.append("换手活跃")
    if abs(daily_pct) > 5:
        parts.append("波动较大")
    return " | ".join(parts) if parts else "综合评分领先"


def _copy_top(results, n):
    """复制前N条，去除内部字段"""
    top = []
    for r in results[:n]:
        clean = {k: v for k, v in r.items() if k not in ("agg_score", "close_val")}
        top.append(clean)
    return top


def _merge_auction(style_picks, all_auction_data):
    """将竞价数据合并回分组结果"""
    lookup = {p["symbol"]: p for p in all_auction_data}
    merged = []
    for p in style_picks:
        if p["symbol"] in lookup:
            auc = lookup[p["symbol"]]
            for key in ("auction_price", "auction_gap", "auction_score",
                        "auction_detail", "auction_verdict"):
                if key in auc:
                    p[key] = auc[key]
        merged.append(p)
    return merged


if __name__ == "__main__":
    from quant.utils.logger import setup_logger
    setup_logger()
    main()
