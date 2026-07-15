#!/usr/bin/env python3
"""
羊量每日选股 — Web 控制台
启动: python web_app.py
访问: http://localhost:5888
"""

import sys, io, json, time, threading
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from flask import Flask, request, jsonify, Response
import numpy as np
import pandas as pd

app = Flask(__name__)

# ─── 股票池（动态加载全市场，自动缓存）──────────────────
_STOCK_POOL_CACHE = None
_NAME_MAP_CACHE = None

def load_stock_pool(sample_size=300):
    """从全A股加载主板股票池（过滤ST，系统采样，自动缓存）"""
    global _STOCK_POOL_CACHE, _NAME_MAP_CACHE
    cache_file = Path("data/stock_pool_cache.json")

    # 内存缓存命中
    if _STOCK_POOL_CACHE and _NAME_MAP_CACHE and len(_STOCK_POOL_CACHE) == sample_size:
        return _STOCK_POOL_CACHE, _NAME_MAP_CACHE

    # 文件缓存命中
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if len(data.get("pool", [])) == sample_size:
                _STOCK_POOL_CACHE = data["pool"]
                _NAME_MAP_CACHE = data["name_map"]
                return _STOCK_POOL_CACHE, _NAME_MAP_CACHE
        except Exception:
            pass

    import akshare as ak
    info = ak.stock_info_a_code_name()
    codes = info["code"].astype(str)
    names = info["name"].astype(str)

    # 过滤主板
    mask = codes.str.match(r"^(60|00)\d{4}$")
    # 排除 ST
    mask &= ~names.str.contains(r"\*?ST", na=True, case=True)
    # 排除 N/C 开头新股标识
    mask &= ~names.str.match(r"^[NC]", na=True)

    valid = info[mask].copy()
    valid = valid.sort_values("code")

    # 系统采样（均匀覆盖所有板块）
    step = max(1, len(valid) // sample_size)
    sampled = valid.iloc[::step].head(sample_size)

    pool = sampled["code"].tolist()
    name_map = dict(zip(sampled["code"], sampled["name"]))

    # 缓存
    Path("data").mkdir(exist_ok=True)
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump({"pool": pool, "name_map": name_map}, f, ensure_ascii=False, indent=2)

    _STOCK_POOL_CACHE = pool
    _NAME_MAP_CACHE = name_map
    return pool, name_map

# ─── 日志流 ──────────────────────────────────────
_log_streams = {}  # job_id → io.StringIO

def get_stream(job_id):
    if job_id not in _log_streams:
        _log_streams[job_id] = io.StringIO()
    return _log_streams[job_id]

def log(job_id, msg):
    s = get_stream(job_id)
    ts = datetime.now().strftime("%H:%M:%S")
    s.write(f"[{ts}] {msg}\n")

# ─── 并行下载器 ──────────────────────────────────

def parallel_download(stock_pool, name_map, start, end, job_id=None, workers=40):
    """多线程并行下载 — 线程本地Session + 温和限速"""
    import requests as req
    import random

    TENCENT_URL = "https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get"
    COLUMNS = ["date","open","close","high","low","volume","_extra","pct_chg","amount","_extra2"]
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://gu.qq.com/",
    }

    # 线程本地 Session
    _local = threading.local()
    def get_session():
        if not hasattr(_local, "session"):
            _local.session = req.Session()
            _local.session.headers.update(HEADERS)
        return _local.session

    def download_one(sym):
        code = f"sh{sym}" if sym.startswith(("6","9")) else f"sz{sym}"
        param = f"{code},day,{start.strftime('%Y-%m-%d')},{end.strftime('%Y-%m-%d')},640,qfq"
        try:
            s = get_session()
            r = s.get(TENCENT_URL, params={"param": param}, timeout=8)
            r.raise_for_status()
            data = r.json()
            if data.get("code") != 0:
                return pd.DataFrame()
            stock_data = data.get("data", {}).get(code, {})
            days = stock_data.get("qfqday") or stock_data.get("newqfqday") or stock_data.get("day") or []
            if not days:
                return pd.DataFrame()
            # 处理数据行：取前6列 (date,open,close,high,low,volume)
            rows = []
            for row in days:
                rows.append(row[:6])
            df = pd.DataFrame(rows, columns=COLUMNS[:6])
            df["symbol"] = sym
            df["date"] = pd.to_datetime(df["date"]).dt.date
            for c in ["open","close","high","low","volume"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df["amount"] = df["volume"] * df["close"]
            df["turnover"] = 3.0
            # 温和限速：每请求5-20ms延迟
            time.sleep(random.uniform(0.005, 0.02))
            return df[["symbol","date","open","high","low","close","volume","amount","turnover"]]
        except Exception:
            return pd.DataFrame()

    all_frames = []
    total = len(stock_pool)
    failed = 0
    error_count = 0
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(download_one, sym): sym for sym in stock_pool}
        done = 0
        for future in as_completed(futures):
            done += 1
            try:
                df = future.result(timeout=8)
                if not df.empty:
                    all_frames.append(df)
                else:
                    failed += 1
            except Exception:
                failed += 1
                error_count += 1
            if job_id and done % 50 == 0:
                elapsed = time.time() - t_start
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / rate if rate > 0 else 0
                log(job_id, f"  📥 下载: {done}/{total} ({rate:.1f}只/秒, ETA {eta:.0f}秒, 成功{done-failed})")

    elapsed = time.time() - t_start
    if job_id:
        log(job_id, f"  📥 下载完成: {total-failed}/{total}, {elapsed:.0f}秒 ({total/elapsed:.1f}只/秒)" + (f", {error_count}网络错误" if error_count else ""))

    if not all_frames:
        return pd.DataFrame(), "N/A"

    result = pd.concat(all_frames, ignore_index=True)
    result = result.sort_values(["symbol", "date"])
    return result, result["date"].max()


# ─── 核心逻辑 ────────────────────────────────────

def do_close_screening(job_id, send_email=False, sample_size=500):
    """收盘多因子筛选 — 全市场采样 + 共享Session极速下载"""
    log(job_id, "📊 开始收盘筛选...")
    today = date.today()

    # 加载全市场股票池（系统采样）
    log(job_id, "📋 加载全A股股票池...")
    stock_pool, name_map = load_stock_pool(sample_size=sample_size)
    log(job_id, f"✅ 股票池: {len(stock_pool)} 只主板（全市场均匀采样）")

    # 快速下载（共享Session直连腾讯API, 40线程）
    start = today - timedelta(days=25)
    log(job_id, f"📡 并行下载 {len(stock_pool)} 只历史数据 (40线程, 共享连接池)...")
    df, latest_date = parallel_download(stock_pool, name_map, start, today, job_id=job_id)

    if df.empty:
        log(job_id, "❌ 数据下载失败")
        return None

    log(job_id, f"📊 数据就绪: {len(df)}行, {df['symbol'].nunique()}只有效")

    # 按股票分组打分（groupby O(n)）
    results = []
    pool_set = set(stock_pool)
    grouped = df.groupby("symbol")
    processed = 0

    for sym, sym_data in grouped:
        if sym not in pool_set:
            continue
        processed += 1

        sym_data = sym_data.sort_values("date")
        if len(sym_data) < 3:
            continue

        close_v = sym_data["close"].values.astype(float)
        close = close_v[-1]
        prev = close_v[-2] if len(close_v) >= 2 else close
        daily_pct = (close / prev - 1) * 100

        r5 = close / close_v[max(0, len(close_v)-6)] - 1
        r20 = close / close_v[max(0, len(close_v)-21)] - 1

        turnover = float(sym_data.iloc[-1].get("turnover", 3) or 3)

        if len(close_v) >= 20:
            ma20 = pd.Series(close_v).rolling(20, min_periods=10).mean().values
            trend_strength = np.sum(close_v[-20:] > ma20[-20:]) / 20
        else:
            trend_strength = 0.5

        mom = min(100, max(0, 50 + r5 * 400)) * 0.5 + min(100, max(0, 50 + r20 * 200)) * 0.5
        trend = min(100, max(0, 50 + r20 * 300))
        to_s = min(100, max(5, 40 if 2 < turnover < 15 else 15))
        vol_q = to_s * 0.6 + 40
        risk = min(100, max(0, 70 - abs(daily_pct) * 10))
        score = mom * 0.30 + trend * 0.25 + vol_q * 0.25 + risk * 0.20

        is_trending = bool(trend_strength > 0.7 and r20 > 0.05)
        if is_trending:
            score *= 1.15

        name = name_map.get(sym, sym)
        results.append({
            "symbol": sym, "name": name,
            "close": f"{close:.2f}", "close_val": float(close),
            "score": round(float(score), 1),
            "momentum_score": round(float(mom), 1), "trend_score": round(float(trend), 1),
            "volume_score": round(float(vol_q), 1),
            "pct_chg": f"{daily_pct:+.2f}%",
            "r5": f"{float(r5):+.1%}", "r20": f"{float(r20):+.1%}",
            "turnover": round(float(turnover), 1),
            "is_trending": is_trending,
            "trend_strength": round(float(trend_strength) * 100),
            "reason": _reason(r5, r20, turnover, is_trending),
        })

        if processed % 100 == 0:
            log(job_id, f"  📊 打分: {processed}只")

    log(job_id, f"  📊 打分完成: {len(results)}只有效")

    # 排名
    results.sort(key=lambda x: x["score"], reverse=True)
    balanced = _copy_top(results, 3)

    for r in results:
        r["agg"] = r["score"] * 1.10 if r["is_trending"] else \
            r["momentum_score"] * 0.45 + r["trend_score"] * 0.15 + r["volume_score"] * 0.30 + r["score"] * 0.10
    results.sort(key=lambda x: x.get("agg", 0), reverse=True)
    aggressive = _copy_top(results, 3)

    # 卖出计划
    from quant.screener.sell_advisor import SellAdvisor
    advisor = SellAdvisor()
    for picks, style in [(balanced, "balanced"), (aggressive, "aggressive")]:
        for p in picks:
            plan = advisor.generate(float(p["close"]), style=style)
            p["sell_plan"] = plan
            p["sell_plan_str"] = (
                f"止损:{plan['stop_loss']}({plan['stop_loss_pct']}) | "
                f"止盈1:{plan['take_profit_1']}({plan['take_profit_1_pct']}) | "
                f"持有≤{plan['max_hold_days']}天"
            )

    # 保存
    output = {
        "balanced": balanced, "aggressive": aggressive,
        "date": str(today),
        "saved_at": datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M"),
        "market_summary": {"total_screened": len(results), "latest_data_date": str(latest_date)},
    }
    Path("data").mkdir(exist_ok=True)
    with open("data/stable_result.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    log(job_id, f"✅ 收盘筛选完成 → 平衡型3只 + 激进型3只")

    # 可选：发邮件
    if send_email:
        log(job_id, "📧 发送邮件...")
        from quant.notify import EmailSender
        sender = EmailSender()
        ok = sender.send_recommendation(
            balanced_picks=balanced,
            aggressive_picks=aggressive,
            market_summary=output["market_summary"],
        )
        if ok:
            log(job_id, f"✅ 邮件已发送 → {sender.receivers}")
            output["email_sent"] = True
        else:
            log(job_id, "❌ 邮件发送失败")
            output["email_sent"] = False

    return output


def do_auction_analysis(job_id):
    """竞价分析 + 发邮件"""
    log(job_id, "📈 开始竞价分析...")

    path = Path("data/stable_result.json")
    if not path.exists():
        log(job_id, "❌ 无昨日数据，请先运行收盘筛选")
        return None

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    saved_date = data.get("date", "未知")
    log(job_id, f"📂 加载昨日候选 ({saved_date})")

    # 去重
    all_picks = data.get("balanced", []) + data.get("aggressive", [])
    seen = set()
    unique = []
    for p in all_picks:
        if p["symbol"] not in seen:
            seen.add(p["symbol"])
            unique.append(p)

    log(job_id, f"🔍 获取 {len(unique)} 只竞价数据...")

    import akshare as ak
    auction_data = []
    for p in unique:
        sym = p["symbol"]
        try:
            auc = ak.stock_zh_a_hist_pre_min_em(sym, "09:15:00", "09:25:00")
            if auc is not None and not auc.empty:
                prices = auc["收盘"].values.astype(float) if "收盘" in auc.columns else None
                volumes = auc["成交量"].values.astype(float) if "成交量" in auc.columns else None

                if prices is not None and len(prices) >= 2:
                    prev_close = float(p.get("close_val", p.get("close", 0)))
                    late_prices = prices[-5:] if len(prices) >= 5 else prices

                    # 价格趋势分
                    if len(late_prices) >= 2:
                        slope = (late_prices[-1] - late_prices[0]) / late_prices[0] * 100
                        price_score = min(100, max(0, 50 + slope * 30))
                    else:
                        price_score = 50

                    # 量能分
                    if volumes is not None and len(volumes) >= 5:
                        late_vol = volumes[-5:].sum()
                        total_vol = volumes.sum()
                        vol_ratio = late_vol / total_vol if total_vol > 0 else 0.5
                        vol_score = min(100, max(0, vol_ratio * 200))
                    else:
                        vol_score = 50

                    # 高开幅度分
                    auction_price = late_prices[-1]
                    gap_pct = (auction_price / prev_close - 1) * 100 if prev_close > 0 else 0
                    if 2 <= gap_pct <= 5:
                        gap_score = 90
                    elif 0 <= gap_pct < 2:
                        gap_score = 65
                    elif 5 < gap_pct <= 7:
                        gap_score = 50
                    elif gap_pct > 7:
                        gap_score = 20
                    else:
                        gap_score = 30

                    auc_score = price_score * 0.35 + vol_score * 0.30 + gap_score * 0.35
                    if auc_score >= 70:
                        verdict = "✅ 竞价强势"
                    elif auc_score >= 55:
                        verdict = "⚠️ 竞价一般"
                    else:
                        verdict = "❌ 竞价偏弱"

                    auction_data.append({
                        "symbol": sym,
                        "auction_price": f"{auction_price:.2f}",
                        "auction_gap": f"{gap_pct:+.2f}%",
                        "auction_score": round(auc_score, 1),
                        "auction_detail": f"竞价:{auction_price:.2f}({gap_pct:+.2f}%) 价格:{price_score:.0f} 量能:{vol_score:.0f} 高开:{gap_score:.0f}",
                        "auction_verdict": verdict,
                    })
                else:
                    auction_data.append({"symbol": sym, "auction_gap": "N/A", "auction_score": 0, "auction_detail": "数据不足", "auction_verdict": "⚠️ 数据不足"})
            else:
                auction_data.append({"symbol": sym, "auction_gap": "N/A", "auction_score": 0, "auction_detail": "无数据", "auction_verdict": "⏳ 数据未出"})
        except Exception as e:
            auction_data.append({"symbol": sym, "auction_gap": "N/A", "auction_score": 0, "auction_detail": str(e)[:50], "auction_verdict": "⚠️ 获取失败"})
            log(job_id, f"  ⚠ {sym} 竞价获取失败")
        time.sleep(0.3)

    # 合并
    auc_lookup = {a["symbol"]: a for a in auction_data}
    balanced_final = _merge_auction(data.get("balanced", []), auc_lookup)
    aggressive_final = _merge_auction(data.get("aggressive", []), auc_lookup)

    log(job_id, "📧 发送邮件...")
    from quant.notify import EmailSender
    sender = EmailSender()
    ok = sender.send_recommendation(
        balanced_picks=balanced_final,
        aggressive_picks=aggressive_final,
        market_summary={"total": len(unique), "date": saved_date},
    )

    if ok:
        log(job_id, f"✅ 邮件已发送 → {sender.receivers}")
    else:
        log(job_id, "❌ 邮件发送失败")

    return {"balanced": balanced_final, "aggressive": aggressive_final, "email_sent": ok}


# ─── 工具函数 ────────────────────────────────────

def _reason(r5, r20, turnover, is_trending):
    parts = []
    if is_trending: parts.append("🔥强趋势")
    if r5 > 0.03: parts.append("短期动量强")
    elif r5 < -0.02: parts.append("短期回调")
    if r20 > 0.05: parts.append("中期上行")
    if 2 < turnover < 15: parts.append("换手活跃")
    return " | ".join(parts) if parts else "综合评分领先"

def _copy_top(results, n):
    return [{k: v for k, v in r.items() if k not in ("agg", "close_val")} for r in results[:n]]

def _merge_auction(picks, lookup):
    for p in picks:
        if p["symbol"] in lookup:
            auc = lookup[p["symbol"]]
            for k in ("auction_price","auction_gap","auction_score","auction_detail","auction_verdict"):
                if k in auc: p[k] = auc[k]
    return picks


# ─── Flask 路由 ──────────────────────────────────

@app.route("/")
def index():
    return HTML_PAGE

@app.route("/api/status")
def api_status():
    """返回当前联动状态"""
    path = Path("data/stable_result.json")
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
            saved_date = d.get("date", "")
            balanced = d.get("balanced", [])
            aggressive = d.get("aggressive", [])
            return jsonify({
                "ready": True,
                "date": saved_date,
                "count": len(balanced) + len(aggressive),
                "preview": [
                    {"symbol": p["symbol"], "name": p["name"], "score": p["score"], "close": p["close"]}
                    for p in (balanced + aggressive)[:6]
                ]
            })
        except Exception:
            pass
    return jsonify({"ready": False})


@app.route("/api/close", methods=["POST"])
def api_close():
    job_id = f"close_{int(time.time())}"
    data = request.get_json(silent=True) or {}
    send_email = data.get("send_email", False)
    sample_size = int(data.get("sample_size", 500))

    log(job_id, f"🚀 启动收盘筛选 (样本={sample_size})...")

    try:
        result = do_close_screening(job_id, send_email=send_email, sample_size=sample_size)
        logs = get_stream(job_id).getvalue()
        return jsonify({"ok": True, "logs": logs, "result": result})
    except Exception as e:
        log(job_id, f"❌ 错误: {traceback.format_exc()}")
        logs = get_stream(job_id).getvalue()
        return jsonify({"ok": False, "logs": logs, "error": str(e)})


@app.route("/api/auction", methods=["POST"])
def api_auction():
    job_id = f"auc_{int(time.time())}"
    log(job_id, "🚀 启动竞价分析...")

    try:
        result = do_auction_analysis(job_id)
        logs = get_stream(job_id).getvalue()
        return jsonify({"ok": True, "logs": logs, "result": result})
    except Exception as e:
        log(job_id, f"❌ 错误: {traceback.format_exc()}")
        logs = get_stream(job_id).getvalue()
        return jsonify({"ok": False, "logs": logs, "error": str(e)})


# ─── HTML 页面 ───────────────────────────────────

HTML_PAGE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>🐑 羊量每日选股</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#f5f6fa;color:#2c3e50;min-height:100vh}
.header{background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;padding:20px 16px;text-align:center}
.header h1{font-size:22px;margin-bottom:4px}
.header p{font-size:13px;opacity:.8}
.btns{display:flex;gap:12px;padding:20px 16px;max-width:500px;margin:0 auto;flex-wrap:wrap}
.btn{flex:1;min-width:140px;padding:16px 20px;border:none;border-radius:12px;font-size:15px;font-weight:600;cursor:pointer;transition:.2s;color:#fff}
.btn:disabled{opacity:.5;cursor:not-allowed}
.btn-close{background:linear-gradient(135deg,#f39c12,#e67e22)}
.btn-close:hover:not(:disabled){box-shadow:0 4px 15px rgba(243,156,18,.4)}
.btn-auction{background:linear-gradient(135deg,#e74c3c,#c0392b)}
.btn-auction:hover:not(:disabled){box-shadow:0 4px 15px rgba(231,76,60,.4)}
.logs{max-width:700px;margin:0 auto 16px;padding:12px 16px;background:#1e1e1e;border-radius:10px;color:#0f0;font-family:Consolas,monospace;font-size:12px;max-height:200px;overflow-y:auto;white-space:pre-wrap;display:none}
.logs.show{display:block}
.results{max-width:700px;margin:0 auto;padding:0 16px 40px}
.card{background:#fff;border-radius:12px;padding:16px;margin-bottom:12px;box-shadow:0 2px 8px rgba(0,0,0,.06)}
.card .title{font-size:16px;font-weight:700;margin-bottom:8px;padding-bottom:8px;border-bottom:2px solid #f0f0f0}
.card .title.balanced{color:#667eea;border-color:#667eea}
.card .title.aggressive{color:#e74c3c;border-color:#e74c3c}
.stock{display:flex;align-items:center;gap:10px;padding:10px 8px;border-bottom:1px solid #f5f5f5}
.stock:last-child{border-bottom:none}
.stock .medal{font-size:20px}
.stock .info{flex:1}
.stock .name{font-size:15px;font-weight:600}
.stock .code{font-size:11px;color:#999}
.stock .scores{font-size:12px;color:#666;margin-top:4px}
.stock .reason{font-size:11px;color:#888;margin-top:2px}
.stock .plan{font-size:11px;color:#e67e22;margin-top:2px}
.stock .auction{margin-top:4px;padding:4px 8px;border-radius:6px;font-size:12px;display:inline-block}
.auction-strong{background:#d4edda;color:#155724}
.auction-warn{background:#fff3cd;color:#856404}
.auction-weak{background:#f8d7da;color:#721c24}
.status{margin-top:4px;font-size:12px}
.spinner{display:inline-block;width:14px;height:14px;border:2px solid #fff;border-top-color:transparent;border-radius:50%;animation:spin .8s linear infinite;vertical-align:middle;margin-right:6px}
@keyframes spin{to{transform:rotate(360deg)}}
.tip{text-align:center;color:#999;font-size:12px;padding:10px}
</style>
</head>
<body>

<div class="header">
  <h1>🐑 羊量每日选股</h1>
  <p>收盘多因子筛选 + 集合竞价验证</p>
</div>

<div id="statusBar" style="max-width:700px;margin:12px auto 0;padding:10px 16px;background:#fff3cd;border-radius:8px;font-size:13px;text-align:center;display:none"></div>

<div class="btns">
  <button class="btn btn-close" id="btnClose" onclick="runClose()">
    📊 收盘筛选
  </button>
  <button class="btn btn-auction" id="btnAuction" onclick="runAuction()">
    📈 竞价分析+推送
  </button>
</div>

<div class="options" style="max-width:700px;margin:0 auto 12px;padding:0 16px;display:flex;gap:16px;align-items:center;flex-wrap:wrap;font-size:13px;color:#666">
  <label>📦 股票数量:
    <select id="sampleSize" style="padding:4px 8px;border-radius:6px;border:1px solid #ddd">
      <option value="200">200只 (~5秒)</option>
      <option value="500" selected>500只 (~12秒)</option>
      <option value="800">800只 (~20秒)</option>
      <option value="1200">1200只 (~30秒)</option>
    </select>
  </label>
  <label>📧 <input type="checkbox" id="sendEmail"> 收盘后同时推送到邮箱</label>
</div>

<div class="logs" id="logs"></div>
<div class="results" id="results"></div>
<div class="tip">⏰ 收盘后 (15:30) 点左边 | 次日 9:15-9:25 点右边 | 勾选📧两个按钮都会发邮件</div>

<script>
// 页面加载时检查联动状态
(async function checkStatus(){
  try{
    const r=await fetch('/api/status',{headers:{'ngrok-skip-browser-warning':'true'}});
    const d=await r.json();
    const el=document.getElementById('statusBar');
    if(d.ready){
      el.style.display='block';
      el.style.background='#d4edda';
      el.innerHTML='✅ 已有收盘数据 ('+d.date+') — '+d.count+'只候选就绪，可直接点【竞价分析+推送】';
    } else {
      el.style.display='block';
      el.style.background='#fff3cd';
      el.innerHTML='⏳ 尚未运行收盘筛选 — 请先点【收盘筛选】获取候选股票';
    }
  }catch(e){}
})();

function showLogs(text) {const el=document.getElementById('logs');el.textContent=text;el.classList.add('show');el.scrollTop=el.scrollHeight;}

async function runClose() {
  const btn=document.getElementById('btnClose');
  btn.disabled=true;
  btn.innerHTML='<span class="spinner"></span>筛选中...';
  document.getElementById('results').innerHTML='';
  showLogs('⏳ 并行下载中...请耐心等待');
  const sendEmail=document.getElementById('sendEmail').checked;
  const sampleSize=document.getElementById('sampleSize').value;

  // 30秒超时（共享Session直连, 1200只~30秒）
  const ctrl=new AbortController();
  const timer=setTimeout(()=>ctrl.abort(),30000);

  try {
    const resp=await fetch('/api/close',{
      method:'POST',
      headers:{'Content-Type':'application/json','ngrok-skip-browser-warning':'true'},
      body:JSON.stringify({send_email:sendEmail,sample_size:parseInt(sampleSize)}),
      signal:ctrl.signal
    });
    clearTimeout(timer);
    if(!resp.ok){showLogs('Server error: '+resp.status);return;}
    const data=await resp.json();
    showLogs(data.logs);
    if(data.ok&&data.result)renderResults(data.result);
    if(data.result&&data.result.email_sent)showLogs(data.logs+'\\n✅ 邮件已发送到两个邮箱');
  }catch(e){
    clearTimeout(timer);
    if(e.name==='AbortError')showLogs('⏰ 请求超时 (30秒)。请减少股票数量或检查网络。');
    else showLogs('❌ 错误: '+e.message);
  }
  btn.disabled=false;
  btn.innerHTML='📊 收盘筛选';
  setTimeout(async()=>{
    const r=await fetch('/api/status',{headers:{'ngrok-skip-browser-warning':'true'}});
    const d=await r.json();
    const el=document.getElementById('statusBar');
    if(d.ready){el.style.display='block';el.style.background='#d4edda';el.innerHTML='✅ 收盘数据就绪 ('+d.date+') — '+d.count+'只候选';}
  },1000);
}

async function runAuction() {
  const btn=document.getElementById('btnAuction');
  btn.disabled=true;
  btn.innerHTML='<span class="spinner"></span>竞价分析中...';
  document.getElementById('results').innerHTML='';
  showLogs('⏳ 获取竞价数据...\\n⚠ 竞价数据仅在 9:15-9:25 可用');

  try {
    const resp=await fetch('/api/auction',{method:'POST',headers:{'ngrok-skip-browser-warning':'true'}});
    const data=await resp.json();
    showLogs(data.logs);
    if(data.ok&&data.result)renderResults(data.result);
    if(data.result&&data.result.email_sent!==false)showLogs(data.logs+'\\n✅ 邮件已发送到两个邮箱');
  }catch(e){showLogs('Error: '+e.message);}
  btn.disabled=false;
  btn.innerHTML='📈 竞价分析+推送';
}

function renderResults(data) {
  let html='';
  ['balanced','aggressive'].forEach((style,si)=>{
    const picks=data[style];
    if(!picks||!picks.length)return;
    const title=style==='balanced'?'🥢 平衡型（稳健持仓）':'🔥 激进型（追求收益）';
    const cls=style==='balanced'?'balanced':'aggressive';
    html+=`<div class="card"><div class="title ${cls}">${title}</div>`;
    picks.forEach((p,i)=>{
      const medal=['🥇','🥈','🥉'][i];
      const aucScore=p.auction_score||0;
      let aucClass='',aucBadge='';
      if(p.auction_verdict){if(p.auction_verdict.includes('强势')){aucClass='auction-strong';aucBadge='✅'}else if(p.auction_verdict.includes('一般')||p.auction_verdict.includes('谨慎')){aucClass='auction-warn';aucBadge='⚠️'}else{aucClass='auction-weak';aucBadge='❌'}}
      html+=`<div class="stock"><div class="medal">${medal}</div><div class="info"><div class="name">${p.symbol} ${p.name} ${p.is_trending?'🔥':''}</div><div class="scores">收盘:${p.close} | 涨跌:${p.pct_chg||'-'} | 综合:${p.score} | 动量:${p.momentum_score} | 趋势:${p.trend_score} | 量价:${p.volume_score}</div><div class="reason">${p.reason||''}</div>${p.sell_plan_str?`<div class="plan">📉 ${p.sell_plan_str}</div>`:''}${p.auction_verdict?`<div class="status"><span class="auction ${aucClass}">${aucBadge} 竞价:${p.auction_score||'-'}分 ${p.auction_gap||'-'} ${p.auction_verdict}</span></div>`:''}</div></div>`;
    });
    html+='</div>';
  });
  if(data.email_sent)html+='<div style="text-align:center;color:#27ae60;padding:12px;font-size:14px">✅ 邮件已发送到 1281074210@qq.com + 1277304115@qq.com</div>';
  document.getElementById('results').innerHTML=html;
}
</script>
</body>
</html>
"""


if __name__ == "__main__":
    import os
    from quant.utils.logger import setup_logger
    setup_logger()

    port = int(os.environ.get("PORT", 5888))
    print("\n" + "=" * 50)
    print("  🐑 羊量 Web 控制台")
    print(f"  http://0.0.0.0:{port}")
    print("=" * 50)
    print("  收盘后点 → 📊 收盘筛选")
    print("  开盘前点 → 📈 竞价分析+推送")
    print("=" * 50 + "\n")

    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
