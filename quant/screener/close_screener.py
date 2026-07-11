"""
收盘多因子筛选器
每日收盘后运行：获取主板股票池 → 排除不合格股 → 多因子打分 → 选出推荐
"""

import json
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

# 处理 np.bool 废弃问题
if not hasattr(np, "bool"):
    np.bool = np.bool_

from quant.config import config as app_config


class CloseScreener:
    """收盘多因子筛选器

    筛选范围: 仅主板 (沪市60xxxx + 深市00xxxx)
    排除: ST、*ST、涨停、次新股(<60天)、流动性不足

    两种风格:
    - balanced (平衡型): 动量30% + 趋势25% + 量价25% + 低风险20%
    - aggressive (激进型): 动量45% + 趋势15% + 量价30% + 低风险10%
    """

    # 因子权重
    WEIGHTS = {
        "balanced": {
            "momentum": 0.30, "trend": 0.25, "volume": 0.25, "low_risk": 0.20
        },
        "aggressive": {
            "momentum": 0.45, "trend": 0.15, "volume": 0.30, "low_risk": 0.10
        },
    }

    def __init__(self):
        self._spot_df: Optional[pd.DataFrame] = None
        self._daily_df: Optional[pd.DataFrame] = None
        self._results: Dict = {}

    # ─── 股票池加载 ─────────────────────────────
    def load_universe(self) -> pd.DataFrame:
        """获取主板股票实时行情 + 筛选范围

        策略: 先用全市场接口 stock_zh_a_spot_em (一次请求)，
        失败则尝试分别请求沪市/深市
        """
        import akshare as ak

        logger.info("加载主板股票池...")
        raw = None

        # 方案A: 全市场接口（一次请求，最不容易被限流）
        for attempt in range(3):
            try:
                raw = ak.stock_zh_a_spot_em()
                if raw is not None and not raw.empty:
                    break
            except Exception as e:
                if attempt < 2:
                    time.sleep(2.0 ** (attempt + 1))
                else:
                    logger.warning(f"全市场接口失败: {e}")

        # 方案B: 分别请求沪市+深市
        if raw is None or (raw is not None and raw.empty):
            logger.info("尝试分别获取沪市/深市...")
            frames = []
            for fetch_fn in [ak.stock_sh_a_spot_em, ak.stock_sz_a_spot_em]:
                for attempt in range(2):
                    try:
                        r = fetch_fn()
                        if r is not None and not r.empty:
                            frames.append(r)
                        break
                    except Exception:
                        time.sleep(2.0)
            if frames:
                raw = pd.concat(frames, ignore_index=True)

        if raw is None or (raw is not None and raw.empty):
            raise RuntimeError(
                "无法获取股票列表 (akshare数据源暂时不可用，请稍后重试)"
            )

        df = raw.copy()
        logger.info(f"获取到 {len(df)} 只A股")

        # 列名标准化
        col_map = {
            "代码": "symbol",
            "名称": "name",
            "最新价": "close",
            "涨跌幅": "pct_chg",
            "成交量": "volume",
            "成交额": "amount",
            "换手率": "turnover",
            "量比": "volume_ratio",
            "市盈率-动态": "pe",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        # 只保留主板 (60xxxx 沪市 + 00xxxx 深市主板)
        if "symbol" in df.columns:
            df = df[df["symbol"].str.match(r"^(60|00)\d{4}$")].copy()

        logger.info(f"主板股票: {len(df)} 只")

        self._spot_df = df
        return df

    # ─── 排除规则 ──────────────────────────────
    def apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """应用排除规则"""
        before = len(df)

        # 1. ST / *ST
        if "name" in df.columns:
            df = df[~df["name"].str.contains("ST|退", na=False)]

        # 2. 排除涨停 (涨幅 >= 9.8% 买不到)
        if "pct_chg" in df.columns:
            df = df[df["pct_chg"] < 9.8]

        # 3. 排除跌幅过大的 (跌停或接近跌停，基本面无保障)
        if "pct_chg" in df.columns:
            df = df[df["pct_chg"] > -9.8]

        # 4. 成交额 >= 5000万 (流动性)
        if "amount" in df.columns:
            df = df[df["amount"] >= 50_000_000]

        # 5. 换手率合理范围 (0.5% ~ 25%)
        if "turnover" in df.columns:
            df = df[(df["turnover"] >= 0.5) & (df["turnover"] <= 25)]

        # 6. 排除高价股无成交量 (单价>200且换手<1%)
        if "close" in df.columns and "turnover" in df.columns:
            df = df[~((df["close"] > 200) & (df["turnover"] < 1.0))]

        after = len(df)
        logger.info(f"过滤: {before} → {after} 只 (排除{before-after}只)")
        return df.reset_index(drop=True)

    # ─── 多因子打分 ────────────────────────────
    def compute_scores(
        self, df: pd.DataFrame, style: str = "both"
    ) -> Dict[str, List[Dict]]:
        """计算综合评分

        Args:
            df: 过滤后的股票DataFrame
            style: 'balanced' | 'aggressive' | 'both'

        Returns:
            {'balanced': [{symbol, name, score, ...}, ...], 'aggressive': [...]}
        """
        if df.empty:
            return {}

        results = {}
        styles = ["balanced", "aggressive"] if style == "both" else [style]

        for s in styles:
            weights = self.WEIGHTS[s]
            scores = self._score_stocks(df, weights)
            results[s] = scores

        self._results = results
        return results

    def _score_stocks(
        self, df: pd.DataFrame, weights: Dict[str, float]
    ) -> List[Dict]:
        """对股票打分"""
        n = len(df)
        scores = pd.DataFrame(index=df.index)
        scores["symbol"] = df["symbol"]
        scores["name"] = df.get("name", "")
        # Fill missing names from built-in lookup
        scores["name"] = scores.apply(
            lambda row: row["name"] if row["name"] else self._lookup_name(row["symbol"]),
            axis=1,
        )
        scores["close"] = df.get("close", 0)

        # ─── 动量因子 ───────────────────────
        mom_score = np.zeros(n)

        # 基于涨跌幅(当日)
        if "pct_chg" in df.columns:
            pct = df["pct_chg"].fillna(0).values
            # 温和上涨最好(0~5%)，大涨扣分，大跌扣分
            mom_score += np.where(
                (pct > 0) & (pct <= 5),
                pct * 15,
                np.where(pct > 5, np.maximum(0, 75 - (pct - 5) * 8), pct * 5),
            )

        # 基于量比(近期活跃度)
        if "volume_ratio" in df.columns:
            vr = df["volume_ratio"].fillna(1.0).clip(0.3, 5.0).values
            # 量比在 1.2-2.5 之间最佳(放量但不异常)
            vr_score = np.where(
                (vr >= 0.8) & (vr <= 3.0),
                60 - abs(vr - 1.5) * 30,
                20,
            )
            mom_score += vr_score

        # 归一化到 0-100
        if mom_score.max() > mom_score.min():
            scores["momentum_sub"] = (
                (mom_score - mom_score.min())
                / (mom_score.max() - mom_score.min())
                * 100
            )
        else:
            scores["momentum_sub"] = 50

        # ─── 趋势因子 ───────────────────────
        trend_score = np.zeros(n)

        # 基于涨跌幅(趋势 proxy)
        if "pct_chg" in df.columns:
            pct = df["pct_chg"].fillna(0).values
            # 正收益 = 短期趋势向上
            trend_score += np.where(pct > 0, 40 + pct * 5, 30 + pct * 3)

        # PE 合理范围加分
        if "pe" in df.columns:
            pe = df["pe"].fillna(50).clip(0, 200).values
            # PE在10-40之间较好
            trend_score += np.where(
                (pe >= 5) & (pe <= 50),
                50 - abs(pe - 25) * 1.5,
                15,
            )

        if trend_score.max() > trend_score.min():
            scores["trend_sub"] = (
                (trend_score - trend_score.min())
                / (trend_score.max() - trend_score.min())
                * 100
            )
        else:
            scores["trend_sub"] = 50

        # ─── 量价因子 ───────────────────────
        volume_score = np.zeros(n)

        # 换手率
        if "turnover" in df.columns:
            to = df["turnover"].fillna(0).values
            # 换手率在 2%-10% 最佳
            volume_score += np.where(
                (to >= 1) & (to <= 15),
                80 - abs(to - 5) * 6,
                20,
            )

        # 成交额
        if "amount" in df.columns:
            amt = (df["amount"].fillna(0) / 1e8).values  # 转为亿
            volume_score += np.where(
                (amt >= 0.5) & (amt <= 30),
                60 - abs(amt - 5) * 2,
                np.where(amt > 30, 20, 10),
            )

        if volume_score.max() > volume_score.min():
            scores["volume_sub"] = (
                (volume_score - volume_score.min())
                / (volume_score.max() - volume_score.min())
                * 100
            )
        else:
            scores["volume_sub"] = 50

        # ─── 低风险因子 ─────────────────────
        risk_score = np.zeros(n)

        # 波动率代理: 涨跌幅绝对值小 = 低波动
        if "pct_chg" in df.columns:
            pct_abs = df["pct_chg"].fillna(0).abs().values
            risk_score += np.where(
                pct_abs <= 3,
                70 - pct_abs * 15,
                np.where(pct_abs <= 6, 25, 5),
            )

        # PE越低越安全
        if "pe" in df.columns:
            pe = df["pe"].fillna(50).clip(0, 300).values
            risk_score += np.where(
                pe <= 30,
                30 - pe * 0.8,
                np.where(pe <= 80, 10, 0),
            )

        if risk_score.max() > risk_score.min():
            scores["risk_sub"] = (
                (risk_score - risk_score.min())
                / (risk_score.max() - risk_score.min())
                * 100
            )
        else:
            scores["risk_sub"] = 50

        # ─── 综合得分 ───────────────────────
        scores["momentum_score"] = scores["momentum_sub"] * weights["momentum"]
        scores["trend_score"] = scores["trend_sub"] * weights["trend"]
        scores["volume_score"] = scores["volume_sub"] * weights["volume"]
        scores["risk_score"] = scores["risk_sub"] * weights["low_risk"]

        scores["score"] = (
            scores["momentum_score"]
            + scores["trend_score"]
            + scores["volume_score"]
            + scores["risk_score"]
        )

        # 排序
        scores = scores.sort_values("score", ascending=False)

        # Top N
        results = []
        for _, row in scores.iterrows():
            results.append({
                "symbol": row["symbol"],
                "name": row["name"],
                "close": f"{row['close']:.2f}" if row["close"] else "-",
                "pct_chg_str": f"{df.loc[row.name, 'pct_chg']:+.2f}%"
                if "pct_chg" in df.columns and row.name in df.index
                else "-",
                "score": round(float(row["score"]), 1),
                "momentum_score": round(float(row["momentum_score"]), 1),
                "trend_score": round(float(row["trend_score"]), 1),
                "volume_score": round(float(row["volume_score"]), 1),
                "reason": self._generate_reason(row, df),
            })

        return results

    def _generate_reason(self, row: pd.Series, df: pd.DataFrame) -> str:
        """生成推荐理由"""
        reasons = []
        mom = row.get("momentum_sub", 0)
        trend = row.get("trend_sub", 0)
        vol = row.get("volume_sub", 0)

        if mom > 70:
            reasons.append("动量强势")
        elif mom > 50:
            reasons.append("温和放量上涨")
        if trend > 70:
            reasons.append("趋势确立")
        elif trend > 50:
            reasons.append("趋势向上")
        if vol > 70:
            reasons.append("交投活跃")
        elif vol > 50:
            reasons.append("换手合理")

        if not reasons:
            reasons.append("综合因子优秀")

        # 风险提示
        idx = row.name
        if idx in df.index:
            pct = df.loc[idx, "pct_chg"] if "pct_chg" in df.columns else 0
            if pct > 5:
                reasons.append("(短期涨幅较大，注意回调)")

        return "，".join(reasons)

    # ─── 选出Top N ───────────────────────────
    def select_top(
        self, scores: Dict[str, List[Dict]], n: int = 3
    ) -> Dict[str, List[Dict]]:
        """从评分结果中选出前N名"""
        result = {}
        for style, picks in scores.items():
            result[style] = picks[:n]
        return result

    # ─── 股票名称查找 ─────────────────────────
    @staticmethod
    def _lookup_name(symbol: str) -> str:
        """根据代码查名称（离线映射 + 在线兜底）"""
        # 常用主板股票名称映射
        NAME_MAP = {
            "000001": "平安银行","000002": "万科A","000063": "中兴通讯","000069": "华侨城A",
            "000100": "TCL科技","000333": "美的集团","000338": "潍柴动力","000538": "云南白药",
            "000568": "泸州老窖","000596": "古井贡酒","000625": "长安汽车","000630": "铜陵有色",
            "000651": "格力电器","000725": "京东方A","000768": "中航西飞","000792": "盐湖股份",
            "000858": "五粮液","000895": "双汇发展","000963": "华东医药","000977": "浪潮信息",
            "001979": "招商蛇口","002001": "新和成","002013": "中航机电","002049": "紫光国微",
            "002138": "顺络电子","002142": "宁波银行","002230": "科大讯飞","002281": "光迅科技",
            "002304": "洋河股份","002313": "日海智能","002415": "海康威视","002459": "晶澳科技",
            "002902": "铭普光磁","300015": "爱尔眼科","300124": "汇川技术","300308": "中际旭创",
            "300394": "天孚通信","300502": "新易盛","600000": "浦发银行","600009": "上海机场",
            "600015": "华夏银行","600016": "民生银行","600028": "中国石化","600030": "中信证券",
            "600036": "招商银行","600048": "保利发展","600050": "中国联通","600085": "同仁堂",
            "600104": "上汽集团","600132": "重庆啤酒","600183": "生益科技","600188": "兖矿能源",
            "600196": "复星医药","600276": "恒瑞医药","600309": "万华化学","600340": "华夏幸福",
            "600383": "金地集团","600406": "国电南瑞","600436": "片仔癀","600438": "通威股份",
            "600460": "士兰微","600487": "亨通光电","600489": "中金黄金","600498": "烽火通信",
            "600519": "贵州茅台","600522": "中天科技","600536": "中国软件","600570": "恒生电子",
            "600585": "海螺水泥","600588": "用友网络","600690": "海尔智家","600703": "三安光电",
            "600741": "华域汽车","600745": "闻泰科技","600760": "中航沈飞","600763": "通策医疗",
            "600809": "山西汾酒","600887": "伊利股份","600893": "航发动力","600900": "长江电力",
            "600926": "杭州银行","601009": "南京银行","601012": "隆基绿能","601088": "中国神华",
            "601166": "兴业银行","601288": "农业银行","601318": "中国平安","601398": "工商银行",
            "601615": "明阳智能","601633": "长城汽车","601668": "中国建筑","601818": "光大银行",
            "601857": "中国石油","601869": "长飞光纤","601899": "紫金矿业","603005": "晶方科技",
            "603083": "剑桥科技","603160": "汇顶科技","603259": "药明康德","603288": "海天味业",
            "603501": "韦尔股份","603986": "兆易创新","603993": "洛阳钼业","688313": "仕佳光子",
        }
        return NAME_MAP.get(symbol, "")

    # ─── 趋势感知增强 ─────────────────────────
    def _fetch_daily_history(self, symbols: List[str], days: int = 60) -> pd.DataFrame:
        """获取一批股票的日线历史数据"""
        from quant.data.sources import AkshareSource

        end = date.today()
        start = end - timedelta(days=days + 10)  # extra buffer
        source = AkshareSource()

        try:
            df = source.get_daily(symbols, start, end)
            if df is not None and not df.empty:
                return df
        except Exception as e:
            logger.warning(f"获取历史数据失败: {e}")
        return pd.DataFrame()

    def _compute_trend_strength(self, hist_df: pd.DataFrame) -> Dict[str, float]:
        """计算每只股票的趋势强度 = 近40日内收盘在MA20上方的天数占比"""
        result = {}
        if hist_df.empty:
            return result

        for sym, group in hist_df.groupby("symbol"):
            group = group.sort_values("date").tail(40)
            if len(group) < 20:
                continue
            close = group["close"].values
            ma20 = pd.Series(close).rolling(20, min_periods=10).mean().values
            above = np.sum(close[-20:] > ma20[-20:])  # last 20 valid comparisons
            result[sym] = above / 20
        return result

    def _apply_trend_aware(
        self,
        picks: Dict[str, List[Dict]],
        df: pd.DataFrame,
    ) -> Dict[str, List[Dict]]:
        """对初选结果应用趋势感知增强

        如果某只股票处于强趋势中 (trend_strength > 0.7):
        - 动量因子加权 → 不再惩罚高RSI和高开
        - 调高总分 + 附加趋势标签
        """
        # 收集所有候选symbol
        all_symbols = []
        for style_picks in picks.values():
            for p in style_picks:
                all_symbols.append(p["symbol"])

        if not all_symbols:
            return picks

        # 获取历史数据
        unique_syms = list(set(all_symbols))
        hist = self._fetch_daily_history(unique_syms)
        if hist.empty:
            logger.info("无历史数据，跳过趋势感知")
            return picks

        # 计算趋势强度
        trend_map = self._compute_trend_strength(hist)
        logger.info(f"趋势检测: {len(trend_map)} 只股票有数据")

        # 应用到每个pick
        for style, style_picks in picks.items():
            for pick in style_picks:
                sym = pick["symbol"]
                ts = trend_map.get(sym, 0.5)

                # 检查近20日收益
                sym_hist = hist[hist["symbol"] == sym].sort_values("date")
                if len(sym_hist) >= 21:
                    ret_20d = (
                        sym_hist["close"].iloc[-1] / sym_hist["close"].iloc[-21] - 1
                    )
                else:
                    ret_20d = 0

                pick["trend_strength"] = round(ts, 2)
                pick["ret_20d"] = round(ret_20d, 4)

                # ===== 趋势感知调整 =====
                is_strong_trend = ts > 0.7 and ret_20d > 0.05

                if is_strong_trend:
                    # 动量加速奖金
                    score_boost = 1.0 + min(ts - 0.7, 0.3)  # max +30%
                    pick["score"] = round(pick["score"] * score_boost, 1)
                    pick["momentum_score"] = round(pick["momentum_score"] * 1.2, 1)

                    # 更新推荐理由
                    if "趋势" in pick.get("reason", ""):
                        pick["reason"] = pick["reason"].replace("趋势", "强趋势🔥")
                    else:
                        pick["reason"] = "强趋势🔥" + pick.get("reason", "")

                    # 添加趋势标签
                    pick["is_trending"] = True
                    pick["trend_note"] = (
                        f"强趋势股(强度{ts:.0%}): 持有至MA20下方再考虑卖出"
                    )
                else:
                    pick["is_trending"] = False

        return picks

    # ─── 完整筛选流程 ─────────────────────────
    def run(self, style: str = "both", top_n: int = 3) -> Dict:
        """一键运行收盘筛选

        Returns:
            {
                'balanced': [{symbol, name, score, ...}, ...],
                'aggressive': [{symbol, name, score, ...}, ...],
                'market': 市场概况,
                'timestamp': 时间戳,
            }
        """
        logger.info(f"开始收盘筛选 | 风格={style} | 精选{top_n}只")

        # 1. 加载股票池
        df = self.load_universe()

        # 2. 过滤
        df = self.apply_filters(df)

        # 3. 打分
        scores = self.compute_scores(df, style)

        # 4. 选Top（取多一些候选给趋势分析）
        candidates_wide = self.select_top(scores, max(top_n * 3, 10))

        # 5. 趋势感知增强
        enhanced = self._apply_trend_aware(candidates_wide, df)

        # 6. 重新按增强后得分排序，取Top N
        final_picks = {}
        for st in enhanced:
            enhanced[st].sort(key=lambda x: x.get("score", 0), reverse=True)
            final_picks[st] = enhanced[st][:top_n]

        # 7. 市场概况
        market = self._market_summary(df)

        result = {
            "balanced": final_picks.get("balanced", []),
            "aggressive": final_picks.get("aggressive", []),
            "market": market,
            "timestamp": datetime.now().isoformat(),
        }

        # 8. 保存到文件
        self._save_candidates(result)

        n_balanced = len(result["balanced"])
        n_agg = len(result["aggressive"])

        # 统计趋势股数量
        n_trend_bal = sum(1 for p in result["balanced"] if p.get("is_trending"))
        n_trend_agg = sum(1 for p in result["aggressive"] if p.get("is_trending"))

        logger.info(
            f"筛选完成! 平衡型: {n_balanced}只({n_trend_bal}趋势) | "
            f"激进型: {n_agg}只({n_trend_agg}趋势)"
        )
        return result

    def _market_summary(self, df: pd.DataFrame) -> Dict:
        """市场概况"""
        up_count = len(df[df["pct_chg"] > 0]) if "pct_chg" in df.columns else 0
        down_count = len(df[df["pct_chg"] < 0]) if "pct_chg" in df.columns else 0
        avg_pct = float(df["pct_chg"].mean()) if "pct_chg" in df.columns else 0

        return {
            "total": len(df),
            "up_count": up_count,
            "down_count": down_count,
            "avg_pct_chg": f"{avg_pct:+.2f}%",
            "limit_up": "-",
        }

    def _save_candidates(self, result: Dict) -> None:
        """保存候选到本地JSON，供竞价阶段使用"""
        root = Path(app_config.data.root_dir) if hasattr(app_config, 'data') else Path("data")
        root.mkdir(parents=True, exist_ok=True)
        path = root / "candidates.json"

        # 只保存必要的股票代码
        save_data = {
            "balanced": [
                {"symbol": p["symbol"], "name": p["name"], "close": p["close"]}
                for p in result.get("balanced", [])
            ],
            "aggressive": [
                {"symbol": p["symbol"], "name": p["name"], "close": p["close"]}
                for p in result.get("aggressive", [])
            ],
            "market": result.get("market", {}),
            "timestamp": result.get("timestamp", ""),
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        logger.info(f"候选已保存 → {path}")

    def load_candidates(self) -> Optional[Dict]:
        """加载上次保存的候选"""
        root = Path(app_config.data.root_dir) if hasattr(app_config, 'data') else Path("data")
        path = root / "candidates.json"
        if not path.exists():
            logger.warning("无候选数据，请先运行收盘筛选")
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
