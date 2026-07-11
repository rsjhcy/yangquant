"""
因子分析工具
IC分析 / 分层回测 / 因子相关性 / 因子衰减
"""

from datetime import date
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats


class FactorAnalyzer:
    """因子分析器

    提供:
    - IC 分析 (Rank IC / Pearson IC)
    - 分层回测 (分位数组合收益)
    - 因子相关性矩阵
    - 因子衰减曲线
    - 因子合成
    """

    def __init__(self, factor_values: pd.DataFrame, returns: pd.DataFrame):
        """
        Args:
            factor_values: index=date, columns=symbol, values=factor_value
            returns: index=date, columns=symbol, values=future_return
        """
        self.factor_values = factor_values
        self.returns = returns
        self._ic_cache: Dict[str, pd.Series] = {}
        self._aligned = False
        self.common_dates: List = []
        self.common_symbols: List[str] = []

    def align(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """对齐因子值和收益的日期和标的"""
        common_dates = self.factor_values.index.intersection(self.returns.index)
        common_symbols = self.factor_values.columns.intersection(self.returns.columns)

        if len(common_dates) == 0 or len(common_symbols) == 0:
            logger.warning("因子值和收益无交集")
            return pd.DataFrame(), pd.DataFrame()

        self.common_dates = common_dates
        self.common_symbols = common_symbols.tolist()
        self._aligned = True

        f_aligned = self.factor_values.loc[common_dates, common_symbols]
        r_aligned = self.returns.loc[common_dates, common_symbols]
        return f_aligned, r_aligned

    # ─── IC 分析 ───────────────────────────────────
    def compute_ic(
        self,
        factor_name: str,
        method: str = "rank",    # "rank" or "pearson"
        lag: int = 1,             # 因子→收益的滞后天数
    ) -> pd.Series:
        """计算IC序列

        Args:
            factor_name: 因子名 (factor_values的column)
            method: 'rank' = Spearman Rank IC, 'pearson' = Pearson IC
            lag: 因子值对N日后收益的预测

        Returns:
            IC时间序列 (index=date)
        """
        if factor_name not in self.factor_values.columns:
            raise ValueError(f"因子 '{factor_name}' 不在数据中")

        if not self._aligned:
            self.align()

        f_aligned, r_aligned = self.factor_values, self.returns

        ic_values = []
        ic_dates = []

        for i, dt in enumerate(f_aligned.index):
            if i + lag >= len(f_aligned.index):
                continue

            factor_t = f_aligned.iloc[i]
            return_t = r_aligned.iloc[i + lag]

            # 筛选有效数据
            mask = factor_t.notna() & return_t.notna()
            if mask.sum() < 10:  # 至少需要10个样本
                continue

            f = factor_t[mask]
            r = return_t[mask]

            try:
                if method == "rank":
                    ic = stats.spearmanr(f, r).statistic
                else:
                    ic = stats.pearsonr(f, r)[0]

                ic_values.append(ic)
                ic_dates.append(dt)
            except Exception:
                continue

        ic_series = pd.Series(ic_values, index=ic_dates, name=f"IC_{factor_name}")
        self._ic_cache[factor_name] = ic_series
        return ic_series

    def ic_summary(self, factor_name: str) -> Dict:
        """IC统计摘要"""
        ic = self._ic_cache.get(factor_name)
        if ic is None:
            ic = self.compute_ic(factor_name)

        if len(ic) == 0:
            return {"error": "IC为空"}

        ic_mean = ic.mean()
        ic_std = ic.std()
        icir = ic_mean / ic_std if ic_std > 0 else 0
        ic_positive_ratio = (ic > 0).mean()

        # 显著性检验 (t检验)
        t_stat = ic_mean / (ic_std / np.sqrt(len(ic))) if ic_std > 0 else 0
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(ic) - 1))

        return {
            "因子名": factor_name,
            "IC均值": f"{ic_mean:.4f}",
            "IC标准差": f"{ic_std:.4f}",
            "ICIR": f"{icir:.4f}",
            "IC正值比率": f"{ic_positive_ratio:.1%}",
            "IC胜率(t检验p值)": f"{p_value:.4f}",
            "观测数": len(ic),
        }

    # ─── 分层回测 ──────────────────────────────────
    def quantile_returns(
        self,
        factor_name: str,
        n_groups: int = 5,
        lag: int = 1,
    ) -> pd.DataFrame:
        """分层回测 — 按因子值分组，计算各组平均收益

        Args:
            factor_name: 因子名
            n_groups: 分几组 (默认5 = 五等分)
            lag: 滞后天数

        Returns:
            DataFrame: index=date, columns=[Q1, Q2, ..., Qn, Qn-Q1(多空)]
        """
        if factor_name not in self.factor_values.columns:
            raise ValueError(f"因子 '{factor_name}' 不在数据中")

        if not self._aligned:
            self.align()

        results = {}
        results_long_short = []

        for i, dt in enumerate(self.factor_values.index):
            if i + lag >= len(self.factor_values.index):
                continue

            factor_t = self.factor_values.iloc[i]
            return_t = self.returns.iloc[i + lag]

            mask = factor_t.notna() & return_t.notna()
            if mask.sum() < n_groups * 3:
                continue

            valid = pd.DataFrame({
                "factor": factor_t[mask],
                "return": return_t[mask],
            })

            # 按因子值分组
            valid["group"] = pd.qcut(
                valid["factor"], n_groups, labels=False, duplicates="drop"
            ) + 1  # Q1=最低, Qn=最高

            group_returns = valid.groupby("group")["return"].mean()
            results[dt] = group_returns

            # 多空收益 (最高组 - 最低组)
            if 1 in group_returns.index and n_groups in group_returns.index:
                results_long_short.append(group_returns[n_groups] - group_returns[1])

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results).T
        df.index.name = "date"
        df["多空(Qn-Q1)"] = results_long_short  # 可能长度不对齐，简化处理
        return df

    def quantile_cumulative(self, factor_name: str, n_groups: int = 5) -> pd.DataFrame:
        """分组的累计收益"""
        q_returns = self.quantile_returns(factor_name, n_groups)
        if q_returns.empty:
            return q_returns
        return (1 + q_returns).cumprod()

    # ─── 因子相关性 ────────────────────────────────
    def factor_correlation(self) -> pd.DataFrame:
        """计算因子间相关系数矩阵"""
        if not self._aligned:
            self.align()
        return self.factor_values.corr(method="spearman")

    # ─── 因子衰减 ──────────────────────────────────
    def decay_analysis(
        self,
        factor_name: str,
        max_lag: int = 20,
        method: str = "rank",
    ) -> pd.Series:
        """因子衰减分析 — IC随滞后天数的变化

        Returns:
            Series: index=lag_days, value=IC_mean
        """
        decay = {}
        for lag in range(1, max_lag + 1):
            ic = self.compute_ic(factor_name, method=method, lag=lag)
            if len(ic) > 0:
                decay[lag] = ic.mean()
            else:
                decay[lag] = np.nan

        return pd.Series(decay, name=f"{factor_name}_decay")

    # ─── 因子合成 ──────────────────────────────────
    @staticmethod
    def combine_factors(
        factor_df: pd.DataFrame,
        weights: Optional[Dict[str, float]] = None,
        method: str = "equal_weight",
    ) -> pd.Series:
        """合成多因子

        Args:
            factor_df: columns=symbol, index=factor_name 或 反转
            weights: {factor_name: weight} 因子权重
            method: 'equal_weight' | 'ic_weighted' | 'custom'

        Returns:
            合成因子值 (index=symbol)
        """
        if method == "equal_weight":
            # 先标准化(去均值/除标准差)，再等权加总
            normalized = (factor_df - factor_df.mean()) / factor_df.std()
            return normalized.mean()

        elif method == "custom" and weights:
            normalized = (factor_df - factor_df.mean()) / factor_df.std()
            total_w = sum(weights.values())
            result = pd.Series(0, index=factor_df.index)
            for name, w in weights.items():
                if name in normalized.columns:
                    result += normalized[name] * w / total_w
            return result

        else:
            # 默认等权
            return factor_df.mean()

    # ─── 统计检验 ──────────────────────────────────
    @staticmethod
    def ttest_ic(ic_series: pd.Series, alpha: float = 0.05) -> Dict:
        """IC序列的t检验"""
        ic = ic_series.dropna()
        if len(ic) < 2:
            return {"error": "样本不足"}

        t_stat, p_value = stats.ttest_1samp(ic, 0)
        return {
            "样本数": len(ic),
            "t统计量": f"{t_stat:.4f}",
            "p值": f"{p_value:.4f}",
            "是否显著": p_value < alpha,
            "置信水平": f"{(1-alpha):.0%}",
        }
