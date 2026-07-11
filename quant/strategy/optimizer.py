"""
策略参数优化器
支持网格搜索和贝叶斯优化(Optuna)
"""

from itertools import product
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


class StrategyOptimizer:
    """策略参数优化器

    两种模式:
    1. grid_search — 穷举搜索 (适合2-3个参数)
    2. bayesian — 贝叶斯优化 (适合多参数, 需要optuna)
    """

    def __init__(
        self,
        strategy_class,
        param_grid: Dict[str, List[Any]],
        objective: Optional[Callable] = None,
    ):
        """
        Args:
            strategy_class: 策略类 (非实例)
            param_grid: {param_name: [values...]}
            objective: 目标函数 (result) -> float (越大越好)
        """
        self.strategy_class = strategy_class
        self.param_grid = param_grid
        self.objective = objective or self._default_objective
        self.results: List[Dict] = []

    @staticmethod
    def _default_objective(result) -> float:
        """默认优化目标: 夏普比率"""
        if hasattr(result, 'describe'):
            d = result.describe()
            if 'sharpe_ratio' in d:
                return float(d['sharpe_ratio'])
        return result.total_return if hasattr(result, 'total_return') else 0

    # ─── 网格搜索 ──────────────────────────────────
    def grid_search(
        self,
        data_provider: Callable,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """网格搜索最优参数

        Args:
            data_provider: 返回 (data_df) 的可调用对象
            verbose: 是否显示进度

        Returns:
            DataFrame 所有参数组合的得分
        """
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())

        total = 1
        for v in param_values:
            total *= len(v)

        logger.info(f"🔍 网格搜索: {total} 个组合, 参数: {param_names}")

        self.results = []
        best_score = -float("inf")
        best_params = None

        for i, combo in enumerate(product(*param_values)):
            params = dict(zip(param_names, combo))

            try:
                # 用参数创建策略实例
                strategy = self.strategy_class(**params)

                # 运行回测
                from quant.backtest.engine import BacktestEngine

                engine = BacktestEngine()
                df = data_provider()
                if isinstance(df, tuple):
                    df = df[0]
                engine.set_data(df)
                engine.set_strategy(strategy)
                result = engine.run(progress=False)

                score = self.objective(result)

                record = {
                    **params,
                    "score": score,
                    "total_return": result.total_return,
                    "max_drawdown": result.describe().get("max_drawdown", 0),
                }
                self.results.append(record)

                if score > best_score:
                    best_score = score
                    best_params = params.copy()

                if verbose and (i + 1) % max(1, total // 10) == 0:
                    logger.info(
                        f"  [{i+1}/{total}] 最佳: {best_params} → "
                        f"score={best_score:.4f}"
                    )

            except Exception as e:
                logger.warning(f"参数组合 {params} 失败: {e}")
                continue

        logger.info(f"✅ 网格搜索完成! 最佳参数: {best_params} → score={best_score:.4f}")

        return pd.DataFrame(self.results).sort_values("score", ascending=False)

    # ─── 贝叶斯优化 ────────────────────────────────
    def bayesian_search(
        self,
        data_provider: Callable,
        n_trials: int = 100,
        direction: str = "maximize",
    ) -> Optional[Dict]:
        """贝叶斯优化 (需要 optuna)

        Args:
            data_provider: 返回 (data_df) 的可调用对象
            n_trials: 搜索次数
            direction: 'maximize' | 'minimize'

        Returns:
            最佳参数字典
        """
        try:
            import optuna
        except ImportError:
            logger.error("贝叶斯优化需要 optuna: pip install optuna")
            return None

        param_grid = self.param_grid

        def objective_fn(trial: optuna.Trial) -> float:
            params = {}
            for name, values in param_grid.items():
                if all(isinstance(v, int) for v in values):
                    params[name] = trial.suggest_int(name, min(values), max(values))
                elif all(isinstance(v, float) for v in values):
                    params[name] = trial.suggest_float(name, min(values), max(values))
                else:
                    params[name] = trial.suggest_categorical(name, values)

            try:
                strategy = self.strategy_class(**params)
                from quant.backtest.engine import BacktestEngine

                engine = BacktestEngine()
                df = data_provider()
                if isinstance(df, tuple):
                    df = df[0]
                engine.set_data(df)
                engine.set_strategy(strategy)
                result = engine.run(progress=False)
                return self.objective(result)
            except Exception as e:
                return float("-inf")

        study = optuna.create_study(direction=direction)
        study.optimize(objective_fn, n_trials=n_trials, show_progress_bar=True)

        logger.info(f"✅ 贝叶斯优化完成!")
        logger.info(f"   最佳参数: {study.best_params}")
        logger.info(f"   最佳得分: {study.best_value:.4f}")

        return {
            "best_params": study.best_params,
            "best_score": study.best_value,
            "study": study,
        }

    def plot_optimization(self, save_path: Optional[str] = None):
        """可视化优化结果"""
        if not self.results:
            logger.warning("无优化结果")
            return

        try:
            import plotly.express as px
            import plotly.graph_objects as go

            df = pd.DataFrame(self.results)
            param_names = [k for k in df.columns if k not in ("score", "total_return", "max_drawdown")]

            if len(param_names) >= 2:
                # 热力图
                pivot = df.pivot_table(
                    index=param_names[0],
                    columns=param_names[1],
                    values="score",
                    aggfunc="max",
                )
                fig = go.Figure(data=go.Heatmap(
                    z=pivot.values,
                    x=pivot.columns,
                    y=pivot.index,
                    colorscale="RdYlGn",
                ))
                fig.update_layout(
                    title="参数优化 - 得分热力图",
                    xaxis_title=param_names[1],
                    yaxis_title=param_names[0],
                )
                fig.show()

        except ImportError:
            logger.warning("需要 plotly 来可视化优化结果")
