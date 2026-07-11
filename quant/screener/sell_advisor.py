"""
卖出建议模块
根据买入价、ATR、市场状态给出止盈/止损/持有期建议
"""

from typing import Dict, List, Optional


class SellAdvisor:
    """卖出建议顾问

    基于 ATR(平均真实波幅) 计算动态止盈止损位:
    - 止损: 买入价 - 2.0×ATR (收紧风险)
    - 保守止盈: 买入价 + 1.5×ATR (风险收益比 1:1.5)
    - 激进止盈: 买入价 + 3.0×ATR (风险收益比 1:3)
    - 最大持有期: 5-10个交易日
    - 移动止损: 盈利超过1R后，止损上移至成本价

    用法:
        advisor = SellAdvisor()
        plan = advisor.generate(entry_price=10.50, atr_pct=0.025, style='balanced')
        # plan = {'stop_loss': 10.00, 'take_profit_conservative': 10.90, ...}
    """

    # ATR 倍数配置
    ATR_MULTIPLIERS = {
        "balanced": {
            "stop_loss_atr": 2.0,
            "take_profit_atr": 3.0,
            "trailing_trigger_atr": 2.0,
            "max_hold_days": 10,
        },
        "aggressive": {
            "stop_loss_atr": 2.5,
            "take_profit_atr": 4.0,
            "trailing_trigger_atr": 3.0,
            "max_hold_days": 7,
        },
    }

    # 固定百分比回退方案 (无ATR数据时使用)
    PERCENT_RULES = {
        "balanced": {
            "stop_loss_pct": 0.05,
            "take_profit_pct": 0.12,
            "trailing_pct": 0.06,
            "max_hold_days": 10,
        },
        "aggressive": {
            "stop_loss_pct": 0.07,
            "take_profit_pct": 0.18,
            "trailing_pct": 0.09,
            "max_hold_days": 7,
        },
    }

    def generate(
        self,
        entry_price: float,
        atr_pct: Optional[float] = None,
        style: str = "balanced",
        market_regime: str = "ranging",
    ) -> Dict:
        """生成卖出计划

        Args:
            entry_price: 建议买入价(昨日收盘)
            atr_pct: ATR占价格的百分比 (如 0.025 = 2.5%)
            style: 'balanced' | 'aggressive'
            market_regime: 'trending_up' | 'ranging' | 'trending_down'

        Returns:
            卖出计划字典
        """
        plan = {}

        if atr_pct and atr_pct > 0:
            cfg = self.ATR_MULTIPLIERS[style]
            atr_value = entry_price * atr_pct

            plan["method"] = "ATR动态"
            plan["atr_pct"] = f"{atr_pct:.1%}"
            plan["stop_loss"] = f"{entry_price - cfg['stop_loss_atr'] * atr_value:.2f}"
            plan["stop_loss_pct"] = f"{-cfg['stop_loss_atr'] * atr_pct:.1%}"
            plan["take_profit_1"] = f"{entry_price + 1.5 * atr_value:.2f}"
            plan["take_profit_1_pct"] = f"{1.5 * atr_pct:.1%}"
            plan["take_profit_2"] = f"{entry_price + cfg['take_profit_atr'] * atr_value:.2f}"
            plan["take_profit_2_pct"] = f"{cfg['take_profit_atr'] * atr_pct:.1%}"
            plan["trailing_start"] = f"{entry_price + cfg['trailing_trigger_atr'] * atr_value:.2f}"
            plan["max_hold_days"] = cfg["max_hold_days"]

        else:
            cfg = self.PERCENT_RULES[style]

            plan["method"] = "固定百分比(无ATR数据)"
            plan["stop_loss"] = f"{entry_price * (1 - cfg['stop_loss_pct']):.2f}"
            plan["stop_loss_pct"] = f"{-cfg['stop_loss_pct']:.1%}"
            plan["take_profit_1"] = f"{entry_price * (1 + cfg['take_profit_pct'] * 0.5):.2f}"
            plan["take_profit_1_pct"] = f"{cfg['take_profit_pct'] * 0.5:.1%}"
            plan["take_profit_2"] = f"{entry_price * (1 + cfg['take_profit_pct']):.2f}"
            plan["take_profit_2_pct"] = f"{cfg['take_profit_pct']:.1%}"
            plan["trailing_start"] = f"{entry_price * (1 + cfg['trailing_pct']):.2f}"
            plan["max_hold_days"] = cfg["max_hold_days"]

        # 市场状态调整
        if market_regime == "trending_down":
            plan["note"] = "⚠ 下跌市: 建议收紧止损，优先止盈1"
            plan["max_hold_days"] = min(plan["max_hold_days"], 5)
        elif market_regime == "trending_up":
            plan["note"] = "📈 上涨市: 可以放宽止盈，让利润奔跑"
            plan["max_hold_days"] = min(plan["max_hold_days"] + 3, 15)
        else:
            plan["note"] = "📊 震荡市: 严格执行止盈止损"

        # 策略说明
        plan["strategy"] = (
            f"买入后设置条件单: "
            f"止损价={plan['stop_loss']}, "
            f"分批止盈: 半仓@{plan['take_profit_1']}, 半仓@{plan['take_profit_2']}, "
            f"移动止损启动价={plan.get('trailing_start','-')}, "
            f"最长持有{plan['max_hold_days']}个交易日"
        )

        return plan

    def get_atr_pct(self, symbol: str) -> Optional[float]:
        """尝试从本地数据获取某只股票的ATR"""
        try:
            from quant.data.storage import DataStorage
            from datetime import date, timedelta
            import pandas as pd
            import numpy as np

            storage = DataStorage()
            end = date.today()
            start = end - timedelta(days=60)
            df = storage.load_daily([symbol], start, end)

            if df.empty or len(df) < 15:
                return None

            df = df.sort_values('date')
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values

            tr = np.maximum(
                high[1:] - low[1:],
                np.maximum(
                    np.abs(high[1:] - close[:-1]),
                    np.abs(low[1:] - close[:-1])
                )
            )
            atr = np.mean(tr[-14:])
            atr_pct = atr / close[-1] if close[-1] > 0 else None
            return atr_pct
        except Exception:
            return None
