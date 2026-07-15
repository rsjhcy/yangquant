"""
卖出建议模块
根据买入价、ATR、市场状态给出止盈/止损/持有期建议

回测优化结论 (2026-07-15, 50只主板A股, 150组参数网格搜索):
- 时间止损是最大改进点: 水下3天(平衡)/2天(激进)即平仓
- 止盈目标需降低: 平衡8%, 激进12% (原来12%/18%太远到不了)
- 移动止损回撤收窄: 平衡3%, 激进7% (原来6%/9%太松)
- 盈亏比从0.99→1.71(平衡), 1.11→2.55(激进)
"""

from typing import Dict, List, Optional


class SellAdvisor:
    """卖出建议顾问

    三层保护:
    1. 时间止损: 连续水下N天 → 止损平仓 (防止死扛)
    2. 移动止损: 盈利后跟涨, 从最高点回撤N% → 止盈平仓
    3. 目标止盈: 到达目标价 → 分批或一次性止盈

    用法:
        advisor = SellAdvisor()
        plan = advisor.generate(entry_price=10.50, atr_pct=0.025, style='balanced')
        # plan = {'stop_loss': 10.00, 'take_profit_1': 10.90, ...}
    """

    # ATR 倍数配置 (有ATR数据时优先使用)
    ATR_MULTIPLIERS = {
        "balanced": {
            "stop_loss_atr": 2.0,          # 止损 = 买入价 - 2.0×ATR
            "take_profit_atr": 2.5,        # 目标止盈 = 买入价 + 2.5×ATR (从3.0降低)
            "trailing_trigger_atr": 2.0,   # 盈利超2.0×ATR后启动移动止损
            "trailing_atr": 1.0,           # 从最高点回撤1.0×ATR即卖出 (从1.8收紧)
            "max_hold_days": 8,            # 最长持有8天 (从10缩短)
            "time_stop_days": 3,           # 连续水下3天即平仓 (新增)
        },
        "aggressive": {
            "stop_loss_atr": 2.5,          # 止损 = 买入价 - 2.5×ATR
            "take_profit_atr": 3.0,        # 目标止盈 = 买入价 + 3.0×ATR (从4.0降低)
            "trailing_trigger_atr": 3.0,   # 盈利超3.0×ATR后启动移动止损
            "trailing_atr": 2.0,           # 从最高点回撤2.0×ATR即卖出 (从2.7收紧)
            "max_hold_days": 5,            # 最长持有5天 (从7缩短)
            "time_stop_days": 2,           # 连续水下2天即平仓 (新增)
        },
    }

    # 固定百分比回退方案 (无ATR数据时使用)
    PERCENT_RULES = {
        "balanced": {
            "stop_loss_pct": 0.05,         # -5% 止损 (保持)
            "take_profit_pct": 0.08,       # +8% 止盈 (从12%大幅降低)
            "trailing_trigger_pct": 0.08,  # 盈利8%后启动移动止损
            "trailing_pct": 0.03,          # 从最高点回撤3%即卖 (从6%收紧)
            "max_hold_days": 8,            # 最长持有8天 (从10缩短)
            "time_stop_days": 3,           # 连续水下3天平仓 (新增)
        },
        "aggressive": {
            "stop_loss_pct": 0.08,         # -8% 止损 (从7%放宽,给更多空间)
            "take_profit_pct": 0.12,       # +12% 止盈 (从18%大幅降低)
            "trailing_trigger_pct": 0.12,  # 盈利12%后启动移动止损
            "trailing_pct": 0.07,          # 从最高点回撤7%即卖 (从9%收紧)
            "max_hold_days": 5,            # 最长持有5天 (从7缩短)
            "time_stop_days": 2,           # 连续水下2天平仓 (新增)
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

            # 初始止损
            plan["stop_loss"] = f"{entry_price - cfg['stop_loss_atr'] * atr_value:.2f}"
            plan["stop_loss_pct"] = f"{-cfg['stop_loss_atr'] * atr_pct:.1%}"

            # 目标止盈 (单目标,不再分批)
            plan["take_profit_1"] = f"{entry_price + cfg['take_profit_atr'] * atr_value:.2f}"
            plan["take_profit_1_pct"] = f"{cfg['take_profit_atr'] * atr_pct:.1%}"
            plan["take_profit_2"] = plan["take_profit_1"]  # 保持兼容
            plan["take_profit_2_pct"] = plan["take_profit_1_pct"]

            # 移动止损
            plan["trailing_start"] = f"{entry_price + cfg['trailing_trigger_atr'] * atr_value:.2f}"
            plan["trailing_pct"] = f"{-cfg['trailing_atr'] * atr_pct:.1%}"

            # 时间止损 & 持有期
            plan["time_stop_days"] = cfg["time_stop_days"]
            plan["max_hold_days"] = cfg["max_hold_days"]

        else:
            cfg = self.PERCENT_RULES[style]

            plan["method"] = "固定百分比(无ATR数据)"

            # 初始止损
            plan["stop_loss"] = f"{entry_price * (1 - cfg['stop_loss_pct']):.2f}"
            plan["stop_loss_pct"] = f"{-cfg['stop_loss_pct']:.1%}"

            # 目标止盈 (单目标)
            plan["take_profit_1"] = f"{entry_price * (1 + cfg['take_profit_pct']):.2f}"
            plan["take_profit_1_pct"] = f"{cfg['take_profit_pct']:.1%}"
            plan["take_profit_2"] = plan["take_profit_1"]
            plan["take_profit_2_pct"] = plan["take_profit_1_pct"]

            # 移动止损
            plan["trailing_start"] = f"{entry_price * (1 + cfg['trailing_trigger_pct']):.2f}"
            plan["trailing_pct"] = f"{-cfg['trailing_pct']:.1%}"

            # 时间止损 & 持有期
            plan["time_stop_days"] = cfg["time_stop_days"]
            plan["max_hold_days"] = cfg["max_hold_days"]

        # ── 市场状态调整 ──
        if market_regime == "trending_down":
            plan["note"] = "⚠ 下跌市: 时间止损提前1天, 目标止盈降低"
            plan["time_stop_days"] = max(1, plan["time_stop_days"] - 1)
            plan["max_hold_days"] = max(3, plan["max_hold_days"] - 2)
        elif market_regime == "trending_up":
            plan["note"] = "📈 上涨市: 可以放宽持有期, 让利润奔跑"
            plan["max_hold_days"] = min(plan["max_hold_days"] + 2, 15)
            plan["time_stop_days"] = min(plan["time_stop_days"] + 1, 5)
        else:
            plan["note"] = "📊 震荡市: 严格执行止盈止损, 不恋战"

        # ── 策略说明 ──
        plan["strategy"] = (
            f"三层保护: "
            f"①时间止损: 水下{plan['time_stop_days']}天平仓 | "
            f"②初始止损: {plan['stop_loss']}({plan['stop_loss_pct']}) | "
            f"③止盈目标: {plan['take_profit_1']}({plan['take_profit_1_pct']}) | "
            f"④移动止损: 盈利>{plan['trailing_start']}后启动, 回撤{plan['trailing_pct']}即卖 | "
            f"最长持有{plan['max_hold_days']}天"
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
