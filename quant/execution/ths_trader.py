"""
同花顺模拟交易执行器
通过 Playwright 自动操作同花顺网页版模拟炒股

首次使用需要手动登录一次（扫码或手机验证），
登录后 session 会保存到本地，后续自动复用。
"""

import json
import time
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger


class THSTrader:
    """同花顺模拟交易执行器

    交易流程:
    1. 打开 https://moni.10jqka.com.cn/
    2. 检查登录状态，未登录则提示扫码
    3. 导航到买入页面
    4. 依次下单
    5. 截图保存交易记录

    用法:
        trader = THSTrader()
        trader.login()                     # 首次手动登录
        trader.execute_buy([
            {'symbol': '002415', 'price': 35.39, 'quantity': 1000},
            {'symbol': '600276', 'price': 55.61, 'quantity': 500},
        ])
    """

    HOME_URL = "https://moni.10jqka.com.cn/"
    LOGIN_URL = "https://upass.10jqka.com.cn/login"
    BUY_URL = "https://moni.10jqka.com.cn/trade/buy"
    AUTH_FILE = "data/ths_auth.json"

    def __init__(self, headless: bool = False):
        self.headless = headless
        self._browser = None
        self._page = None
        self._playwright = None

    def _ensure_browser(self):
        """启动浏览器"""
        if self._browser is not None:
            return

        from playwright.sync_api import sync_playwright

        self._playwright = sync_playwright().start()

        # 使用持久化上下文保存登录状态
        auth_dir = Path(self.AUTH_FILE).parent / "ths_browser_data"
        auth_dir.mkdir(parents=True, exist_ok=True)

        self._context = self._playwright.chromium.launch_persistent_context(
            str(auth_dir),
            headless=self.headless,
            viewport={"width": 1280, "height": 800},
            locale="zh-CN",
        )
        self._page = self._context.new_page()
        logger.info("浏览器已启动")

    def login(self) -> bool:
        """登录同花顺

        如果不是 headless 模式，会打开浏览器窗口让你手动扫码登录。
        headless 模式下需要已保存的登录状态。
        """
        self._ensure_browser()

        # 先尝试用已保存的 cookie 访问
        self._page.goto(self.HOME_URL, wait_until="domcontentloaded", timeout=30000)
        time.sleep(3)

        # 检查是否已登录
        if self._is_logged_in():
            logger.info("已登录(复用session)")
            return True

        # 需要登录
        logger.info("需要登录，请在浏览器中完成登录...")

        if self.headless:
            logger.error("headless模式下无法交互登录，请先运行 trader.login() 手动登录一次")
            return False

        # 非 headless: 点击登录按钮
        try:
            login_btn = self._page.locator("text=登录").first
            if login_btn.is_visible():
                login_btn.click()
                time.sleep(2)
        except Exception:
            pass

        # 等待用户完成登录 (扫码/验证码)
        logger.info("请在弹出的浏览器中完成登录（扫码或手机验证）...")
        logger.info("登录成功后按 Enter 继续...")

        input(">>> 登录完成后按 Enter 键继续: ")

        self._page.goto(self.HOME_URL, wait_until="domcontentloaded")
        time.sleep(2)

        if self._is_logged_in():
            logger.info("登录成功！session 已保存")
            return True

        logger.error("登录失败，请重试")
        return False

    def _is_logged_in(self) -> bool:
        """检查是否已登录"""
        try:
            # 检查页面是否显示了用户名或资产信息
            user_element = self._page.locator(".username, .user-name, [class*=user]").first
            asset_element = self._page.locator("text=总资产").first
            return user_element.is_visible() or asset_element.is_visible()
        except Exception:
            return False

    def execute_buy(
        self,
        orders: List[Dict],
        dry_run: bool = False,
    ) -> List[Dict]:
        """执行买入订单

        Args:
            orders: [
                {'symbol': '002415', 'price': 35.39, 'quantity': 1000},
                ...
            ]
            dry_run: True=只模拟不下单

        Returns:
            执行结果列表 [{symbol, status, message}]
        """
        self._ensure_browser()

        if not self._is_logged_in():
            logger.error("未登录，请先调用 trader.login()")
            return [{"symbol": o["symbol"], "status": "FAILED", "message": "未登录"} for o in orders]

        results = []
        for i, order in enumerate(orders):
            symbol = order["symbol"]
            price = order.get("price", 0)
            quantity = order.get("quantity", 1000)

            logger.info(f"[{i+1}/{len(orders)}] 买入 {symbol} {quantity}股 @ {price}")

            if dry_run:
                logger.info(f"  [模拟] 跳过实际下单")
                results.append({
                    "symbol": symbol, "status": "DRY_RUN",
                    "price": price, "quantity": quantity,
                    "message": "模拟模式，未实际下单",
                })
                continue

            try:
                result = self._place_buy_order(symbol, price, quantity)
                results.append(result)
            except Exception as e:
                logger.error(f"  {symbol} 下单失败: {e}")
                results.append({"symbol": symbol, "status": "FAILED", "message": str(e)})

        # 截图保存交易记录
        try:
            screenshot_path = f"data/trade_{date.today().isoformat()}.png"
            self._page.screenshot(path=screenshot_path, full_page=True)
            logger.info(f"交易截图: {screenshot_path}")
        except Exception:
            pass

        return results

    def _place_buy_order(self, symbol: str, price: float, quantity: int) -> Dict:
        """在页面上执行一笔买入"""
        self._page.goto(self.BUY_URL, wait_until="domcontentloaded")
        time.sleep(2)

        # 输入股票代码
        code_input = self._page.locator("input[placeholder*=代码], input[name*=code], #stockCode")
        if code_input.is_visible():
            code_input.click()
            code_input.fill(symbol)
            time.sleep(1)

        # 可能弹出搜索下拉，按 Enter 确认
        self._page.keyboard.press("Enter")
        time.sleep(1)

        # 输入价格
        price_input = self._page.locator("input[placeholder*=价格], input[name*=price], #price")
        if price_input.is_visible():
            price_input.click()
            price_input.fill(str(price))
            time.sleep(0.5)

        # 输入数量
        qty_input = self._page.locator("input[placeholder*=数量], input[name*=amount], #amount, #quantity")
        if qty_input.is_visible():
            qty_input.click()
            qty_input.fill(str(quantity))
            time.sleep(0.5)

        # 点击买入按钮
        buy_btn = self._page.locator("button:has-text('买入'), button:has-text('下单'), button:has-text('确定')").first
        if buy_btn.is_visible():
            buy_btn.click()
            time.sleep(2)

        # 确认弹窗
        try:
            confirm_btn = self._page.locator("button:has-text('确认'), button:has-text('是'), button:has-text('确定')").first
            if confirm_btn.is_visible(timeout=3000):
                confirm_btn.click()
                time.sleep(2)
        except Exception:
            pass

        return {
            "symbol": symbol,
            "status": "SUBMITTED",
            "price": price,
            "quantity": quantity,
            "message": "订单已提交",
        }

    def get_account_info(self) -> Dict:
        """获取账户信息"""
        self._ensure_browser()

        if not self._is_logged_in():
            return {"error": "未登录"}

        self._page.goto(self.HOME_URL, wait_until="domcontentloaded")
        time.sleep(2)

        info = {
            "total_asset": "N/A",
            "available_cash": "N/A",
            "market_value": "N/A",
            "total_pnl": "N/A",
        }

        try:
            # 尝试提取页面上的数字
            page_text = self._page.content()
            info["page_loaded"] = True
        except Exception as e:
            info["error"] = str(e)

        return info

    def close(self):
        """关闭浏览器"""
        if self._browser:
            self._browser.close()
        if self._playwright:
            self._playwright.stop()
        logger.info("浏览器已关闭")


# ─── CLI 入口 ────────────────────────
def run_trade_plan(trade_plan_path: str = "data/trade_plan.json"):
    """从交易计划文件读取并执行交易"""
    import sys

    plan_path = Path(trade_plan_path)
    if not plan_path.exists():
        logger.error(f"交易计划不存在: {plan_path}")
        return

    with open(plan_path, "r", encoding="utf-8") as f:
        plan = json.load(f)

    logger.info(f"加载交易计划: {plan.get('date', 'unknown')}")
    logger.info(f"  平衡型: {len(plan.get('balanced', []))} 只")
    logger.info(f"  激进型: {len(plan.get('aggressive', []))} 只")

    trader = THSTrader(headless=False)

    # 登录
    if not trader.login():
        logger.error("登录失败，退出")
        return

    # 检查是否为模拟交易
    dry_run = "--dry-run" in sys.argv

    # 执行买入（默认使用激进型推荐）
    style = plan.get("style", "aggressive")
    orders = []
    for pick in plan.get(style, []):
        close = float(pick["close"])
        orders.append({
            "symbol": pick["symbol"],
            "price": close,
            "quantity": _calc_quantity(close, plan.get("capital", 100000), len(plan.get(style, []))),
        })

    results = trader.execute_buy(orders, dry_run=dry_run)
    trader.close()

    # 输出结果
    success = sum(1 for r in results if r["status"] in ("SUBMITTED", "DRY_RUN"))
    logger.info(f"交易完成: {success}/{len(results)} 成功")
    return results


def _calc_quantity(price: float, total_capital: int, n_stocks: int) -> int:
    """计算买入数量（整手，每只等权）"""
    per_stock = total_capital / n_stocks
    qty = int(per_stock / price / 100) * 100  # 整手
    return max(100, qty)


if __name__ == "__main__":
    # 首次使用:
    #   python -m quant.execution.ths_trader
    # 会自动打开浏览器让你扫码登录同花顺

    trader = THSTrader(headless=False)
    if trader.login():
        info = trader.get_account_info()
        print(json.dumps(info, ensure_ascii=False, indent=2))
    trader.close()
