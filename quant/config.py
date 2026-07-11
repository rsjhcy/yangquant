"""
全局配置管理
加载 config.yaml，提供统一的配置访问接口
"""

from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel


# 项目根目录
ROOT_DIR = Path(__file__).parent.parent
CONFIG_PATH = ROOT_DIR / "config.yaml"


class DataConfig(BaseModel):
    root_dir: str = "./data"
    cache_enabled: bool = True
    cache_ttl: int = 3600


class SourcesConfig(BaseModel):
    primary: str = "akshare"
    tushare_token: str = ""


class BacktestConfig(BaseModel):
    initial_capital: float = 1_000_000.0
    commission_rate: float = 0.00025
    stamp_duty: float = 0.001
    min_commission: float = 5.0
    slippage: float = 0.0001
    benchmark: str = "000300"


class RiskConfig(BaseModel):
    max_position_pct: float = 0.20
    max_positions: int = 20
    stop_loss_pct: float = 0.08
    stop_profit_pct: float = 0.30


class LoggingConfig(BaseModel):
    level: str = "INFO"
    file: str = "./logs/quant.log"
    rotation: str = "10 MB"
    retention: str = "30 days"


class EmailConfig(BaseModel):
    smtp_server: str = "smtp.qq.com"
    smtp_port: int = 465
    sender: str = ""
    password: str = ""
    receiver: str = ""


class ScreenerConfig(BaseModel):
    style: str = "both"
    top_n: int = 3


class AppConfig(BaseModel):
    data: DataConfig = DataConfig()
    sources: SourcesConfig = SourcesConfig()
    backtest: BacktestConfig = BacktestConfig()
    risk: RiskConfig = RiskConfig()
    email: EmailConfig = EmailConfig()
    screener: ScreenerConfig = ScreenerConfig()
    logging: LoggingConfig = LoggingConfig()


class Config:
    """全局配置单例"""

    _instance: Optional["Config"] = None
    _app_config: Optional[AppConfig] = None

    def __new__(cls) -> "Config":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load()
        return cls._instance

    def _load(self) -> None:
        """加载配置文件"""
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
        else:
            raw = {}

        self._app_config = AppConfig(
            data=DataConfig(**raw.get("data", {})),
            sources=SourcesConfig(**raw.get("sources", {})),
            backtest=BacktestConfig(**raw.get("backtest", {})),
            risk=RiskConfig(**raw.get("risk", {})),
            email=EmailConfig(**raw.get("email", {})),
            screener=ScreenerConfig(**raw.get("screener", {})),
            logging=LoggingConfig(**raw.get("logging", {})),
        )

    def reload(self) -> None:
        """重新加载配置"""
        self._load()

    @property
    def data(self) -> DataConfig:
        return self._app_config.data

    @property
    def sources(self) -> SourcesConfig:
        return self._app_config.sources

    @property
    def backtest(self) -> BacktestConfig:
        return self._app_config.backtest

    @property
    def risk(self) -> RiskConfig:
        return self._app_config.risk

    @property
    def email(self) -> "EmailConfig":
        return self._app_config.email

    @property
    def screener(self) -> "ScreenerConfig":
        return self._app_config.screener

    @property
    def logging(self) -> LoggingConfig:
        return self._app_config.logging


# 全局配置实例
config = Config()
