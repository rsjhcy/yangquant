"""
日志系统 — 基于 loguru
"""

import sys
from pathlib import Path

from loguru import logger

from quant.config import config


def setup_logger() -> None:
    """初始化日志配置"""
    # 移除默认 handler
    logger.remove()

    # 终端输出（彩色）
    logger.add(
        sys.stderr,
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        level=config.logging.level,
        colorize=True,
    )

    # 文件输出
    log_cfg = config.logging
    log_path = Path(log_cfg.file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger.add(
        str(log_path),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        rotation=log_cfg.rotation,
        retention=log_cfg.retention,
        encoding="utf-8",
    )

    logger.info(f"📝 日志系统初始化完成 | 日志文件: {log_path}")


def get_logger(name: str = None):
    """获取命名 logger"""
    if name:
        return logger.bind(name=name)
    return logger
