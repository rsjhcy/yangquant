"""
数据源模块
"""

from quant.data.sources.base import DataSource
from quant.data.sources.akshare_ import AkshareSource

__all__ = ["DataSource", "AkshareSource"]
