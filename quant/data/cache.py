"""
数据缓存 — DataFrame 内存缓存
"""

import time
from collections import OrderedDict
from typing import Any, Optional

import pandas as pd
from loguru import logger


class DataCache:
    """LRU 内存缓存，用于缓存 DataFrame 查询结果"""

    def __init__(self, max_size: int = 64, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self._cache: OrderedDict[str, tuple] = OrderedDict()

    def _make_key(self, *args, **kwargs) -> str:
        """生成缓存 key"""
        parts = [str(a) for a in args]
        parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
        return "|".join(parts)

    def get(self, *args, **kwargs) -> Optional[pd.DataFrame]:
        """获取缓存数据"""
        key = self._make_key(*args, **kwargs)
        if key in self._cache:
            data, timestamp = self._cache[key]
            if time.time() - timestamp < self.ttl:
                self._cache.move_to_end(key)
                logger.debug(f"✅ 缓存命中: {key[:80]}")
                return data
            else:
                del self._cache[key]
        return None

    def set(self, data: Any, *args, **kwargs) -> None:
        """写入缓存"""
        key = self._make_key(*args, **kwargs)
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self.max_size:
                oldest = next(iter(self._cache))
                del self._cache[oldest]
            self._cache[key] = (data, time.time())

    def clear(self) -> None:
        """清空缓存"""
        count = len(self._cache)
        self._cache.clear()
        logger.debug(f"🗑 缓存已清空: {count} 条")

    def stats(self) -> dict:
        """缓存统计"""
        return {"size": len(self._cache), "max_size": self.max_size, "ttl": self.ttl}


# 全局缓存实例
cache = DataCache()
