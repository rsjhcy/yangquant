"""
工具装饰器: 计时、重试
"""

import functools
import time

from loguru import logger


def timer(func=None, *, label: str = ""):
    """计时装饰器 — 记录函数执行时间"""
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            name = label or f.__name__
            start = time.perf_counter()
            result = f(*args, **kwargs)
            elapsed = time.perf_counter() - start
            logger.debug(f"⏱  {name} 耗时 {elapsed:.3f}s")
            return result
        return wrapper
    return decorator(func) if func else decorator


def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """重试装饰器 — 指数退避"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts:
                        logger.error(f"❌ {func.__name__} 重试 {max_attempts} 次后仍失败: {e}")
                        raise
                    logger.warning(
                        f"⚠ {func.__name__} 第 {attempt}/{max_attempts} 次失败: {e}，"
                        f"{current_delay:.1f}s 后重试..."
                    )
                    time.sleep(current_delay)
                    current_delay *= backoff
        return wrapper
    return decorator
