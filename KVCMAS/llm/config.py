from __future__ import annotations

from dataclasses import dataclass, asdict, replace
from typing import Any, Dict, Optional
import os


@dataclass(frozen=True)
class KVCommConfig:
    """Configuration for KV communication and scheduling. """
    threshold: float = 0.3
    max_anchor_num: int = 20
    window_size: int = 5
    thread_pool_workers: int = 8
    worker_timeout: float = 30.0
    svd_rank: int = 0
    svd_rank_ph: Optional[int] = None
    svd_rank_pf: Optional[int] = None
    svd_rank_base: int = 0
    svd_rank_base_key: Optional[int] = None
    svd_rank_base_value: Optional[int] = None
    drop_shared_cache_on_finalize: bool = True
    hot_anchor_num: Optional[int] = None
    renormalize_hot_weights: bool = True

    @classmethod
    def from_env(cls) -> "KVCommConfig":
        """Create a config from environment variables with safe defaults."""
        return cls(
            threshold=float(os.environ.get("THRESHOLD", cls.threshold)),
            max_anchor_num=int(os.environ.get("MAX_ANCHOR_NUM", cls.max_anchor_num)),
            window_size=int(os.environ.get("WINDOW_SIZE", cls.window_size)),
            thread_pool_workers=int(os.environ.get("KVCMAS_THREAD_WORKERS", cls.thread_pool_workers)),
            worker_timeout=float(os.environ.get("KVCMAS_WORKER_TIMEOUT", cls.worker_timeout)),
            svd_rank=int(os.environ.get("KVCMAS_RANK", cls.svd_rank)),
            svd_rank_ph=(
                int(os.environ["KVCMAS_RANK_PH"])
                if os.environ.get("KVCMAS_RANK_PH") not in (None, "", "None")
                else cls.svd_rank_ph
            ),
            svd_rank_pf=(
                int(os.environ["KVCMAS_RANK_PF"])
                if os.environ.get("KVCMAS_RANK_PF") not in (None, "", "None")
                else cls.svd_rank_pf
            ),
            svd_rank_base=int(os.environ.get("KVCMAS_RANK_BASE", cls.svd_rank_base)),
            svd_rank_base_key=(
                int(os.environ["KVCMAS_RANK_BASE_KEY"])
                if os.environ.get("KVCMAS_RANK_BASE_KEY") not in (None, "", "None")
                else cls.svd_rank_base_key
            ),
            svd_rank_base_value=(
                int(os.environ["KVCMAS_RANK_BASE_VALUE"])
                if os.environ.get("KVCMAS_RANK_BASE_VALUE") not in (None, "", "None")
                else cls.svd_rank_base_value
            ),
            drop_shared_cache_on_finalize=os.environ.get(
                "KVCMAS_DROP_SHARED_CACHE",
                str(cls.drop_shared_cache_on_finalize),
            ).strip().lower() in ("1", "true", "yes", "on"),
            hot_anchor_num=(
                int(os.environ["KVCMAS_HOT_ANCHOR_NUM"])
                if os.environ.get("KVCMAS_HOT_ANCHOR_NUM") not in (None, "", "None")
                else cls.hot_anchor_num
            ),
            renormalize_hot_weights=os.environ.get(
                "LITEKV_RENORMALIZE_HOT_WEIGHTS",
                str(cls.renormalize_hot_weights),
            ).strip().lower() in ("1", "true", "yes", "on"),
        ).validate()

    def apply_overrides(self, **overrides: Any) -> "KVCommConfig":
        """Return a copy with provided non-None fields overridden."""
        current: Dict[str, Any] = asdict(self)
        for key, value in overrides.items():
            if value is None or key not in current:
                continue
            current[key] = value
        return replace(self, **current).validate()

    def validate(self) -> "KVCommConfig":
        """Validate value ranges and return self."""
        if self.thread_pool_workers <= 0:
            raise ValueError("thread_pool_workers must be positive")
        if self.worker_timeout <= 0:
            raise ValueError("worker_timeout must be positive")
        if self.svd_rank < 0:
            raise ValueError("svd_rank must be non-negative (0 disables compression)")
        for _name, _v in (("svd_rank_ph", self.svd_rank_ph), ("svd_rank_pf", self.svd_rank_pf)):
            if _v is not None and _v < 0:
                raise ValueError(f"{_name} must be a non-negative int or None (None = use svd_rank)")
        if self.svd_rank_base < 0:
            raise ValueError("svd_rank_base must be non-negative (0 disables base compression)")
        for _name, _v in (("svd_rank_base_key", self.svd_rank_base_key), ("svd_rank_base_value", self.svd_rank_base_value)):
            if _v is not None and _v < 0:
                raise ValueError(f"{_name} must be a non-negative int or None (None = use svd_rank_base)")
        if self.hot_anchor_num is not None and self.hot_anchor_num < 0:
            raise ValueError("hot_anchor_num must be a non-negative int or None (0/None disables)")
        return self

    @property
    def rank_ph(self) -> int:
        """Effective placeholder-delta SVD rank: svd_rank_ph, else the shared svd_rank."""
        return self.svd_rank_ph if self.svd_rank_ph is not None else self.svd_rank

    @property
    def rank_pf(self) -> int:
        """Effective prefix-delta SVD rank: svd_rank_pf, else the shared svd_rank."""
        return self.svd_rank_pf if self.svd_rank_pf is not None else self.svd_rank

    @property
    def rank_base_key(self) -> int:
        """Effective base-KEY SVD rank: svd_rank_base_key, else the shared svd_rank_base."""
        return self.svd_rank_base_key if self.svd_rank_base_key is not None else self.svd_rank_base

    @property
    def rank_base_value(self) -> int:
        """Effective base-VALUE SVD rank: svd_rank_base_value, else the shared svd_rank_base."""
        return self.svd_rank_base_value if self.svd_rank_base_value is not None else self.svd_rank_base
