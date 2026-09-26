"""Preallocated KV cache — one decode-time cache policy for ALL methods. """

import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers.cache_utils import DynamicCache

__all__ = ["PreallocCache", "to_prealloc", "prealloc_enabled"]


def prealloc_enabled() -> bool:
    """Uniform preallocated decode cache (default ON). """
    return os.environ.get("KVCMAS_PREALLOC_KV", "1").strip().lower() not in ("0", "false", "no", "off")


class PreallocCache(DynamicCache):
    """DynamicCache with index-write growth instead of ``torch.cat`` growth."""

    def __init__(self, capacity: int = 0) -> None:
        super().__init__()
        self._capacity = int(capacity)
        self._buf_k: List[torch.Tensor] = []
        self._buf_v: List[torch.Tensor] = []
        self._plain = False  # set once buffers are released (see _release_buffers)

    # ---- construction ----------------------------------------------------
    @classmethod
    def from_cache(cls, cache, extra_tokens: int, slack: int = 8, consume: bool = True):
        """Copy an assembled cache into buffers sized ``seen + extra + slack``. """
        keys = list(getattr(cache, "key_cache", []) or [])
        values = list(getattr(cache, "value_cache", []) or [])
        seen = int(keys[0].shape[-2]) if keys else 0
        out = cls(capacity=seen + int(extra_tokens) + int(slack))
        for idx in range(len(keys)):
            k, v = keys[idx], values[idx]
            bk = k.new_empty((*k.shape[:-2], out._capacity, k.shape[-1]))
            bv = v.new_empty((*v.shape[:-2], out._capacity, v.shape[-1]))
            bk[..., :seen, :] = k
            bv[..., :seen, :] = v
            if consume:
                keys[idx] = None
                values[idx] = None
                try:
                    cache.key_cache[idx] = None
                    cache.value_cache[idx] = None
                except Exception:
                    pass
            out._buf_k.append(bk)
            out._buf_v.append(bv)
            out.key_cache.append(bk[..., :seen, :])
            out.value_cache.append(bv[..., :seen, :])
        out._seen_tokens = seen
        return out

    # ---- write-in-place construction -------------------------------------
    def reserve_layer(self, layer_idx: int, ref: torch.Tensor, length: int):
        """Allocate layer ``layer_idx``'s buffer and return WRITABLE ``length``-row views. """
        self._ensure(layer_idx, max(int(length), self._capacity), ref)
        k = self._buf_k[layer_idx][..., :int(length), :]
        v = self._buf_v[layer_idx][..., :int(length), :]
        self.key_cache[layer_idx] = k
        self.value_cache[layer_idx] = v
        if layer_idx == 0:
            self._seen_tokens = int(length)
        return k, v

    # ---- growth ----------------------------------------------------------
    def _ensure(self, layer_idx: int, need: int, ref: torch.Tensor) -> None:
        """Make sure layer ``layer_idx`` has a buffer with room for ``need``."""
        while len(self._buf_k) <= layer_idx:
            self._buf_k.append(None)  # type: ignore[arg-type]
            self._buf_v.append(None)  # type: ignore[arg-type]
        while len(self.key_cache) <= layer_idx:
            self.key_cache.append(None)  # type: ignore[arg-type]
            self.value_cache.append(None)  # type: ignore[arg-type]
        cap = max(self._capacity, need)
        if self._buf_k[layer_idx] is None:
            self._buf_k[layer_idx] = ref.new_empty((*ref.shape[:-2], cap, ref.shape[-1]))
            self._buf_v[layer_idx] = ref.new_empty((*ref.shape[:-2], cap, ref.shape[-1]))
            self._capacity = cap
            return
        have = self._buf_k[layer_idx].shape[-2]
        if have >= need:
            return
        # Budget was too small (e.g. more tokens generated than reserved):
        # double until it fits. Rare by construction, never silent-wrong.
        new_cap = have
        while new_cap < need:
            new_cap *= 2
        for buf in (self._buf_k, self._buf_v):
            old = buf[layer_idx]
            grown = old.new_empty((*old.shape[:-2], new_cap, old.shape[-1]))
            filled = int(self.key_cache[layer_idx].shape[-2]) if self.key_cache[layer_idx] is not None else 0
            if filled:
                grown[..., :filled, :] = old[..., :filled, :]
            buf[layer_idx] = grown
        self._capacity = max(self._capacity, new_cap)

    # ---- buffer lifetime -------------------------------------------------
    # The buffers are only useful while key_cache[i] is a VIEW of them.
    def _release_buffers(self) -> None:
        self._buf_k = []
        self._buf_v = []
        self._plain = True

    def slice_(self, start=None, end=None):
        out = DynamicCache.slice_(self, start, end)
        self._release_buffers()
        return out

    def slice(self, start=None, end=None):
        return DynamicCache.slice(self, start, end)

    def copy(self):
        return DynamicCache.copy(self)

    def _buffers_live(self, layer_idx: int) -> bool:
        if getattr(self, "_plain", False):
            return False
        if layer_idx >= len(self._buf_k) or self._buf_k[layer_idx] is None:
            return True  # not allocated yet; _ensure will create it
        cur = self.key_cache[layer_idx] if layer_idx < len(self.key_cache) else None
        if cur is None:
            return True
        # still a view of our buffer?
        return cur.untyped_storage().data_ptr() == self._buf_k[layer_idx].untyped_storage().data_ptr()

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self._buffers_live(layer_idx):
            # Views were replaced (or buffers released): behave like a stock cache.
            self._release_buffers()
            return DynamicCache.update(self, key_states, value_states, layer_idx, cache_kwargs)
        n_new = int(key_states.shape[-2])
        start = int(self.key_cache[layer_idx].shape[-2]) if (
            layer_idx < len(self.key_cache) and self.key_cache[layer_idx] is not None
        ) else 0
        end = start + n_new
        self._ensure(layer_idx, end, key_states)
        self._buf_k[layer_idx][..., start:end, :] = key_states
        self._buf_v[layer_idx][..., start:end, :] = value_states
        # Re-view to the true length so external shape reads stay correct.
        self.key_cache[layer_idx] = self._buf_k[layer_idx][..., :end, :]
        self.value_cache[layer_idx] = self._buf_v[layer_idx][..., :end, :]
        if layer_idx == 0:
            self._seen_tokens = end
        return self.key_cache[layer_idx], self.value_cache[layer_idx]


def to_prealloc(cache, extra_tokens: int, slack: int = 8, consume: bool = True):
    """Convert ``cache`` to a PreallocCache (no-op if it already is one). """
    if isinstance(cache, PreallocCache):
        return cache
    return PreallocCache.from_cache(cache, extra_tokens=extra_tokens, slack=slack, consume=consume)
