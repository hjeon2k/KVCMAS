"""Utilities to manipulate DynamicCache and coordinate KV anchor workflows. """
from __future__ import annotations

import copy
import json
import os
import threading
from collections.abc import MutableMapping
from collections.abc import Sequence
from time import perf_counter
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
from transformers.cache_utils import DynamicCache

from KVCMAS.llm.token_ops import concat
from KVCMAS.utils.log import logger

_MISSING = object()
_DELETED = object()

# The cache ops below run per layer. Set to 0 for the stacked fast-path, which materializes a
# full [L, B, H, S, D] tensor and OOMs on long sequences.
_LAYERWISE_CACHE_OPS = os.environ.get("KVCMAS_LAYERWISE_CACHE_OPS", "1") == "1"

def _is_layered_cache(cache: DynamicCache) -> bool:
    """Return True if the cache uses the newer `layers` structure."""
    return hasattr(cache, "layers")


def _get_layer_count(cache: DynamicCache) -> int:
    """Return number of transformer layers tracked in the cache."""
    if _is_layered_cache(cache):
        return len(cache.layers)
    return len(getattr(cache, "key_cache", []))


def _get_layer_kv(cache: DynamicCache, idx: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Return key/value tensors (or None) for a given layer index."""
    if _is_layered_cache(cache):
        layer = cache.layers[idx]
        return getattr(layer, "keys", None), getattr(layer, "values", None)
    key_cache = getattr(cache, "key_cache", [])
    value_cache = getattr(cache, "value_cache", [])
    key = key_cache[idx] if idx < len(key_cache) else None
    value = value_cache[idx] if idx < len(value_cache) else None
    return key, value


def _set_layer_kv(
    cache: DynamicCache,
    idx: int,
    key: Optional[torch.Tensor],
    value: Optional[torch.Tensor],
) -> None:
    """Assign key/value tensors to a specific layer, updating metadata if present."""
    if _is_layered_cache(cache):
        layer = cache.layers[idx]
        layer.keys = key
        layer.values = value
        if hasattr(layer, "is_initialized"):
            layer.is_initialized = bool(isinstance(key, torch.Tensor) and key.numel() > 0)
        if hasattr(layer, "dtype") and isinstance(key, torch.Tensor):
            layer.dtype = key.dtype
        if hasattr(layer, "device") and isinstance(key, torch.Tensor):
            layer.device = key.device
        if hasattr(layer, "cumulative_length") and isinstance(key, torch.Tensor):
            layer.cumulative_length = key.shape[-2]
    else:
        key_cache = getattr(cache, "key_cache")
        value_cache = getattr(cache, "value_cache")
        key_cache[idx] = key
        value_cache[idx] = value


def _stack_cache_tensors(cache: DynamicCache) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Return stacked key/value tensors when all layers are dense tensors."""
    layer_count = _get_layer_count(cache)
    if layer_count == 0:
        return None
    keys: List[torch.Tensor] = []
    values: List[torch.Tensor] = []
    for idx in range(layer_count):
        key, value = _get_layer_kv(cache, idx)
        if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
            return None
        keys.append(key)
        values.append(value)
    if not keys:
        return None
    try:
        key_stack = torch.stack(keys)
        value_stack = torch.stack(values)
    except RuntimeError:
        return None
    return key_stack, value_stack


def _assign_stack_to_cache(cache: DynamicCache, key_stack: torch.Tensor, value_stack: torch.Tensor) -> None:
    """Overwrite cache layers with stacked tensors maintaining per-layer metadata."""
    layer_count = _get_layer_count(cache)
    if _is_layered_cache(cache):
        if layer_count != key_stack.shape[0]:
            raise ValueError("Layer count mismatch while assigning stacked cache tensors.")
        for idx in range(layer_count):
            layer = cache.layers[idx]
            layer.keys = key_stack[idx]
            layer.values = value_stack[idx]
            if hasattr(layer, "is_initialized"):
                layer.is_initialized = key_stack[idx].shape[-2] > 0
            if hasattr(layer, "dtype"):
                layer.dtype = key_stack[idx].dtype
            if hasattr(layer, "device"):
                layer.device = key_stack[idx].device
            if hasattr(layer, "cumulative_length"):
                layer.cumulative_length = key_stack[idx].shape[-2]
    else:
        cache.key_cache = list(key_stack)
        cache.value_cache = list(value_stack)


def _layer_is_empty(tensor: Optional[torch.Tensor]) -> bool:
    if tensor is None:
        return True
    if isinstance(tensor, list):
        return len(tensor) == 0
    if isinstance(tensor, torch.Tensor):
        return tensor.numel() == 0 or tensor.shape[-2] == 0
    return False


def _layer_length(tensor: Optional[torch.Tensor]) -> int:
    if isinstance(tensor, torch.Tensor) and tensor.ndim >= 2:
        return int(tensor.shape[-2])
    return 0


def _normalize_indices(cache: DynamicCache, start: Optional[int], end: Optional[int]) -> Tuple[int, int]:
    """Convert None/negative indices into bounded absolute [start, end)."""
    seq_len = _safe_seq_len(cache)
    if start is None:
        start = 0
    elif start < 0:
        start = seq_len + start
    if end is None:
        end = seq_len
    elif end < 0:
        end = seq_len + end
    start = max(0, min(seq_len, start))
    end = max(0, min(seq_len, end))
    return start, end


def _safe_seq_len(cache: DynamicCache) -> int:
    """Best-effort sequence length from cache (APIs vary across transformers)."""
    getter = getattr(cache, "get_seq_length", None)
    if callable(getter):
        try:
            length = getter()
        except TypeError:
            length = getter(0)
        if length is not None:
            return int(length)

    for idx in range(_get_layer_count(cache)):
        key, _ = _get_layer_kv(cache, idx)
        if not _layer_is_empty(key):
            return _layer_length(key)
    return 0


def _clone_tensor_or_empty(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if tensor is None or isinstance(tensor, list):
        return tensor
    return tensor.clone()


def _empty_like_tensor(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if tensor is None or isinstance(tensor, list):
        return tensor
    return tensor[..., :0, :].clone()


def _ensure_same_layout(cache: DynamicCache, other: DynamicCache) -> None:
    if _get_layer_count(cache) != _get_layer_count(other):
        raise ValueError("Layer count mismatch between DynamicCache objects.")


def _set_seen_tokens(cache: DynamicCache, length: int) -> None:
    try:
        setattr(cache, "_seen_tokens", int(length))
    except Exception:
        pass


def _copy_cache(cache: DynamicCache) -> DynamicCache:
    new_cache = type(cache)()
    if _is_layered_cache(cache):
        new_cache.layers = []
        for idx in range(len(cache.layers)):
            original_layer = cache.layers[idx]
            cloned_layer = copy.deepcopy(original_layer)
            if hasattr(cloned_layer, "keys") and isinstance(cloned_layer.keys, torch.Tensor):
                cloned_layer.keys = cloned_layer.keys.clone()
            if hasattr(cloned_layer, "values") and isinstance(cloned_layer.values, torch.Tensor):
                cloned_layer.values = cloned_layer.values.clone()
            new_cache.layers.append(cloned_layer)
    else:
        new_cache.key_cache = []
        new_cache.value_cache = []
        for idx in range(len(cache.key_cache)):
            key, value = cache.key_cache[idx], cache.value_cache[idx]
            new_cache.key_cache.append(_clone_tensor_or_empty(key))
            new_cache.value_cache.append(_clone_tensor_or_empty(value))
    for attr in ("offloading", "only_non_sliding", "prefetch_stream", "layer_class_to_replicate"):
        if hasattr(cache, attr):
            setattr(new_cache, attr, getattr(cache, attr))
    if hasattr(cache, "_seen_tokens"):
        _set_seen_tokens(new_cache, getattr(cache, "_seen_tokens"))
    return new_cache


def _slice_inplace(cache: DynamicCache, start: Optional[int], end: Optional[int]) -> DynamicCache:
    """In-place slice of KV cache along sequence dimension."""
    start, end = _normalize_indices(cache, start, end)
    if start >= end:
        for idx in range(_get_layer_count(cache)):
            key, value = _get_layer_kv(cache, idx)
            _set_layer_kv(cache, idx, _empty_like_tensor(key), _empty_like_tensor(value))
        _set_seen_tokens(cache, 0)
        return cache

    stacked = None if _LAYERWISE_CACHE_OPS else _stack_cache_tensors(cache)
    if stacked is not None:
        key_stack, value_stack = stacked
        slice_start = min(start, key_stack.shape[-2])
        slice_end = min(end, key_stack.shape[-2])
        if slice_end <= slice_start:
            new_key_stack = key_stack[..., :0, :].clone()
            new_value_stack = value_stack[..., :0, :].clone()
        else:
            new_key_stack = key_stack[..., slice_start:slice_end, :].clone()
            new_value_stack = value_stack[..., slice_start:slice_end, :].clone()
        _assign_stack_to_cache(cache, new_key_stack, new_value_stack)
    else:
        for idx in range(_get_layer_count(cache)):
            key, value = _get_layer_kv(cache, idx)
            if _layer_is_empty(key):
                continue
            current_len = _layer_length(key)
            slice_start = min(start, current_len)
            slice_end = min(end, current_len)
            if slice_end <= slice_start:
                new_key = _empty_like_tensor(key)
                new_value = _empty_like_tensor(value)
            else:
                new_key = key[..., slice_start:slice_end, :].clone()
                new_value = value[..., slice_start:slice_end, :].clone()
            _set_layer_kv(cache, idx, new_key, new_value)

    _set_seen_tokens(cache, end - start)
    return cache


def _slice_functional(cache: DynamicCache, start: Optional[int], end: Optional[int]) -> DynamicCache:
    """Return a NEW cache containing only ``[start:end]``, source left untouched. """
    if _is_layered_cache(cache):
        return _slice_inplace(_copy_cache(cache), start, end)
    start_n, end_n = _normalize_indices(cache, start, end)
    new_cache = type(cache)()
    new_cache.key_cache = []
    new_cache.value_cache = []
    for key, value in zip(cache.key_cache, cache.value_cache):
        if not isinstance(key, torch.Tensor) or _layer_is_empty(key):
            new_cache.key_cache.append(_empty_like_tensor(key))
            new_cache.value_cache.append(_empty_like_tensor(value))
            continue
        cur = key.shape[-2]
        s = min(start_n, cur)
        e = min(end_n, cur)
        if e <= s:
            new_cache.key_cache.append(key[..., :0, :].clone())
            new_cache.value_cache.append(value[..., :0, :].clone())
        else:
            new_cache.key_cache.append(key[..., s:e, :].clone())
            new_cache.value_cache.append(value[..., s:e, :].clone())
    for attr in ("offloading", "only_non_sliding", "prefetch_stream", "layer_class_to_replicate"):
        if hasattr(cache, attr):
            setattr(new_cache, attr, getattr(cache, attr))
    _set_seen_tokens(new_cache, max(0, end_n - start_n))
    return new_cache


def _slice_view(cache: DynamicCache, start: Optional[int], end: Optional[int]) -> DynamicCache:
    """Return a NEW cache whose layers are VIEWS of ``cache[start:end]`` — no clone. """
    if _is_layered_cache(cache):
        return _slice_functional(cache, start, end)
    start_n, end_n = _normalize_indices(cache, start, end)
    new_cache = type(cache)()
    new_cache.key_cache = []
    new_cache.value_cache = []
    for key, value in zip(cache.key_cache, cache.value_cache):
        if not isinstance(key, torch.Tensor) or _layer_is_empty(key):
            new_cache.key_cache.append(key)
            new_cache.value_cache.append(value)
        else:
            new_cache.key_cache.append(key[..., start_n:end_n, :])      # view, no clone
            new_cache.value_cache.append(value[..., start_n:end_n, :])
    for attr in ("offloading", "only_non_sliding", "prefetch_stream", "layer_class_to_replicate"):
        if hasattr(cache, attr):
            setattr(new_cache, attr, getattr(cache, attr))
    _set_seen_tokens(new_cache, max(0, end_n - start_n))
    return new_cache


def _concat_tensors(
    base: Optional[torch.Tensor],
    additions: Sequence[Optional[torch.Tensor]],
) -> Optional[torch.Tensor]:
    tensors: List[torch.Tensor] = []
    if isinstance(base, torch.Tensor) and base.shape[-2] > 0:
        tensors.append(base)
    for tensor in additions:
        if isinstance(tensor, torch.Tensor) and tensor.shape[-2] > 0:
            tensors.append(tensor)
    if not tensors:

        for candidate in [base, *additions]:
            if isinstance(candidate, torch.Tensor):
                return candidate[..., :0, :].clone()
        return base
    first_tensor = tensors[0]
    other_tensors = [t if t is first_tensor else t.to(first_tensor.device) for t in tensors[1:]]
    return torch.cat([first_tensor] + other_tensors, dim=-2)


def _ensure_cache_sequence(
    caches: Union[DynamicCache, Sequence[DynamicCache], None]
) -> List[DynamicCache]:
    if caches is None:
        return []
    if isinstance(caches, (list, tuple)):
        return [cache for cache in caches if cache is not None]
    return [caches]


def _concat_inplace(cache: DynamicCache, others: Sequence[DynamicCache]) -> DynamicCache:
    """In-place concatenate multiple caches along sequence dimension."""
    if not others:
        return cache

    usable = [other for other in others if other is not None]
    if not usable:
        return cache

    for other in usable:
        _ensure_same_layout(cache, other)

    if not _LAYERWISE_CACHE_OPS:
        base_stack = _stack_cache_tensors(cache)
        other_stacks = [_stack_cache_tensors(other) for other in usable]
        if base_stack is not None and all(stack is not None for stack in other_stacks):
            base_keys, base_values = base_stack
            other_keys = [stack[0] for stack in other_stacks]
            other_values = [stack[1] for stack in other_stacks]
            key_stack = torch.cat([base_keys] + other_keys, dim=-2)
            value_stack = torch.cat([base_values] + other_values, dim=-2)
            _assign_stack_to_cache(cache, key_stack, value_stack)
            _set_seen_tokens(cache, key_stack.shape[-2])
            return cache

    for idx in range(_get_layer_count(cache)):
        base_key, base_value = _get_layer_kv(cache, idx)
        other_keys = []
        other_values = []
        for other in usable:
            key, value = _get_layer_kv(other, idx)
            other_keys.append(key)
            other_values.append(value)

        new_key = _concat_tensors(base_key, other_keys)
        new_value = _concat_tensors(base_value, other_values)
        _set_layer_kv(cache, idx, new_key, new_value)

    new_length = _safe_seq_len(cache)
    _set_seen_tokens(cache, new_length)
    return cache


def _concat_functional(cache: DynamicCache, others: Sequence[DynamicCache]) -> DynamicCache:
    """Return a new cache that is the concatenation of base and others."""
    copied = _copy_cache(cache)
    return _concat_inplace(copied, others)


def _replace_inplace(cache: DynamicCache, start: int, end: int, real: DynamicCache) -> DynamicCache:
    """In-place replace [start, end) with the content from `real`."""
    left = cache.slice(start=0, end=start)
    middle = real.copy()
    right = cache.slice(start=end, end=None)
    replaced = left.concat([middle, right])
    if _is_layered_cache(cache):
        cache.layers = replaced.layers
    else:
        cache.key_cache = replaced.key_cache
        cache.value_cache = replaced.value_cache
    _set_seen_tokens(cache, _safe_seq_len(cache))
    return cache


def _replace_functional(cache: DynamicCache, start: int, end: int, real: DynamicCache) -> DynamicCache:
    """Return a copy of cache with [start, end) replaced by `real`."""
    copied = _copy_cache(cache)
    return _replace_inplace(copied, start, end, real)


def _select_indices(cache: DynamicCache, indices: torch.Tensor) -> DynamicCache:
    """Select positions by index tensor, preserving layout and metadata."""
    stacked = None if _LAYERWISE_CACHE_OPS else _stack_cache_tensors(cache)
    if stacked is not None:
        key_stack, value_stack = stacked
        selected_keys = key_stack[..., indices, :].clone()
        selected_values = value_stack[..., indices, :].clone()
        _assign_stack_to_cache(cache, selected_keys, selected_values)
        _set_seen_tokens(cache, indices.shape[-1])
        return cache

    for idx in range(_get_layer_count(cache)):
        key, value = _get_layer_kv(cache, idx)
        if _layer_is_empty(key):
            continue
        _set_layer_kv(cache, idx, key[..., indices, :].clone(), value[..., indices, :].clone())
    _set_seen_tokens(cache, indices.shape[-1])
    return cache


def _to_device(cache: DynamicCache, device: Union[str, torch.device]) -> DynamicCache:
    """Move all tensors in the cache to the specified device."""
    if _is_layered_cache(cache):
        for layer in cache.layers:
            if isinstance(getattr(layer, "keys", None), torch.Tensor):
                layer.keys = layer.keys.to(device)
            if isinstance(getattr(layer, "values", None), torch.Tensor):
                layer.values = layer.values.to(device)
    else:
        for idx in range(len(cache.key_cache)):
            if isinstance(cache.key_cache[idx], torch.Tensor):
                cache.key_cache[idx] = cache.key_cache[idx].to(device)
            if isinstance(cache.value_cache[idx], torch.Tensor):
                cache.value_cache[idx] = cache.value_cache[idx].to(device)
    return cache


def _split_cache_by_placeholders(
    cache: DynamicCache,
    placeholder_dict: Dict[str, Tuple[int, int]],
) -> Tuple[List[DynamicCache], List[DynamicCache]]:
    """Split a cache into placeholder and prefix segments per provided spans."""
    if not placeholder_dict:
        return [], [cache.copy()]

    total_len = _safe_seq_len(cache)
    intervals: List[Tuple[int, int, bool]] = []
    last = 0
    for start, end in sorted(placeholder_dict.values(), key=lambda pair: pair[0]):
        if start > last:
            intervals.append((last, start, False))
        intervals.append((start, end, True))
        last = end
    if last < total_len:
        intervals.append((last, total_len, False))

    placeholder_caches: List[DynamicCache] = []
    prefix_caches: List[DynamicCache] = []
    for start, end, is_placeholder in intervals:
        segment = cache.slice(start=start, end=end)
        segment_length = max(end - start, 0)
        _set_seen_tokens(segment, segment_length)
        if is_placeholder:
            placeholder_caches.append(segment)
        else:
            prefix_caches.append(segment)
    return placeholder_caches, prefix_caches


def _elementwise_binary_op(
    cache: DynamicCache,
    other: DynamicCache,
    op,
) -> DynamicCache:
    """Apply an elementwise binary op to two caches layer-by-layer."""
    _ensure_same_layout(cache, other)
    if not _LAYERWISE_CACHE_OPS:
        base_stack = _stack_cache_tensors(cache)
        other_stack = _stack_cache_tensors(other)
        if base_stack is not None and other_stack is not None:
            result = _copy_cache(cache)
            key_stack = op(base_stack[0], other_stack[0])
            value_stack = op(base_stack[1], other_stack[1])
            _assign_stack_to_cache(result, key_stack, value_stack)
            _set_seen_tokens(result, key_stack.shape[-2])
            return result

    result = type(cache)()
    if _is_layered_cache(cache):
        result.layers = []
    else:
        result.key_cache = []
        result.value_cache = []
    for idx in range(_get_layer_count(cache)):
        key_a, value_a = _get_layer_kv(cache, idx)
        key_b, value_b = _get_layer_kv(other, idx)
        if _layer_is_empty(key_a):
            new_key = _clone_tensor_or_empty(key_b)
            new_value = _clone_tensor_or_empty(value_b)
        elif _layer_is_empty(key_b):
            new_key = _clone_tensor_or_empty(key_a)
            new_value = _clone_tensor_or_empty(value_a)
        else:
            new_key = op(key_a, key_b)
            new_value = op(value_a, value_b)
        if _is_layered_cache(cache):
            layer = copy.deepcopy(cache.layers[idx])
            layer.keys = new_key
            layer.values = new_value
            if hasattr(layer, "is_initialized"):
                layer.is_initialized = not _layer_is_empty(new_key)
            if hasattr(layer, "dtype") and isinstance(new_key, torch.Tensor):
                layer.dtype = new_key.dtype
            if hasattr(layer, "device") and isinstance(new_key, torch.Tensor):
                layer.device = new_key.device
            result.layers.append(layer)
        else:
            result.key_cache.append(new_key)
            result.value_cache.append(new_value)
    _set_seen_tokens(result, _safe_seq_len(cache))
    return result


def _split_cache(cache: DynamicCache, sizes: Sequence[int]) -> List[DynamicCache]:
    """Split a cache into multiple segments by the given lengths (sum sizes)."""
    offsets = []
    start = 0
    for size in sizes:
        offsets.append((start, start + size))
        start += size
    return [cache.slice(start=s, end=e) for s, e in offsets]


# --- KVCMAS: per-anchor SVD compression of deltas -----------------------
# The full delta tensor has shape [L_layers, B, H, S, D].
_SVD_LOCK = threading.Lock()


def _lkv_chain_delta() -> bool:
    """Path-aware (CHAINED) delta correction: the anchor holds one delta per graph EDGE. """
    return os.environ.get("KVCMAS_CHAIN_DELTA", "1").strip().lower() not in ("0", "false", "no", "off")


def _is_question_span(ph_id: Any) -> bool:
    """Whether a placeholder id names the shared QUESTION span. """
    return "user_question" in str(ph_id)



def _consume_responses_enabled() -> bool:
    """Extend layer-wise base consumption to RELAYED-OUTPUT placeholders. """
    return os.environ.get("KVCMAS_CONSUME_RESPONSES", "0").strip().lower() in ("1", "true", "yes", "on")


def _ph_reader_count(llm_cls, ph_id: str) -> int:
    """How many agents' prompts contain ``ph_id``, i.e. """
    n = 0
    for _nid, store in (getattr(llm_cls, "_shared_kv_cache_memory", {}) or {}).items():
        if isinstance(store, dict) and isinstance(store.get("placeholder_info"), dict):
            if ph_id in store["placeholder_info"]:
                n += 1
    return max(1, n)


def _ph_delta_enabled(ph_id: str) -> bool:
    """Per-PLACEHOLDER delta switch, for isolating which span the correction damages. """
    skip = os.environ.get("KVCMAS_PH_DELTA_SKIP", "").strip()
    if not skip:
        return True
    return not any(tok and tok in (ph_id or "") for tok in skip.split(","))


def _pf_delta_enabled() -> bool:
    """Prefix (pf) delta correction — default OFF: the prefix is reused rotate-only. """
    return os.environ.get("KVCMAS_PF_DELTA", "0").strip().lower() in ("1", "true", "yes", "on")


def _svd_compress_delta(delta: torch.Tensor, rank: int) -> Dict[str, Any]:
    """Compress a 5-D delta tensor via per-(layer, batch) truncated SVD. """
    if delta.ndim != 5:
        raise ValueError(f"Expected 5-D delta, got shape {tuple(delta.shape)}")
    L, B, H, S, D = delta.shape
    HD = H * D
    # [L, B, H, S, D] -> [L, B, S, HD]
    M = delta.permute(0, 1, 3, 2, 4).reshape(L, B, S, HD)
    eff_rank = max(1, min(rank, S, HD))
    # Randomized truncated SVD in fp32 for numerical stability. Serialized: the
    # underlying torch.linalg.qr lazy init is not thread-safe (see _SVD_LOCK).
    with _SVD_LOCK:
        U, Sv, V = torch.svd_lowrank(M.float(), q=eff_rank)
    # U: [L, B, S, r], Sv: [L, B, r], V: [L, B, HD, r]
    US = U * Sv.unsqueeze(-2)              # [L, B, S, r]
    Vt = V.transpose(-1, -2).contiguous()  # [L, B, r, HD]
    return {
        "US": US.to(delta.dtype).contiguous(),
        "Vt": Vt.to(delta.dtype),
        "shape": (L, B, H, S, D),
        "rank": eff_rank,
    }


def _materialize_delta(stored: Any, slice_end: Optional[int] = None) -> torch.Tensor:
    """Return a [L, B, H, S, D] delta tensor from either a raw tensor or SVD factors. """
    if isinstance(stored, torch.Tensor):
        if slice_end is not None:
            return stored[..., :slice_end, :]
        return stored
    # SVD-compressed form
    US = stored["US"]
    Vt = stored["Vt"]
    L, B, H, S, D = stored["shape"]
    if slice_end is not None and slice_end < S:
        US = US[..., :slice_end, :]
        S = slice_end
    out_dtype = US.dtype
    # fp16 matmul is unsupported on CPU; upcast there. On GPU half matmul is fine,
    # but accumulating in fp32 also matches the downstream weighted-sum precision.
    if US.dtype == torch.float16 and not US.is_cuda:
        flat = torch.matmul(US.float(), Vt.float())
        flat = flat.to(out_dtype)
    else:
        flat = torch.matmul(US, Vt)  # [L, B, S, HD]
    return flat.view(L, B, S, H, D).permute(0, 1, 3, 2, 4).contiguous()


def preview_corr_mag(segments, n_layers_sample: int = 3):
    """Per-span relative correction magnitude from a few sampled layers, BEFORE materialization consumes anything. """
    out = []
    for s in segments:
        idx = s.get("anchor_index")
        if not idx or not s.get("key_field"):
            continue
        L = len(s["key_src"])
        layers = list(range(1, L))
        d2 = b2 = 0.0
        n = int(s["out_len"])
        n_d = min(int(s.get("cover_len") or n), n)
        w_key, w_value = s["w_key"], s["w_value"]
        for l in layers:
            if l == 0:
                continue
            idx_l = idx[l] if isinstance(idx[0], (list, tuple)) else idx
            wk_l = w_key[:, 0] if w_key.shape[1] == 1 else w_key[:, l]
            wv_l = w_value[:, 0] if w_value.shape[1] == 1 else w_value[:, l]
            fk = _fused_delta_flat(s["anchors"], idx_l, s["key_field"], wk_l, l, n_d)
            fv = _fused_delta_flat(s["anchors"], idx_l, s["value_field"], wv_l, l, n_d)
            if fk is None or fv is None:
                return []          # raw deltas / unmapped: no cheap preview, gate abstains
            d2 += float(fk.float().pow(2).sum()) + float(fv.float().pow(2).sum())
            dd = s["drop"]
            b2 += float(s["key_src"][l][..., dd:dd + n, :].float().pow(2).sum())
            b2 += float(s["value_src"][l][..., dd:dd + n, :].float().pow(2).sum())
            del fk, fv
        if b2 > 0:
            out.append((n, (d2 / b2) ** 0.5))
    return out


def _fused_materialize_enabled() -> bool:
    """Fused low-rank delta materialization (default ON). ``KVCMAS_FUSED_MATERIALIZE=0``
    restores the per-anchor path, which is the exact fallback for raw deltas anyway."""
    return os.environ.get("KVCMAS_FUSED_MATERIALIZE", "1").strip().lower() not in ("0", "false", "no", "off")


def _zscore_anchors(s: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Standardize a per-anchor distance tensor along the ANCHOR axis (dim 0). """
    mu = s.mean(dim=0, keepdim=True)
    sd = (s - mu).pow(2).mean().sqrt().clamp_min(eps)
    return (s - mu) / sd


def _anchor_pad_enabled() -> bool:
    """Partial anchor coverage (default ON). """
    return os.environ.get("KVCMAS_ANCHOR_PAD", "1").strip().lower() not in ("0", "false", "no", "off")


def _pad_rows_zero(t: torch.Tensor, n: int, dim: int = -2) -> torch.Tensor:
    """Zero-extend ``t`` along ``dim`` to ``n`` rows (no-op when already ≥ ``n``). """
    have = int(t.shape[dim])
    if have >= n:
        return t
    shape = list(t.shape)
    shape[dim] = n - have
    return torch.cat([t, t.new_zeros(shape)], dim=dim)


_UNSUPPORTED_W = object()  # sentinel: weight layout the fused path cannot map


def _us_weight(w: Optional[torch.Tensor], S: int):
    """Reshape an anchor weight so it can scale ``US [B, S, r]``. """
    if w is None:
        return None
    n = w.numel()
    if n == 1:
        return w.reshape(1, 1, 1).float()
    flat = w.reshape(-1)
    if n >= S:
        return flat[:S].reshape(1, S, 1).float()   # per-token
    return _UNSUPPORTED_W


def _fused_delta_flat(
    anchors: Any,
    idx_l: Any,
    field: str,
    weights_l: torch.Tensor,
    layer: int,
    slice_end: Optional[int],
) -> Optional[torch.Tensor]:
    """``Σ_i w_i · δ_i[layer]`` as ONE matmul, returned FLAT as ``[B, S, H*D]``. """
    us_parts: List[torch.Tensor] = []
    vt_parts: List[torch.Tensor] = []
    for vi, ai in enumerate(idx_l):
        stored = anchors[ai][field]
        if isinstance(stored, torch.Tensor):
            return None  # raw delta (svd_rank<=0): no low-rank form to fuse
        US = stored["US"][layer]  # [B, S, r]
        Vt = stored["Vt"][layer]  # [B, r, HD]
        if slice_end is not None and slice_end < US.shape[-2]:
            US = US[..., :slice_end, :]
        w_us = _us_weight(weights_l[vi] if weights_l is not None else None, US.shape[-2])
        if w_us is _UNSUPPORTED_W:
            return None  # unrecognized weight layout: use the per-anchor path
        if w_us is not None:
            # Scaling US's row s by w_s scales delta row s by w_s, since delta[:, s, :] = US[:, s,
            # so a per-token weight on US applies unchanged to the delta row.
            US = (US.float() * w_us).to(Vt.dtype)
        elif US.dtype != Vt.dtype:
            US = US.to(Vt.dtype)
        us_parts.append(US)
        vt_parts.append(Vt)
    if not us_parts:
        return None
    if len(us_parts) == 1:
        US_cat, Vt_cat = us_parts[0], vt_parts[0]
    else:
        US_cat = torch.cat(us_parts, dim=-1)   # [B, S, k*r]
        Vt_cat = torch.cat(vt_parts, dim=-2)   # [B, k*r, HD]
    if US_cat.dtype != Vt_cat.dtype:
        US_cat = US_cat.to(Vt_cat.dtype)
    if US_cat.dtype == torch.float16 and not US_cat.is_cuda:
        return torch.matmul(US_cat.float(), Vt_cat.float()).to(US_cat.dtype)
    return torch.matmul(US_cat, Vt_cat)        # [B, S, HD]


def _add_flat_delta_(dst: torch.Tensor, flat: torch.Tensor) -> None:
    """``dst[B,H,S,D] += flat[B,S,H*D]`` with no transpose copy. """
    B, H, S, D = dst.shape
    view = dst.permute(0, 2, 1, 3)             # [B, S, H, D] view of dst
    S_d = int(flat.shape[-2])
    if S_d < S:
        # Partial anchor coverage: the delta only spans the first S_d rows, the rest keep the
        # reused base (delta 0).
        view = view[:, :S_d]
    view.add_(flat.view(B, S_d, H, D).to(dst.dtype))


def _materialize_delta_layer(
    stored: Any, layer: int, slice_end: Optional[int] = None
) -> torch.Tensor:
    """One layer ``[B, H, S, D]`` of a delta, from a raw tensor or SVD factors. """
    if isinstance(stored, torch.Tensor):
        t = stored[layer]
        return t[..., :slice_end, :] if slice_end is not None else t
    US = stored["US"][layer]  # [B, S, r]
    Vt = stored["Vt"][layer]  # [B, r, HD]
    _, B, H, S, D = stored["shape"]
    if slice_end is not None and slice_end < S:
        US = US[..., :slice_end, :]
        S = slice_end
    if US.dtype == torch.float16 and not US.is_cuda:
        flat = torch.matmul(US.float(), Vt.float()).to(US.dtype)
    else:
        flat = torch.matmul(US, Vt)  # [B, S, HD]
    return flat.view(B, S, H, D).permute(0, 2, 1, 3).contiguous()  # [B, H, S, D]


class _LowRankBase:
    """Lazy low-rank view of a stored base embedding (``ph_key_embedding`` / ``ph_value_embedding``, shape ``[L, B, H, S, D]``). """
    __slots__ = ("_stored",)

    def __init__(self, stored: Dict[str, Any]) -> None:
        self._stored = stored

    def __getitem__(self, layer: int) -> torch.Tensor:
        return _materialize_delta_layer(self._stored, int(layer))

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self._stored["shape"])


_PATH_SEEN: Dict[str, int] = {}


def _path_note(tag: str) -> None:
    """Log the FIRST time each code path is taken, and count every time after. """
    n = _PATH_SEEN.get(tag, 0)
    _PATH_SEEN[tag] = n + 1
    if n == 0:
        logger.info(f"[LKV-PATH] first use: {tag}")


def path_counts_and_reset() -> Dict[str, int]:
    """Drain the path counters (batched vs fallback, fused vs per-anchor)."""
    out = dict(_PATH_SEEN)
    _PATH_SEEN.clear()
    return out


def _sims_batch_size() -> int:
    """Anchors evaluated per pass in the similarity loop (``KVCMAS_SIMS_BATCH``). """
    raw = os.environ.get("KVCMAS_SIMS_BATCH", "8")
    try:
        return max(1, int(raw))
    except ValueError:
        # Do NOT swallow.
        logger.warning(f"[LKV] KVCMAS_SIMS_BATCH={raw!r} is not an int — using 8")
        return 8


def _sims_batch_bytes() -> int:
    """Byte budget for ONE expanded anchor block (``KVCMAS_SIMS_BATCH_BYTES``, default 1 GiB). """
    try:
        return max(0, int(os.environ.get("KVCMAS_SIMS_BATCH_BYTES", str(1 << 30))))
    except ValueError:
        return 1 << 30


def _sims_chunk_fit(chunk: int, anchors, field: str, count: int, ref: torch.Tensor) -> int:
    """Largest chunk whose expanded block fits :func:`_sims_batch_bytes`."""
    budget = _sims_batch_bytes()
    if budget <= 0 or chunk <= 1:
        return chunk
    B, H, _, D = ref.shape
    per = int(B) * int(H) * int(count) * int(D) * int(ref.element_size())
    # K and V are expanded together and both stay live across the metrics.
    blocks = 2
    # The tail is a SEPARATE tensor whenever any anchor stores more rows than the segment
    # needs; when every span matches exactly, tail IS head and costs nothing extra.
    for a in anchors:
        stored = getattr(a[field], "_stored", None)
        if isinstance(stored, dict) and "US" in stored:
            if int(stored["US"][0].shape[1]) != int(count):
                blocks = 4
                break
    per *= blocks
    if per <= 0:
        return chunk
    return max(1, min(int(chunk), int(budget // per)))


def _pad_rank_us(u: torch.Tensor, r_max: int) -> torch.Tensor:
    """``[B, S, r]`` -> ``[B, S, r_max]`` with zero columns appended."""
    r = int(u.shape[-1])
    if r == r_max:
        return u
    return torch.nn.functional.pad(u, (0, r_max - r))


def _pad_rank_vt(v: torch.Tensor, r_max: int) -> torch.Tensor:
    """``[B, r, H, D]`` -> ``[B, r_max, H, D]`` with zero rows appended."""
    r = int(v.shape[1])
    if r == r_max:
        return v
    return torch.nn.functional.pad(v, (0, 0, 0, 0, 0, r_max - r))


def _expand_anchor_chunk(anchors_chunk, field: str, layer: int,
                         count: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """``(head, tail)`` blocks of ``count`` rows each, one batched GEMM per block. """
    us_head, us_tail, vt = [], [], []
    same = True
    r_max = 0
    for a in anchors_chunk:
        stored = getattr(a[field], "_stored", None)
        if not isinstance(stored, dict) or "US" not in stored:
            return None
        US = stored["US"][layer]                             # [B, S_a, r]
        S_a = int(US.shape[1])
        if S_a < count:
            return None
        us_head.append(US[:, :count, :])
        us_tail.append(US[:, S_a - count:, :])
        same = same and (S_a == count)
        v = stored["Vt"][layer]                              # [B, r, HD]
        _, B_, H_, _, D_ = stored["shape"]
        vt.append(v.reshape(v.shape[0], v.shape[1], H_, D_))  # [B, r, H, D]
        r_max = max(r_max, int(US.shape[-1]))
    if any(int(u.shape[-1]) != r_max for u in us_head):
        # Ragged rank: pad the rank axis with zeros (US columns, Vt rows).
        us_head = [_pad_rank_us(u, r_max) for u in us_head]
        us_tail = [_pad_rank_us(u, r_max) for u in us_tail]
        vt = [_pad_rank_vt(v, r_max) for v in vt]
    Vt = torch.stack(vt, dim=0)                              # [k, B, r, H, D]
    head = torch.einsum("kbsr,kbrhd->kbhsd", torch.stack(us_head, dim=0), Vt)
    if same:
        return head, head
    tail = torch.einsum("kbsr,kbrhd->kbhsd", torch.stack(us_tail, dim=0), Vt)
    return head, tail


def _stack_base_factors(US: List[torch.Tensor], Vt: List[torch.Tensor],
                        raw: List[torch.Tensor], rank: int,
                        shape: Tuple[int, ...]) -> Any:
    """Assemble a stored base embedding from PER-LAYER pieces produced upstream. """
    if rank and rank > 0:
        return _LowRankBase({
            "US": torch.stack(US),               # [L, B, S, r]
            "Vt": torch.stack(Vt),               # [L, B, r, HD]
            "shape": shape,
            "rank": int(US[0].shape[-1]),
        })
    return torch.stack(raw)                       # [L, B, H, S, D]


class StreamingCorrectionCache(DynamicCache):
    """Reference-based KV cache: keeps the shared base by segment, never copied or merged, and
    reconstructs each layer's rotated and corrected KV on demand."""

    def __init__(self) -> None:
        super().__init__()
        self._segments: List[Dict[str, Any]] = []
        self._base_len = 0

    @classmethod
    def from_segments(cls, segments: List[Dict[str, Any]]):
        c = cls()
        c._segments = segments
        c._base_len = int(sum(int(s["out_len"]) for s in segments))
        return c

    def get_seq_length(self, layer_idx: int = 0) -> int:
        new = 0
        if layer_idx < len(self.key_cache):
            kc = self.key_cache[layer_idx]
            if isinstance(kc, torch.Tensor):
                new = kc.shape[-2]
        return self._base_len + new

    def get_max_cache_shape(self):
        return None

    def get_max_length(self):
        return None

    def slice(self, start=None, end=None) -> DynamicCache:
        """Slice in LOGICAL positions, returning a plain ``DynamicCache``. """
        base = self._base_len
        s = 0 if start is None else max(0, int(start) - base)
        e = None if end is None else max(0, int(end) - base)
        out = DynamicCache()
        out.key_cache = [k[..., s:e, :].contiguous() for k in self.key_cache]
        out.value_cache = [v[..., s:e, :].contiguous() for v in self.value_cache]
        out._seen_tokens = out.key_cache[0].shape[-2] if out.key_cache else 0
        return out

    def materialize(self, consume_base: bool = False) -> DynamicCache:
        """Assemble every layer once into a plain ``DynamicCache`` — original KVComm. """
        out = DynamicCache()
        n_layers = len(self._segments[0]["key_src"]) if self._segments else 0
        ks, vs = [], []
        # DELTA RECONSTRUCTION phase: expand the low-rank delta and assemble each layer once.
        for l in range(n_layers):
            # Per-LAYER probe.
            k, v = self._assemble_layer(l)
            ks.append(k.contiguous())
            vs.append(v.contiguous())
            if consume_base:
                # REPLACE, do not duplicate: layer l of the base has just been folded into layer l
                # of the corrected cache and is never read again, so drop it now.
                for _s in self._segments:
                    if _s.get("consumable"):
                        _s["key_src"][l] = None
                        _s["value_src"][l] = None
        # Decode tokens already in key_cache (if any) belong AFTER the base.
        for l in range(min(len(self.key_cache), n_layers)):
            if self.key_cache[l] is not None and self.key_cache[l].shape[-2] > 0:
                ks[l] = torch.cat([ks[l], self.key_cache[l]], dim=-2)
                vs[l] = torch.cat([vs[l], self.value_cache[l]], dim=-2)
        out.key_cache, out.value_cache = ks, vs
        out._seen_tokens = ks[0].shape[-2] if ks else 0
        return out

    def _assemble_layer(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build layer ``layer_idx`` of the base = concat over segments of ``rotate(src[drop:drop+out_len]) + Σ_i w_i·δ_i[l]``. """
        ks: List[torch.Tensor] = []
        vs: List[torch.Tensor] = []
        for s in self._segments:
            d, n = s["drop"], s["out_len"]
            k = s["key_src"][layer_idx][..., d:d + n, :]
            v = s["value_src"][layer_idx][..., d:d + n, :]
            cos = s["cos"]
            if cos is not None:
                cos_, sin_ = cos.unsqueeze(1), s["sin"].unsqueeze(1)
                # In-place rotary: peak holds 2×[layer] (rot + k*cos) instead of 4× (k*cos, rot,
                # rot*sin, sum) — halves the per-step rotate transient on the 64k base.
                rot = _rotate_half(k)
                rot.mul_(sin_)
                k = k * cos_
                k.add_(rot)
                del rot
            idx = s["anchor_index"]
            if idx and layer_idx != 0:
                # Selective anchoring: a flat [V] list (same anchors every layer) or a per-layer
                # [L][k] list-of-lists -> use THIS layer's anchor set.
                idx_l = idx[layer_idx] if isinstance(idx[0], (list, tuple)) else idx
                # Partial anchor coverage: the weights and the delta span only the
                # first `n_d` rows of the segment; the rest keep the reused base.
                n_d = min(int(s.get("cover_len") or n), n)
                w_key, w_value = s["w_key"], s["w_value"]
                wk_l = w_key[:, 0] if w_key.shape[1] == 1 else w_key[:, layer_idx]
                wv_l = w_value[:, 0] if w_value.shape[1] == 1 else w_value[:, layer_idx]
                anchors = s["anchors"]
                fused_k = fused_v = None
                if _fused_materialize_enabled():
                    # ONE GEMM per K/V over the rank-concatenated hot anchors, added through a
                    # permuted view (no transpose copy).
                    fused_k = _fused_delta_flat(anchors, idx_l, s["key_field"], wk_l, layer_idx, n_d)
                    fused_v = _fused_delta_flat(anchors, idx_l, s["value_field"], wv_l, layer_idx, n_d)
                if fused_k is not None and fused_v is not None:
                    _path_note(f"delta FUSED GEMM (hot={len(idx_l)})")
                    if not k.is_contiguous():
                        k = k.contiguous()
                    if not v.is_contiguous():
                        v = v.contiguous()
                    _add_flat_delta_(k, fused_k)
                    _add_flat_delta_(v, fused_v)
                    del fused_k, fused_v
                else:
                    # Raw (un-factorized) deltas: per-anchor accumulate.
                    _path_note("delta per-anchor FALLBACK (raw delta or unmapped weight layout)")
                    acc_k = acc_v = None
                    for vi, ai in enumerate(idx_l):
                        dk = _materialize_delta_layer(anchors[ai][s["key_field"]], layer_idx, slice_end=n_d)
                        tk = wk_l[vi] * dk
                        acc_k = tk if acc_k is None else acc_k + tk
                        del dk, tk
                        dv = _materialize_delta_layer(anchors[ai][s["value_field"]], layer_idx, slice_end=n_d)
                        tv = wv_l[vi] * dv
                        acc_v = tv if acc_v is None else acc_v + tv
                        del dv, tv
                    # Under partial coverage acc_* is shorter than the segment; the missing rows
                    # are delta 0, materialized here rather than left as a narrowed view.
                    acc_k = _pad_rows_zero(acc_k, n)
                    acc_v = _pad_rows_zero(acc_v, n)
                    k = k + acc_k.to(k.dtype)
                    v = v + acc_v.to(v.dtype)
                    del acc_k, acc_v
            ks.append(k.contiguous())
            vs.append(v.contiguous())
        return torch.cat(ks, dim=-2), torch.cat(vs, dim=-2)

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        # STREAMING decode path: this re-assembles the layer on EVERY step, which is the cost
        # chaining exists to remove (it materialises once instead and hands.
        new_k, new_v = super().update(key_states, value_states, layer_idx, cache_kwargs)
        base_k, base_v = self._assemble_layer(layer_idx)
        return torch.cat([base_k, new_k], dim=-2), torch.cat([base_v, new_v], dim=-2)

    # Fused decode hooks: expose the base by segment (raw K/V + per-anchor SVD factors +
    # per-segment RoPE offset).

    def flra_supported(self) -> bool:
        """The fused path applies only when every anchored segment stores SVD factors (US/Vt). """
        if not self._segments:
            return False
        for s in self._segments:
            if s.get("key_field") and s.get("anchor_index"):
                idx = s["anchor_index"]
                first = idx[0][0] if isinstance(idx[0], (list, tuple)) else idx[0]
                stored = s["anchors"][first][s["key_field"]]
                if isinstance(stored, torch.Tensor):
                    return False
        return True

    def _seg_anchor_layer(self, s: Dict[str, Any], layer_idx: int):
        """This segment's per-layer anchor index list, or ``None`` for a plain
        (un-anchored) segment or layer 0 (base only, matching ``_assemble_layer``)."""
        idx = s.get("anchor_index")
        if not idx or layer_idx == 0:
            return None
        return idx[layer_idx] if isinstance(idx[0], (list, tuple)) else idx

    def get_base_segments(self, layer_idx: int):
        """Per-segment raw base views for the kernel gather. """
        k_segs, v_segs, bounds, acc = [], [], [], 0
        for s in self._segments:
            d, n = s["drop"], s["out_len"]
            k_segs.append(s["key_src"][layer_idx][..., d:d + n, :])
            v_segs.append(s["value_src"][layer_idx][..., d:d + n, :])
            acc += n
            bounds.append(acc)
        return k_segs, v_segs, bounds

    def build_qvar(self, q: torch.Tensor) -> torch.Tensor:
        """``q_var[:, s] = R(offset_s)ᵀ·q`` per segment (identity where cos is None), so ``q_var[s]·k_raw = q·(R(offset_s)·k_raw)`` reproduces the per-segment."""
        qp = q.permute(0, 2, 1, 3)  # [B, 1, nh, D]
        outs = []
        for s in self._segments:
            cos = s["cos"]
            if cos is None:
                outs.append(qp)
            else:
                cos_ = cos.unsqueeze(1).to(q.dtype)  # [B,1,1,D] broadcast over nh
                sin_ = s["sin"].unsqueeze(1).to(q.dtype)
                outs.append((qp * cos_ - _rotate_half(qp) * sin_).to(q.dtype))
        return torch.stack(outs, dim=1).contiguous()  # [B, n_seg, 1, nh, D]

    def get_lr_factors(self, layer_idx: int, zero_width: int = 16):
        """Per-segment low-rank adapters folded the kvcmas way, matching ``_assemble_layer``'s ``Σ_i w_i · (US_i @ Vt_i)`` for each anchor i: lr_a[seg] = [ …."""
        ks0 = self._segments[0]["key_src"][layer_idx]
        nhk, D = ks0.shape[1], ks0.shape[3]
        dev, dt = ks0.device, ks0.dtype
        lr_k_a, lr_v_a, lr_k_b, lr_v_b = [], [], [], []
        for s in self._segments:
            n = s["out_len"]
            idx_l = self._seg_anchor_layer(s, layer_idx)
            if idx_l is None:
                lr_k_a.append(torch.zeros(1, n, zero_width, device=dev, dtype=dt))
                lr_v_a.append(torch.zeros(1, n, zero_width, device=dev, dtype=dt))
                lr_k_b.append(torch.zeros(nhk, zero_width, D, device=dev, dtype=dt))
                lr_v_b.append(torch.zeros(nhk, zero_width, D, device=dev, dtype=dt))
                continue
            wk, wv = s["w_key"], s["w_value"]
            wk_l = wk[:, 0] if wk.shape[1] == 1 else wk[:, layer_idx]  # [V,B,H,Sw,Dw]
            wv_l = wv[:, 0] if wv.shape[1] == 1 else wv[:, layer_idx]
            kf, vf = s["key_field"], s["value_field"]
            USk, USv, Vtk, Vtv = [], [], [], []
            for vi, ai in enumerate(idx_l):
                sk, sv = s["anchors"][ai][kf], s["anchors"][ai][vf]
                uk = sk["US"][layer_idx][..., :n, :]                 # [1, n, r]
                bk = self._vt_head_major(sk["Vt"][layer_idx], nhk, D)  # [nhk, r, D]
                uk, bk = self._fold_weight(uk, bk, wk_l[vi], nhk, D, dt)
                uv = sv["US"][layer_idx][..., :n, :]
                bv = self._vt_head_major(sv["Vt"][layer_idx], nhk, D)
                uv, bv = self._fold_weight(uv, bv, wv_l[vi], nhk, D, dt)
                # Partial coverage: the anchor is shorter than the segment, so US carries fewer
                # rows than the kernel's `n`.
                uk = _pad_rows_zero(uk, n)
                uv = _pad_rows_zero(uv, n)
                USk.append(uk); Vtk.append(bk); USv.append(uv); Vtv.append(bv)
            # Σr = (#anchors this layer)·r can be any value (selective anchoring picks a data-
            # dependent, possibly odd count); the kernel needs lora_dim a multiple.
            ka, kb = self._pad_slab(torch.cat(USk, dim=-1), torch.cat(Vtk, dim=1))
            va, vb = self._pad_slab(torch.cat(USv, dim=-1), torch.cat(Vtv, dim=1))
            lr_k_a.append(ka); lr_k_b.append(kb)   # [1,n,Σr8] / [nhk,Σr8,D]
            lr_v_a.append(va); lr_v_b.append(vb)
        return lr_k_a, lr_v_a, lr_k_b, lr_v_b

    @staticmethod
    def _pad_slab(a: torch.Tensor, b: torch.Tensor, slab: int = 16):
        """Zero-pad the rank axis of ``a`` ([B,n,W]) and ``b`` ([nhk,W,D]) up to a
        multiple of ``slab`` so the kernel's per-segment lora_dim is slab-aligned."""
        W = a.shape[-1]
        W2 = (W + slab - 1) // slab * slab
        if W2 == W:
            return a.contiguous(), b.contiguous()
        pa = torch.zeros(a.shape[0], a.shape[1], W2 - W, device=a.device, dtype=a.dtype)
        pb = torch.zeros(b.shape[0], W2 - W, b.shape[2], device=b.device, dtype=b.dtype)
        return torch.cat([a, pa], dim=-1).contiguous(), torch.cat([b, pb], dim=1).contiguous()

    @staticmethod
    def _vt_head_major(Vt: torch.Tensor, nhk: int, D: int) -> torch.Tensor:
        """One anchor's ``[B, r, H·D]`` SVD ``Vt`` (heads folded) -> ``[nhk, r, D]``
        head-major (B==1 at decode)."""
        _, r, _ = Vt.shape
        return Vt.view(1, r, nhk, D)[0].permute(1, 0, 2).contiguous()  # [nhk, r, D]

    @staticmethod
    def _fold_weight(uk: torch.Tensor, bk: torch.Tensor, w: torch.Tensor,
                     nhk: int, D: int, dt: torch.dtype):
        """Fold one anchor's per-channel kvcmas weight ``w`` (``[B,H,Sw,Dw]``, B==1) into ``US`` (``uk``, per-seq/scalar weight) or head-major ``Vt`` (``bk``."""
        w = w[0]                       # drop batch -> [H, Sw, Dw]
        H_, Sw, Dw = w.shape
        if Dw > 1:                     # per-(head,channel) -> scale Vt (lr_b)
            bk = (bk.float() * w.reshape(H_, 1, Dw)).to(dt)   # [nhk,r,D]·[nhk,1,D]
            uk = uk.to(dt)
        elif Sw > 1:                   # per-seq, head-shared -> scale US (lr_a)
            uk = (uk.float() * w.reshape(1, Sw, 1)).to(dt)    # [1,n,r]·[1,n,1]
            bk = bk.to(dt)
        else:                          # scalar
            uk = (uk.float() * float(w.reshape(-1)[0])).to(dt)
            bk = bk.to(dt)
        return uk, bk


def _svd_one_layer(delta_l: torch.Tensor, rank: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rank-r truncated SVD of a single layer's delta [B, H, S, D] -> (US, Vt)."""
    B_, H_, S_, D_ = delta_l.shape
    M = delta_l.permute(0, 2, 1, 3).reshape(B_, S_, H_ * D_)  # [B, S, HD]
    eff_rank = max(1, min(rank, S_, H_ * D_))
    U, Sv, V = torch.svd_lowrank(M.float(), q=eff_rank)
    US = (U * Sv.unsqueeze(-2)).to(delta_l.dtype).contiguous()  # [B, S, r]
    Vt = V.transpose(-1, -2).contiguous().to(delta_l.dtype)      # [B, r, HD]
    return US, Vt


def _match_frame() -> str:
    """RoPE frame in which the KEY-side anchor matching (weights) is done -- ``KVCMAS_MATCH_FRAME``. """
    v = os.environ.get("KVCMAS_MATCH_FRAME", "canonical").strip().lower()
    return v if v in ("canonical", "reader", "legacy") else "canonical"


def _svd_chunk_bytes() -> int:
    """fp32 byte budget for one batched ``svd_lowrank`` call (``KVCMAS_SVD_CHUNK_BYTES``,
    default 1 GiB). ``0`` disables layer batching and restores the per-layer path."""
    raw = os.environ.get("KVCMAS_SVD_CHUNK_BYTES", str(1 << 30))
    try:
        return max(0, int(raw))
    except ValueError:
        return 1 << 30


def _orth_cholqr(Y: torch.Tensor) -> torch.Tensor:
    """Orthonormalize the columns of ``Y`` ``[N, m, q]`` by Cholesky-QR (two passes). """
    for _ in range(2):
        G = Y.transpose(-1, -2) @ Y
        eye = torch.eye(G.shape[-1], device=G.device, dtype=G.dtype)
        G = G + eye * (1e-6 * G.diagonal(dim1=-2, dim2=-1).mean(-1, keepdim=True).unsqueeze(-1))
        L, info = torch.linalg.cholesky_ex(G)
        if bool((info != 0).any()):
            return torch.linalg.qr(Y).Q
        Y = torch.linalg.solve_triangular(L.transpose(-1, -2), Y, upper=True, left=False)  # Y @ L^{-T}
    return Y


def _rsvd_batched(M: torch.Tensor, rank: int, niter: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rank-``r`` randomized range factorization of a batch ``M [N, S, HD]`` (fp32) -> ``(US [N, S, r], Vt [N, r, HD])`` with ``US @ Vt == Q Q^T M``. """
    N, S_, HD = M.shape
    r = max(1, min(int(rank), S_, HD))
    omega = torch.randn(HD, r, device=M.device, dtype=M.dtype)
    Q = _orth_cholqr(M @ omega)                                  # [N, S, r]
    for _ in range(max(0, int(niter))):
        Z = _orth_cholqr(M.transpose(-1, -2) @ Q)               # [N, HD, r]
        Q = _orth_cholqr(M @ Z)
    Vt = Q.transpose(-1, -2) @ M                                 # [N, r, HD]
    return Q, Vt


class _LayerSVDBatcher:
    """Truncated SVD over several layers in ONE batched call (``_rsvd_batched``). """

    def __init__(self, rank: int) -> None:
        self.rank = int(rank)
        self._pending: List[torch.Tensor] = []
        self._pending_bytes = 0
        self.US: List[torch.Tensor] = []
        self.Vt: List[torch.Tensor] = []
        self._budget = _svd_chunk_bytes()

    def add(self, layer: torch.Tensor) -> None:
        """Queue one layer ``[B, H, S, D]``; flushes when the budget would overflow."""
        fp32_bytes = layer.numel() * 4
        if self._budget <= 0:
            u, v = _svd_one_layer(layer, self.rank)
            self.US.append(u); self.Vt.append(v)
            return
        if self._pending and self._pending_bytes + fp32_bytes > self._budget:
            self.flush()
        self._pending.append(layer)
        self._pending_bytes += fp32_bytes

    def flush(self) -> None:
        if not self._pending:
            return
        C = len(self._pending)
        B_, H_, S_, D_ = self._pending[0].shape
        dt = self._pending[0].dtype
        # [C, B, H, S, D] -> [C*B, S, H*D] in fp32, one batched call.
        M = torch.stack(self._pending).permute(0, 1, 3, 2, 4).reshape(C * B_, S_, H_ * D_).float()
        US, Vt = _rsvd_batched(M, self.rank)
        del M
        r = US.shape[-1]
        US = US.to(dt).reshape(C, B_, S_, r).contiguous()
        Vt = Vt.to(dt).reshape(C, B_, r, H_ * D_).contiguous()
        for c in range(C):
            self.US.append(US[c]); self.Vt.append(Vt[c])
        self._pending = []
        self._pending_bytes = 0

    def finish(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        self.flush()
        return self.US, self.Vt


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return torch.cat((-x[..., d:], x[..., :d]), dim=-1)


def _svd_compress_streaming(
    real_cache: "DynamicCache",
    real_start: int,
    real_end: int,
    base_cache: "DynamicCache",
    base_start: int,
    base_end: Optional[int],
    key_cos: Optional[torch.Tensor],
    key_sin: Optional[torch.Tensor],
    rank: int,
) -> Tuple[Any, Any]:
    """Streaming per-(segment, layer) anchor compression — the single-use path. """
    rk, rv = real_cache.key_cache, real_cache.value_cache
    bk, bv = base_cache.key_cache, base_cache.value_cache
    L = len(rk)
    cos_k = key_cos.unsqueeze(1) if key_cos is not None else None  # [B,1,S,D] over heads
    sin_k = key_sin.unsqueeze(1) if key_sin is not None else None

    raw_k: List[torch.Tensor] = []
    raw_v: List[torch.Tensor] = []
    kb = _LayerSVDBatcher(rank) if rank > 0 else None
    vb = _LayerSVDBatcher(rank) if rank > 0 else None
    H_ = S_ = D_ = None
    with _SVD_LOCK:
        for l in range(L):
            # Pull ONLY layer l of the base to the real cache's device.
            base_k_l = bk[l][..., base_start:base_end, :].to(rk[l].device)
            if cos_k is not None:
                base_k_rot = base_k_l * cos_k + _rotate_half(base_k_l) * sin_k  # key rotation
            else:
                base_k_rot = base_k_l                                          # already rotated
            delta_k = rk[l][..., real_start:real_end, :] - base_k_rot
            B_, H_, S_, D_ = delta_k.shape
            base_v_l = bv[l][..., base_start:base_end, :].to(rv[l].device)      # value: no rotation
            delta_v = rv[l][..., real_start:real_end, :] - base_v_l
            if rank <= 0:
                raw_k.append(delta_k.clone())
                raw_v.append(delta_v.clone())
            else:
                kb.add(delta_k); vb.add(delta_v)      # batched across layers, flushed by budget
            del delta_k, delta_v, base_k_rot, base_k_l, base_v_l
        if rank > 0:
            USk, Vtk = kb.finish()
            USv, Vtv = vb.finish()

    if rank <= 0:
        key_delta: Any = torch.stack(raw_k)
        value_delta: Any = torch.stack(raw_v)
    else:
        key_delta = {"US": torch.stack(USk).contiguous(), "Vt": torch.stack(Vtk),
                     "shape": (L, USk[0].shape[0], H_, S_, D_), "rank": int(USk[0].shape[-1])}
        value_delta = {"US": torch.stack(USv).contiguous(), "Vt": torch.stack(Vtv),
                       "shape": (L, USv[0].shape[0], H_, S_, D_), "rank": int(USv[0].shape[-1])}
    return key_delta, value_delta


def _install_dynamic_cache_extensions() -> None:
    """Monkey-patch DynamicCache with convenience methods used by KVCMAS."""
    if getattr(DynamicCache, "_kvcomm_extensions_installed", False):
        return

    DynamicCache._normalize_slice_indices = lambda self, start=None, end=None: _normalize_indices(self, start, end)
    DynamicCache.slice_ = lambda self, start=None, end=None: _slice_inplace(self, start, end)
    DynamicCache.slice = lambda self, start=None, end=None: _slice_functional(self, start, end)
    DynamicCache.slice_view = lambda self, start=None, end=None: _slice_view(self, start, end)
    DynamicCache.concat_ = lambda self, other: _concat_inplace(self, _ensure_cache_sequence(other))
    DynamicCache.concat = lambda self, other: _concat_functional(self, _ensure_cache_sequence(other))
    DynamicCache.replace_ = lambda self, start, end, real: _replace_inplace(self, start, end, real)
    DynamicCache.replace = lambda self, start, end, real: _replace_functional(self, start, end, real)
    DynamicCache.select_indices = lambda self, indices: _select_indices(self, indices)
    DynamicCache.to = lambda self, device: _to_device(self, device)
    DynamicCache.copy = lambda self: _copy_cache(self)
    DynamicCache.split_cache_by_placeholders = lambda self, placeholder_dict: _split_cache_by_placeholders(
        self, placeholder_dict
    )
    DynamicCache.__add__ = lambda self, other: _elementwise_binary_op(self, other, torch.add)
    DynamicCache.__sub__ = lambda self, other: _elementwise_binary_op(self, other, torch.sub)
    DynamicCache.split = lambda self, sizes: _split_cache(self, sizes)
    DynamicCache._kvcomm_extensions_installed = True


_install_dynamic_cache_extensions()


def _clone_default(value: Any) -> Any:
    if isinstance(value, (dict, list, set, tuple)):
        return copy.deepcopy(value)
    return copy.copy(value) if hasattr(value, "__copy__") else value


def _cow_copy(obj: Any) -> Any:
    """Copy-on-write for the request scope: deep-copy container *structure* (dict/list/tuple) so a request can add/drop/replace anchor entries privately."""
    if isinstance(obj, dict):
        return {k: _cow_copy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_cow_copy(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_cow_copy(v) for v in obj)
    if torch.is_tensor(obj):
        return obj
    if isinstance(obj, _LowRankBase):
        # Read-only after creation, exactly like anchor tensors -- but it is a __slots__ class, so
        # the generic fallthrough below deep-copied its whole SVD.
        return obj
    return copy.deepcopy(obj)


class _ScopedDict(MutableMapping):
    """Request-scoped view over a shared dictionary with deferred commits."""

    def __init__(self, base: Dict[str, Any]):
        self._base = base
        self._local: Dict[str, Any] = {}

    def _ensure_local(self, key: str) -> None:
        if key in self._local:
            return
        if key in self._base:
            self._local[key] = _cow_copy(self._base[key])

    def __getitem__(self, key: str) -> Any:
        if key in self._local:
            value = self._local[key]
            if value is _DELETED:
                raise KeyError(key)
            return value
        if key in self._base:
            value = _cow_copy(self._base[key])
            self._local[key] = value
            if value is _DELETED:
                raise KeyError(key)
            return value
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        self._local[key] = value

    def __delitem__(self, key: str) -> None:
        self.pop(key)

    def __iter__(self) -> Iterable[str]:
        return iter(self.keys())

    def __len__(self) -> int:
        return len(self.keys())

    def keys(self) -> List[str]:
        merged = set(self._base.keys()) | set(self._local.keys())
        return [
            key
            for key in merged
            if self._local.get(key, None) is not _DELETED
        ]

    def items(self):
        for key in self.keys():
            yield key, self[key]

    def values(self):
        for _, value in self.items():
            yield value

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def setdefault(self, key: str, default: Any = None):
        if key in self._local:
            value = self._local[key]
            if value is _DELETED:
                new_value = _clone_default(default)
                self._local[key] = new_value
                return new_value
            return value
        if key in self._base:
            value = _cow_copy(self._base[key])
            self._local[key] = value
            return value
        new_value = _clone_default(default)
        self._local[key] = new_value
        return new_value

    def pop(self, key: str, default: Any = _MISSING) -> Any:
        self._ensure_local(key)
        if key not in self._local:
            if default is _MISSING:
                raise KeyError(key)
            return default
        value = self._local[key]
        if value is _DELETED:
            if default is _MISSING:
                raise KeyError(key)
            return default
        self._local[key] = _DELETED
        return value

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        if key in self._local:
            return self._local[key] is not _DELETED
        return key in self._base

    def commit(self) -> None:
        for key, value in self._local.items():
            if value is _DELETED:
                self._base.pop(key, None)
            else:
                self._base[key] = value
        self._local.clear()


class _RequestState:
    """Container tracking deferred mutations for a single request."""

    def __init__(
        self,
        request_uid: str,
        anchor_dict: Dict[str, Any],
        anchor_len_dict: Dict[str, Any],
        anchor_info_dict: Dict[str, Any],
        weight_dict: Dict[str, Any],
        anchors: Dict[str, Any],
        global_anchor_info_dict: Dict[str, Any],
    ):
        self.request_uid = request_uid
        self.anchor_dict = _ScopedDict(anchor_dict)
        self.anchor_len_dict = _ScopedDict(anchor_len_dict)
        self.anchor_info_dict = _ScopedDict(anchor_info_dict)
        self.weight_dict = _ScopedDict(weight_dict)
        self.anchors = _ScopedDict(anchors)
        self.global_anchor_info = _ScopedDict(global_anchor_info_dict)

    def commit(self) -> None:
        self.anchor_dict.commit()
        self.anchor_len_dict.commit()
        self.anchor_info_dict.commit()
        self.weight_dict.commit()
        self.anchors.commit()
        self.global_anchor_info.commit()


class KVCMASEngine:
    """Central coordinator for anchor-related KV cache interactions."""

    anchors: Dict[str, Any] = {}
    anchor_dict: Dict[str, Any] = {}
    anchor_len_dict: Dict[str, Any] = {}
    anchor_info_dict: Dict[str, Any] = {}
    weight_dict: Dict[str, Any] = {}
    global_anchor_info_dict: Dict[str, Any] = {}

    _request_lock = threading.Lock()
    _request_states: Dict[str, _RequestState] = {}
    _active_requests: set[str] = set()
    _staged_commits: List[_RequestState] = []

    def __init__(self, llm: "LLMChat"):
        self.llm = llm
        self._warning_prefix = "[KVCMASEngine]"

    def _log_warning(self, message: str) -> None:
        logger.opt(colors=True).warning("<yellow>{}</yellow> {}", self._warning_prefix, message)

    @staticmethod
    def _stack_cache_tensors(cache: DynamicCache) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.stack(cache.key_cache), torch.stack(cache.value_cache)

    @staticmethod
    def _placeholder_length(cache: DynamicCache) -> int:
        return cache.key_cache[0].shape[-2]

    def _rotate_segment_caches(self, segment_meta: Dict[str, Any]) -> Tuple[DynamicCache, DynamicCache]:
        rotated_placeholder = self.apply_rotary_pos_emb(
            segment_meta["ph_cache"],
            offset=segment_meta["start"] - segment_meta["drop_num"] + segment_meta["offset_before"],
            drop_num=segment_meta["drop_num"],
        )
        rotated_prefix = self.apply_rotary_pos_emb(
            segment_meta["pf_kv"],
            offset=segment_meta["offset_after"],
        )
        return rotated_placeholder, rotated_prefix

    def _segment_rope(self, seq_len: int, offset: int, sample_key: torch.Tensor):
        """RoPE cos/sin that shift a segment to absolute position ``offset`` — all positions get the relative rotation R(offset), matching apply_rotary."""
        rotate_emb = self.llm.model.model.rotary_emb
        position_ids = (
            torch.ones(seq_len, dtype=torch.long).unsqueeze(0).to(self.llm.model.device) * offset
        )
        return rotate_emb(sample_key, position_ids)

    @classmethod
    def _get_request_state(cls, request_uid: str) -> _RequestState:
        """Return or create a request-scoped state container under a lock."""
        if not request_uid:
            raise ValueError("request_uid must be provided for scoped anchor updates.")
        with cls._request_lock:
            state = cls._request_states.get(request_uid)
            if state is None:
                state = _RequestState(
                    request_uid,
                    cls.anchor_dict,
                    cls.anchor_len_dict,
                    cls.anchor_info_dict,
                    cls.weight_dict,
                    cls.anchors,
                    cls.global_anchor_info_dict,
                )
                cls._request_states[request_uid] = state
                cls._active_requests.add(request_uid)
            return state

    @classmethod
    def finalize_request(cls, request_uid: str, llm: Optional[Any] = None) -> None:
        """Stage a request state for commit; runs shared-cache drop when llm given."""
        if not request_uid:
            return
        with cls._request_lock:
            state = cls._request_states.pop(request_uid, None)
            if state is None:
                return
            if llm is not None:
                cls._maybe_drop_shared_cache(llm)
            cls._staged_commits.append(state)
            cls._active_requests.discard(request_uid)
            if not cls._active_requests:
                cls._commit_staged_states_locked()

    _DROP_BUCKETS = (
        # input_tags must die with the buckets it describes.
        "input", "input_ids", "input_drop_num", "input_tags",
        "response", "response_ids", "response_drop_num",
        "condition", "condition_ids", "condition_drop_num",
    )

    @classmethod
    def _maybe_drop_shared_cache(cls, llm: Any) -> None:
        """Evict raw decoding KV buckets when ``drop_shared_cache_on_finalize`` is set."""
        cfg = getattr(llm, "config", None)
        if cfg is None or not getattr(cfg, "drop_shared_cache_on_finalize", False):
            return
        try:
            from KVCMAS.llm.gpt_chat import LLMChat as _LLMChat
        except Exception:
            return
        shared = getattr(_LLMChat, "_shared_kv_cache_memory", None)
        if shared is None:
            return
        # The input buckets (input/input_ids/input_drop_num) are TOP-LEVEL dicts keyed by message
        # — they hold the full ~8 GB per-request input base.
        for top in cls._DROP_BUCKETS:
            b = shared.get(top)
            if isinstance(b, dict) and b:
                b.clear()
        # Per-node decoding buckets (response/condition under each node id).
        for mem in shared.values():
            if not isinstance(mem, dict):
                continue
            for bucket in cls._DROP_BUCKETS:
                existing = mem.get(bucket)
                if isinstance(existing, dict) and existing:
                    existing.clear()

    @classmethod
    def _commit_staged_states_locked(cls) -> None:
        """Commit all staged request states into the global dictionaries."""
        if not cls._staged_commits:
            return
        for state in cls._staged_commits:
            state.commit()
        cls._staged_commits.clear()

    def resolve_request_state(self, request_uid: str) -> _RequestState:
        """Public alias to access or create the request-scoped state."""
        return self._get_request_state(request_uid)

    def get_request_state(self, request_uid: str) -> _RequestState:
        """Return the request state; identical to resolve_request_state."""
        return self.resolve_request_state(request_uid)

    @staticmethod
    def anchor_signature(anchor_list: List[Dict[str, Any]]) -> Tuple[int, ...]:
        """Create a lightweight fingerprint for the active anchors."""
        return tuple(id(anchor) for anchor in anchor_list)

    def _get_cached_anchor_weights(
        self,
        request_uid: str,
        ph_id: str,
        message: str,
        signature: Tuple[int, ...],
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Look up cached anchor weights for a specific placeholder/message/signature."""
        state = self.resolve_request_state(request_uid)
        bucket = state.weight_dict.get(ph_id)
        if bucket is None:
            return None
        entry = bucket.get(message)
        if not entry:
            return None
        if entry.get("anchor_signature") != signature:
            return None
        return entry

    def _set_cached_anchor_weights(
        self,
        request_uid: str,
        ph_id: str,
        message: str,
        entry: Dict[str, torch.Tensor],
    ) -> None:
        """Store computed anchor weights for reuse within the same request."""
        state = self.resolve_request_state(request_uid)
        bucket = state.weight_dict.setdefault(ph_id, {})
        bucket[message] = entry

    @staticmethod
    def _select_anchor_indices(
        anchor_list: List[Dict[str, Any]], placeholder_len: int
    ) -> Tuple[List[int], int]:
        """``(indices, cover_len)``: the anchors used, and how many rows they cover. """
        lens = [int(a["ph_key_embedding"].shape[-2]) for a in anchor_list]
        idx = [i for i, s_a in enumerate(lens) if s_a >= placeholder_len]
        if idx or not lens or not _anchor_pad_enabled():
            return idx, int(placeholder_len)
        cover = max(lens)
        if cover <= 0:
            return [], int(placeholder_len)
        return [i for i, s_a in enumerate(lens) if s_a >= cover], cover

    @staticmethod
    def _topk_select_sims(
        sims_for_ranking: torch.Tensor,
        hot_k: Optional[int],
    ) -> Optional[torch.Tensor]:
        """Return ascending-sorted top-k anchor indices (smallest sims = closest), or None. """
        if hot_k is None:
            return None
        V = sims_for_ranking.shape[0]
        if V <= hot_k:
            return None
        if sims_for_ranking.dim() > 1:
            score = sims_for_ranking.float().mean(dim=tuple(range(1, sims_for_ranking.dim())))
        else:
            score = sims_for_ranking.float()
        _, top_pos = torch.topk(score, k=hot_k, dim=0, largest=False)  # smallest sims first
        top_sorted, _ = torch.sort(top_pos)
        return top_sorted

    @classmethod
    def _compute_anchor_weight_entry_full(
        cls,
        used_anchors: List[Dict[str, Any]],
        anchor_indices: List[int],
        real_key_embedding: torch.Tensor,
        real_value_embedding: torch.Tensor,
        placeholder_len: int,
        temperature: float,
        hot_k: Optional[int],
        renorm_hot: bool = False,
        cover_len: Optional[int] = None,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Original KVComm per-channel weighting against full-rank base embeddings. """
        # The placeholder span is the LAST ``placeholder_len`` rows of the base (the leading rows
        # are the prefix this segment trims), so the span's HEAD begins.
        _keys = list(real_key_embedding) if not isinstance(real_key_embedding, torch.Tensor) \
            else [real_key_embedding[i] for i in range(real_key_embedding.shape[0])]
        _vals = list(real_value_embedding) if not isinstance(real_value_embedding, torch.Tensor) \
            else [real_value_embedding[i] for i in range(real_value_embedding.shape[0])]
        _rows = int(_keys[0].shape[-2])
        _cover = min(int(placeholder_len if cover_len is None else cover_len), int(placeholder_len))
        _s0 = max(0, _rows - int(placeholder_len))
        _s1 = min(_rows, _s0 + max(1, _cover))
        # Every anchor-side slice below is over the covered rows only.
        placeholder_len = _s1 - _s0
        L_ = len(_keys)
        B_, H_, _, D_ = _keys[0].shape
        S_ = placeholder_len
        Vn_ = len(used_anchors)
        dev = _keys[0].device

        # Layer-wise: stack one layer's V anchors [V, B, H, S, D] at a time rather than the whole
        # [V, L, B, H, S, D], which is L times the working set.
        _hot = hot_k is not None and Vn_ > hot_k
        sims_key_prefix = torch.empty(Vn_, L_, B_, H_, D_, dtype=torch.float32, device=dev)
        sims_val_prefix = torch.empty(Vn_, L_, B_, H_, D_, dtype=torch.float32, device=dev)
        ph_sum_key = torch.zeros(Vn_, S_, dtype=torch.float32, device=dev)
        ph_sum_val = torch.zeros(Vn_, S_, dtype=torch.float32, device=dev)
        # Per-layer placeholder rank score [V, L] (mean|·| over B,H,S,D, KEEP L) for
        # per-(segment,layer) selective anchoring; only needed when hot.
        ph_score_key = torch.zeros(Vn_, L_, dtype=torch.float32, device=dev) if _hot else None
        _cw = False
        if _cw:
            _path_note("weights COSINE-PROPORTIONAL (w ∝ relu(cos))")
            # Cosine accumulators.
            ph_dot_key = torch.zeros(Vn_, S_, dtype=torch.float32, device=dev)
            ph_dot_val = torch.zeros(Vn_, S_, dtype=torch.float32, device=dev)
            ph_na_key = torch.zeros(Vn_, S_, dtype=torch.float32, device=dev)
            ph_na_val = torch.zeros(Vn_, S_, dtype=torch.float32, device=dev)
            nc_row_key = torch.zeros(S_, dtype=torch.float32, device=dev)
            nc_row_val = torch.zeros(S_, dtype=torch.float32, device=dev)
        _cos_k = cos.unsqueeze(1) if cos is not None else None   # [B,1,1,D]: broadcast over H, S
        _sin_k = sin.unsqueeze(1) if sin is not None else None
        for l in range(L_):
            # Slice THIS layer only.
            rk_l = _keys[l][..., _s0:_s1, :]      # [B, H, S, D]
            if _cos_k is not None:
                rk_l = rk_l * _cos_k + _rotate_half(rk_l) * _sin_k   # candidate key -> anchor frame
            rv_l = _vals[l][..., _s0:_s1, :]
            if _cw:
                nc_row_key += rk_l.pow(2).sum(dim=(0, 1, 3), dtype=torch.float32)
                nc_row_val += rv_l.pow(2).sum(dim=(0, 1, 3), dtype=torch.float32)
                _ncp_key = rk_l.pow(2).sum(dim=-2, dtype=torch.float32)   # [B,H,D] this layer
                _ncp_val = rv_l.pow(2).sum(dim=-2, dtype=torch.float32)
            # Stream over anchors too.
            _chunk = _sims_chunk_fit(_sims_batch_size(), used_anchors,
                                     "ph_key_embedding", placeholder_len, rk_l)
            _vi = 0
            while _vi < Vn_:
                _cs = used_anchors[_vi:_vi + _chunk]
                _k = len(_cs)
                _bk = _bv = None
                if _chunk > 1:
                    _bk = _expand_anchor_chunk(_cs, "ph_key_embedding", l, placeholder_len)
                    _bv = (_expand_anchor_chunk(_cs, "ph_value_embedding", l, placeholder_len)
                           if _bk is not None else None)
                if _bk is not None and _bv is not None:
                    _path_note(f"sims ANCHOR-BATCHED (k={_k})")
                    AK, AK_t = _bk
                    AV, AV_t = _bv
                    _rk = rk_l.unsqueeze(0)                                   # [1,B,H,S,D]
                    _rv = rv_l.unsqueeze(0)
                    # Tail FIRST: when the anchors all store exactly placeholder_len rows the tail
                    # block IS the head block.
                    if _cw:
                        # (1 - cos): distance semantics preserved for hot-k/profiling.
                        _dp = (AK_t * _rk).sum(dim=-2, dtype=torch.float32)
                        _np = AK_t.pow(2).sum(dim=-2, dtype=torch.float32)
                        sims_key_prefix[_vi:_vi + _k, l] = 1.0 - _dp / ((_np * _ncp_key.unsqueeze(0)).sqrt() + 1e-12)
                        _dp = (AV_t * _rv).sum(dim=-2, dtype=torch.float32)
                        _np = AV_t.pow(2).sum(dim=-2, dtype=torch.float32)
                        sims_val_prefix[_vi:_vi + _k, l] = 1.0 - _dp / ((_np * _ncp_val.unsqueeze(0)).sqrt() + 1e-12)
                        del _dp, _np
                        ph_dot_key[_vi:_vi + _k] += (AK * _rk).sum(dim=(1, 2, 4), dtype=torch.float32)
                        ph_na_key[_vi:_vi + _k] += AK.pow(2).sum(dim=(1, 2, 4), dtype=torch.float32)
                        ph_dot_val[_vi:_vi + _k] += (AV * _rv).sum(dim=(1, 2, 4), dtype=torch.float32)
                        ph_na_val[_vi:_vi + _k] += AV.pow(2).sum(dim=(1, 2, 4), dtype=torch.float32)
                    else:
                        sims_key_prefix[_vi:_vi + _k, l] = (AK_t - _rk).norm(2, dim=-2).float()
                        sims_val_prefix[_vi:_vi + _k, l] = (AV_t - _rv).norm(2, dim=-2).float()
                    del AK_t, AV_t
                    # sub_ + abs_ in place: the expanded block is dead after this, so the diff
                    # reuses its memory instead of doubling.
                    DK = AK.sub_(_rk).abs_()                                  # [k,B,H,S,D]
                    DV = AV.sub_(_rv).abs_()
                    ph_sum_key[_vi:_vi + _k] += DK.sum(dim=(1, 2, 4), dtype=torch.float32)
                    if _hot:
                        ph_score_key[_vi:_vi + _k, l] = DK.mean(dim=(1, 2, 3, 4))
                    ph_sum_val[_vi:_vi + _k] += DV.sum(dim=(1, 2, 4), dtype=torch.float32)
                    del AK, AV, DK, DV, _rk, _rv
                else:
                    # Per-anchor fallback: raw (un-factorized) base, a span shorter than
                    # the placeholder, or batching disabled. Exact, just slower.
                    _path_note("sims per-anchor FALLBACK (base not low-rank, span < placeholder, or batch=1)")
                    for _j, a in enumerate(_cs):
                        ak = a["ph_key_embedding"][l]      # [B, H, S_a, D]
                        av = a["ph_value_embedding"][l]
                        if _cw:
                            _akt = ak[..., -placeholder_len:, :]
                            _avt = av[..., -placeholder_len:, :]
                            sims_key_prefix[_vi + _j, l] = 1.0 - (_akt * rk_l).sum(dim=-2, dtype=torch.float32) / ((_akt.pow(2).sum(dim=-2, dtype=torch.float32) * _ncp_key).sqrt() + 1e-12)
                            sims_val_prefix[_vi + _j, l] = 1.0 - (_avt * rv_l).sum(dim=-2, dtype=torch.float32) / ((_avt.pow(2).sum(dim=-2, dtype=torch.float32) * _ncp_val).sqrt() + 1e-12)
                            _akh = ak[..., :placeholder_len, :]
                            _avh = av[..., :placeholder_len, :]
                            ph_dot_key[_vi + _j] += (_akh * rk_l).sum(dim=(0, 1, 3), dtype=torch.float32)
                            ph_na_key[_vi + _j] += _akh.pow(2).sum(dim=(0, 1, 3), dtype=torch.float32)
                            ph_dot_val[_vi + _j] += (_avh * rv_l).sum(dim=(0, 1, 3), dtype=torch.float32)
                            ph_na_val[_vi + _j] += _avh.pow(2).sum(dim=(0, 1, 3), dtype=torch.float32)
                            del _akt, _avt, _akh, _avh
                        else:
                            sims_key_prefix[_vi + _j, l] = (rk_l - ak[..., -placeholder_len:, :]).norm(2, dim=-2).float()   # [B,H,D]
                            sims_val_prefix[_vi + _j, l] = (rv_l - av[..., -placeholder_len:, :]).norm(2, dim=-2).float()
                        _dk = (rk_l - ak[..., :placeholder_len, :]).abs()                 # [B, H, S, D]
                        ph_sum_key[_vi + _j] += _dk.sum(dim=(0, 1, 3), dtype=torch.float32)     # sum B,H,D -> [S]
                        if _hot:
                            ph_score_key[_vi + _j, l] = _dk.mean()                             # mean B,H,S,D -> scalar
                        ph_sum_val[_vi + _j] += (rv_l - av[..., :placeholder_len, :]).abs().sum(dim=(0, 1, 3), dtype=torch.float32)
                        del ak, av, _dk
                _vi += _chunk
            del rk_l, rv_l
        denom = float(L_ * B_ * H_ * D_)
        if _cw:
            _ck = ph_dot_key / ((ph_na_key.sqrt() * nc_row_key.sqrt().unsqueeze(0)) + 1e-12)
            _cv = ph_dot_val / ((ph_na_val.sqrt() * nc_row_val.sqrt().unsqueeze(0)) + 1e-12)
            sims_key_placeholder = (1.0 - _ck).view(Vn_, 1, 1, 1, S_, 1)
            sims_val_placeholder = (1.0 - _cv).view(Vn_, 1, 1, 1, S_, 1)
        else:
            sims_key_placeholder = (ph_sum_key / denom).view(Vn_, 1, 1, 1, S_, 1)
            sims_val_placeholder = (ph_sum_val / denom).view(Vn_, 1, 1, 1, S_, 1)

        # Softmax over the FULL V — preserves the true expected-correction magnitude.
        _nw = False
        _sc = _zscore_anchors if _nw else (lambda t: t)
        if _nw:
            _path_note("weights Z-SCORED over anchor axis")
        weights_key_prefix = torch.softmax(-_sc(sims_key_prefix.float()) / temperature, dim=0).unsqueeze(-2)
        weights_value_prefix = torch.softmax(-_sc(sims_val_prefix.float()) / temperature, dim=0).unsqueeze(-2)
        weights_key_placeholder = torch.softmax(-_sc(sims_key_placeholder.float()) / temperature, dim=0)
        weights_value_placeholder = torch.softmax(-_sc(sims_val_placeholder.float()) / temperature, dim=0)
        if _cw:
            # w ∝ relu(cos) = relu(1 - sims), normalized over the anchor axis -- the probe's
            # cos_prop row exactly.
            def _wprop(s):
                w = (1.0 - s.float()).clamp_min(0.0)
                return w / w.sum(0, keepdim=True).clamp_min(1e-12)
            weights_key_prefix = _wprop(sims_key_prefix).unsqueeze(-2)
            weights_value_prefix = _wprop(sims_val_prefix).unsqueeze(-2)
            weights_key_placeholder = _wprop(sims_key_placeholder)
            weights_value_placeholder = _wprop(sims_val_placeholder)
        # Selective anchoring (kvcmas-sa): keep top-k anchors PER (segment, layer), NO
        # renormalization.
        if _hot:
            k = int(hot_k)
            pf_rank = sims_key_prefix.mean(dim=(2, 3, 4))   # [V, L]  (fp32)
            ph_rank = ph_score_key                          # [V, L]
            anchor_index_prefix: List[List[int]] = []       # [L][k] anchor identities
            anchor_index_placeholder: List[List[int]] = []
            wkp, wvp, wkph, wvph = [], [], [], []
            for l in range(L_):
                kp = torch.sort(torch.topk(pf_rank[:, l], k=k, largest=False).indices).values    # [k] smallest-dist
                kph = torch.sort(torch.topk(ph_rank[:, l], k=k, largest=False).indices).values
                anchor_index_prefix.append([anchor_indices[i] for i in kp.tolist()])
                anchor_index_placeholder.append([anchor_indices[i] for i in kph.tolist()])
                wkp.append(weights_key_prefix[kp, l])         # [k, B, H, 1, D]
                wvp.append(weights_value_prefix[kp, l])
                wkph.append(weights_key_placeholder[kph, 0])  # [k, 1, 1, S, 1] (ph weight is L-uniform)
                wvph.append(weights_value_placeholder[kph, 0])
            weights_key_prefix = torch.stack(wkp, dim=1)        # [k, L, B, H, 1, D]
            weights_value_prefix = torch.stack(wvp, dim=1)
            weights_key_placeholder = torch.stack(wkph, dim=1)  # [k, L, 1, 1, S, 1]
            weights_value_placeholder = torch.stack(wvph, dim=1)
            if renorm_hot:
                # Restore the correction magnitude the truncation dropped: rescale each channel's
                # kept weights to sum 1 over the anchor axis (dim 0).
                weights_key_prefix = weights_key_prefix / weights_key_prefix.sum(0, keepdim=True).clamp_min(1e-9)
                weights_value_prefix = weights_value_prefix / weights_value_prefix.sum(0, keepdim=True).clamp_min(1e-9)
                weights_key_placeholder = weights_key_placeholder / weights_key_placeholder.sum(0, keepdim=True).clamp_min(1e-9)
                weights_value_placeholder = weights_value_placeholder / weights_value_placeholder.sum(0, keepdim=True).clamp_min(1e-9)
        else:
            anchor_index_prefix = anchor_indices            # flat [V]; same for all layers
            anchor_index_placeholder = anchor_indices

        return {
            "anchor_index_prefix": anchor_index_prefix,
            "anchor_index_placeholder": anchor_index_placeholder,
            "weights_key_for_prefix": weights_key_prefix.detach(),
            "weights_value_for_prefix": weights_value_prefix.detach(),
            "weights_key_for_placeholder": weights_key_placeholder.detach(),
            "weights_value_for_placeholder": weights_value_placeholder.detach(),
        }

    def _anchor_weight_entry_streaming(
        self,
        anchor_list: List[Dict[str, Any]],
        anchor_indices: List[int],
        base_placeholder_cache: DynamicCache,
        placeholder_len: int,
        temperature: float,
        drop: int = 0,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        cover_len: Optional[int] = None,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Build the anchor-weight entry without ever stacking the full base. """
        if not anchor_indices:
            return None
        used_anchors = [anchor_list[idx] for idx in anchor_indices]
        hot_k = getattr(self.llm.config, "hot_anchor_num", None)
        if hot_k is not None and hot_k <= 0:
            hot_k = None
        # ANCHOR MATCHING phase: score anchors by L2 distance to the candidate base (the softmax
        # weights).
        renorm_hot = bool(getattr(self.llm.config, "renormalize_hot_weights", False))
        return self._compute_anchor_weight_entry_full(
        used_anchors, anchor_indices,
        list(base_placeholder_cache.key_cache), list(base_placeholder_cache.value_cache),
        placeholder_len, temperature, hot_k,
        renorm_hot=renorm_hot, cover_len=cover_len,
        # reader frame: rotate the raw candidate to this reader's offset to meet a base
        # stored at the writer's. canonical/legacy: compare the raw 0-base candidate.
        cos=(cos if _match_frame() == "reader" else None),
        sin=(sin if _match_frame() == "reader" else None),
        )

    def _resolve_anchor_weights(
        self,
        request_uid: str,
        ph_id: str,
        message: str,
        base_placeholder_cache: DynamicCache,
        anchor_list: List[Dict],
        temperature: float = 1.0,
        placeholder_len: Optional[int] = None,
        drop: int = 0,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get-or-compute the cached anchor-weight entry for a placeholder. """
        if placeholder_len is None:
            placeholder_len = int(base_placeholder_cache._seen_tokens)
        if placeholder_len <= 0:
            return None
        anchor_signature = self.anchor_signature(anchor_list)
        cache_entry = self._get_cached_anchor_weights(
            request_uid, ph_id, message=message, signature=anchor_signature,
        )
        if cache_entry is None:
            anchor_index, cover_len = self._select_anchor_indices(anchor_list, placeholder_len)
            if not anchor_index:
                if anchor_list:
                    self._log_warning(
                        f"No anchors cover placeholder {ph_id} for Agent {self.llm.node_id} ({self.llm.role})."
                    )
                return None
            if cover_len < placeholder_len:
                self._log_warning(
                    f"Partial anchor coverage on {ph_id} for Agent {self.llm.node_id} "
                    f"({self.llm.role}): {cover_len}/{placeholder_len} rows corrected by "
                    f"{len(anchor_index)}/{len(anchor_list)} anchors, tail reused as-is."
                )
            cache_entry = self._anchor_weight_entry_streaming(
                anchor_list, anchor_index, base_placeholder_cache, placeholder_len,
                float(temperature), drop=drop, cos=cos, sin=sin, cover_len=cover_len,
            )
            if cache_entry is None:
                return None
            cache_entry["anchor_signature"] = anchor_signature
            cache_entry["placeholder_len"] = placeholder_len
            cache_entry["cover_len"] = int(cover_len)
            self._set_cached_anchor_weights(request_uid, ph_id, message, cache_entry)
        return cache_entry

    def predict_as_anchor(
        self,
        candidate_kv_cache: DynamicCache,
        anchor_kv_cache_list: List[Dict],
        anchor_len_list: List[Tuple[int, int]],
        anchor_activated_list: List[int],
        top_p: float = 0.9,
        entropy_eps: float = 1e-40,
        test_time: bool = False,
    ) -> Tuple[bool, List[int]]:
        if len(anchor_kv_cache_list) in [0, 1]:
            return True, anchor_activated_list

        if test_time:
            torch.cuda.synchronize()
            start_time = perf_counter()
        k = candidate_kv_cache.value_cache[0].shape[-2]
        # An anchor can only serve a candidate at least as long as itself: `j >= k`.
        anchor_available = [i for i, (j, _accum_j) in enumerate(anchor_len_list) if j >= k]

        if len(anchor_len_list) != len(anchor_kv_cache_list):
            self._log_warning(
                "The length of anchor_len_list is not equal to the length of anchor_available, "
                f"with {len(anchor_len_list)} and {len(anchor_available)}."
            )
            return True, anchor_activated_list

        if len(anchor_available) > 1:
            # Layer-wise global value-L2 (gate).
            vlist = candidate_kv_cache.value_cache
            _cosg = False
            acc = torch.zeros(len(anchor_available), dtype=torch.float32, device=vlist[0].device)
            dot = na = nc = None
            if _cosg:
                dot = torch.zeros_like(acc)   # Σ_l <cand_l, anchor_l>
                na = torch.zeros_like(acc)    # Σ_l ||anchor_l||²
                nc = torch.zeros((), dtype=torch.float32, device=vlist[0].device)  # Σ_l ||cand_l||²
            for l in range(len(vlist)):
                cand_l = vlist[l][..., :k, :]  # [B, H, k, D]
                if _cosg:
                    nc += cand_l.pow(2).sum(dtype=torch.float32)
                # Stream over anchors too (not just layers).
                for vi, i in enumerate(anchor_available):
                    anc_l = anchor_kv_cache_list[i]["ph_value_embedding"][l][..., :k, :]  # [B, H, k, D]
                    if _cosg:
                        dot[vi] += (cand_l * anc_l).sum(dtype=torch.float32)
                        na[vi] += anc_l.pow(2).sum(dtype=torch.float32)
                    else:
                        acc[vi] += (cand_l - anc_l).pow(2).sum(dtype=torch.float32)
                    del anc_l
                del cand_l
            if _cosg:
                _path_note("gate COSINE (1 - cos over all-layer value cache)")
                # Global cosine over the same flattened [L,B,H,k,D] the L2 used; the
                # softmax/entropy/threshold/top-p machinery below is shared unchanged.
                diff = 1.0 - dot / (na.sqrt() * nc.sqrt() + 1e-12)  # [V], in [0, 2]
            else:
                diff = acc.sqrt()  # [V] — same global L2 as the stacked norm
            sim = torch.softmax(-diff, dim=0)
            threshold = self.llm.config.threshold
            entropy = -(sim * (sim + entropy_eps).log2()).sum()
            if entropy > threshold * torch.log2(torch.tensor(sim.shape[0])):
                logger.opt(colors=True).debug(
                    f"<yellow>Entropy {entropy:.4f} exceeds threshold {threshold * torch.log2(torch.tensor(sim.shape[0])):.4f}, "
                    "skip activating anchors.</yellow>"
                )
                if test_time:
                    torch.cuda.synchronize()
                    end_time = perf_counter()
                    logger.opt(colors=True).debug(
                        f"<cyan>Latency for Anchor prediction: {end_time - start_time} s</cyan>"
                    )
                return True, anchor_activated_list
            sorted_sim, sorted_indices = torch.sort(sim, descending=True)
            cumulative_sum = torch.cumsum(sorted_sim, dim=0)
            cutoff_index_candidates = (cumulative_sum < top_p).nonzero(as_tuple=True)[0]
            cutoff_index = cutoff_index_candidates[-1] if len(cutoff_index_candidates) > 0 else len(sorted_sim) - 1
            selected_indices = sorted_indices[:cutoff_index + 1]
            for i in selected_indices:
                if anchor_available[i] >= len(anchor_activated_list):
                    self._log_warning(
                        "anchor_available index "
                        f"{anchor_available[i]} out of range for anchor_activated_list with length "
                        f"{len(anchor_activated_list)}"
                    )
                    continue
                anchor_activated_list[anchor_available[i]] += 1
            if test_time:
                torch.cuda.synchronize()
                end_time = perf_counter()
                logger.opt(colors=True).debug(
                    f"<cyan>Latency for Anchor prediction: {end_time - start_time} s</cyan>"
                )
            return False, anchor_activated_list
        logger.opt(colors=True).debug("<yellow>No available anchors to activate.</yellow>")
        return True, anchor_activated_list

    def update_anchor(self, request_uid: str, ph_id: str, window_length: int = 5) -> None:
        """Update the anchor list by filtering out the least frequent anchors in the oldest anchor set."""
        state = self.resolve_request_state(request_uid)
        anchor_store = state.anchors.setdefault(ph_id, {})
        anchor_info_dict = state.anchor_info_dict.setdefault(ph_id, {})
        info_list = list(anchor_info_dict.values())[:window_length]
        if not info_list:
            return
        min_idx = info_list.index(min(info_list))
        message = list(anchor_info_dict.keys())[min_idx]
        anchor_store.pop(message, None)
        state.anchor_len_dict.setdefault(ph_id, {}).pop(message, None)
        freq = anchor_info_dict.pop(message, None)
        state.global_anchor_info.setdefault(ph_id, {}).pop(message, None)
        self._log_warning(
            f"Removed anchor for message '{message}' in {self.llm.node_id} ({self.llm.role}) due to low frequency: {freq}"
        )

    def set_anchor_streaming_raw(
        self,
        request_uid: str,
        message: str,
        ph_id_list: List[str],
        real_full: DynamicCache,
        meta: List[Dict[str, Any]],
        placeholder_indices: Dict[str, Tuple[int, int]],
        max_anchor_num: int = 20,
        window_length: int = 5,
    ) -> Dict[str, List[List[Dict]]]:
        """Stage-2 single-use ``set_anchor``: stream the per-placeholder delta from the RAW base segments (``m['ph_cache']`` / ``m['pf_kv']``) with per-segment."""
        state = self.resolve_request_state(request_uid)
        anchor_store = state.anchors
        anchor_flags = {ph_id: state.anchor_dict.setdefault(ph_id, {}) for ph_id in ph_id_list}
        # Decoupled per-segment SVD ranks (kvcmas-sa): pf eff-rank << ph (profiling), so the
        # prefix delta can be compressed harder than the placeholder delta.
        rank_ph = int(getattr(self.llm.config, "rank_ph", 0) or 0)
        rank_pf = int(getattr(self.llm.config, "rank_pf", 0) or 0)
        # The prefix span carries NO delta by default: it is reused rotate-only.
        _pf_delta = _pf_delta_enabled()
        # Base-embedding SVD ranks (prefill-only matching; 0 = store full base).
        # value kept >= key by default since base_value feeds the entropy gate.
        rank_base_key = int(getattr(self.llm.config, "rank_base_key", 0) or 0)
        rank_base_value = int(getattr(self.llm.config, "rank_base_value", 0) or 0)
        node_id = self.llm.node_id
        L_ = len(real_full.key_cache)
        _, H_, _, D_ = real_full.key_cache[0].shape
        dev = real_full.key_cache[0].device

        def _make_anchor_raw(idx, m):
            ph_id = m["ph_id"]
            s, e = placeholder_indices[ph_id]
            ph_cache = m["ph_cache"]
            ph_drop = m["drop_num"]
            ph_offset = m["start"] - m["drop_num"] + m["offset_before"]
            pf_kv = m["pf_kv"]
            pf_len = pf_kv.key_cache[0].shape[-2]
            pf_offset = m["offset_after"]
            # Sample the real cache (on GPU) for RoPE dtype/device, so cos/sin land on-device even
            # though the base segments stay parked on CPU (pulled one layer.
            rope_sample = real_full.key_cache[0]
            ph_cos, ph_sin = self._segment_rope(e - s, ph_offset, rope_sample)
            pf_cos, pf_sin = self._segment_rope(pf_len, pf_offset, rope_sample)
            if _ph_delta_enabled(ph_id):
                ph_kd, ph_vd = _svd_compress_streaming(
                    real_full, s, e, ph_cache, ph_drop, None, ph_cos, ph_sin, rank_ph)
            else:
                # Same treatment the prefix gets under KVCMAS_PF_DELTA=0: skipped, not
                # built-then-ignored, so the per-anchor SVD and the pool memory go too.
                ph_kd = ph_vd = None
            if _pf_delta:
                pf_kd, pf_vd = _svd_compress_streaming(
                    real_full, e, e + pf_len, pf_kv, 0, None, pf_cos, pf_sin, rank_pf)
            else:
                # Skipped entirely, not built-then-ignored: this is the per-anchor SVD of a full-
                # length span, so skipping it saves the work AND the pool memory.
                pf_kd = pf_vd = None
            cos2, sin2 = ph_cos.unsqueeze(1), ph_sin.unsqueeze(1)
            dev2 = cos2.device  # base parked on CPU; rotate on the RoPE (GPU) device
            if rank_base_key > 0 or rank_base_value > 0:
                # STREAMING base compression: build + SVD one layer at a time, never stacking the
                # full [L,B,H,S,D] base.
                ksU, ksV, vsU, vsV, ks, vs = [], [], [], [], [], []
                base_shape = None
                kbb = _LayerSVDBatcher(rank_base_key) if rank_base_key > 0 else None
                vbb = _LayerSVDBatcher(rank_base_value) if rank_base_value > 0 else None
                _canon = _match_frame() == "canonical"
                with _SVD_LOCK:
                    for l in range(L_):
                        bk = ph_cache.key_cache[l][..., ph_drop:, :].to(dev2)
                        # Matching frame for the stored base KEY: canonical = the bucket's
                        # 0-base frame (no rotation); otherwise the writer's offset.
                        bk_rot = bk if _canon else (bk * cos2 + _rotate_half(bk) * sin2)
                        bv = ph_cache.value_cache[l][..., ph_drop:, :].to(dev2)
                        if base_shape is None:
                            base_shape = (L_,) + tuple(bk_rot.shape)  # (L,B,H,S,D)
                        if rank_base_key > 0:
                            kbb.add(bk_rot)
                        else:
                            ks.append(bk_rot)
                        if rank_base_value > 0:
                            vbb.add(bv)
                        else:
                            vs.append(bv)
                        del bk, bk_rot, bv
                    if kbb is not None:
                        ksU, ksV = kbb.finish()
                    if vbb is not None:
                        vsU, vsV = vbb.finish()
                key_base_stored = _stack_base_factors(ksU, ksV, ks, rank_base_key, base_shape)
                val_base_stored = _stack_base_factors(vsU, vsV, vs, rank_base_value, base_shape)
            else:
                # base compression off: original full-base path (unchanged).
                ks, vs = [], []
                for l in range(L_):
                    bk = ph_cache.key_cache[l][..., ph_drop:, :].to(dev2)
                    ks.append(bk if _match_frame() == "canonical" else (bk * cos2 + _rotate_half(bk) * sin2))
                    vs.append(ph_cache.value_cache[l][..., ph_drop:, :].to(dev2))
                key_base_stored, val_base_stored = torch.stack(ks), torch.stack(vs)
            entry = {
                "ph_key_embedding": key_base_stored,
                "ph_value_embedding": val_base_stored,
                "placeholder_len": int(e - s),
                f"{node_id}_ph_key_delta": ph_kd,
                f"{node_id}_ph_value_delta": ph_vd,
                f"{node_id}_pf_key_delta": pf_kd,
                f"{node_id}_pf_value_delta": pf_vd,
            }
            return idx, entry

        # No-encode mode: the only writer of the user_question anchor FLAG is the standalone-
        # encode path (gpt_chat.update_input_anchor), which no-encode never.
        def _admit_q(ph_id: str) -> bool:
            if "user_question" not in ph_id:
                return False
            if float(getattr(self.llm.config, "threshold", 1.0) or 0.0) <= 0.0:
                return True
            return len(anchor_store.get(ph_id, {})) < max_anchor_num
        args = [(idx, m) for idx, m in enumerate(meta)
                if anchor_flags[m["ph_id"]].get(message) is True
                or _admit_q(m["ph_id"])]
        if not args:
            return anchor_store
        # ANCHOR CREATION phase.
        results = list(self.llm._map_in_pool(_make_anchor_raw, args, timeout=30))
        results.sort(key=lambda x: x[0])
        anchor_dict = {i: entry for i, entry in results}

        accumulate_len = 0
        for idx, m in enumerate(meta):
            ph_id = m["ph_id"]
            s, e = placeholder_indices[ph_id]
            placeholder_len = e - s
            if idx not in anchor_dict:
                accumulate_len += placeholder_len
                continue
            entry = anchor_dict[idx]
            if len(anchor_store.setdefault(ph_id, {})) > max_anchor_num:
                self.update_anchor(request_uid, ph_id, window_length)
            if message not in anchor_store[ph_id]:
                anchor_store[ph_id][message] = entry
                state.anchor_info_dict.setdefault(ph_id, {})[message] = 0
                state.anchor_len_dict.setdefault(ph_id, {})[message] = [placeholder_len, accumulate_len]
                state.global_anchor_info.setdefault(ph_id, {}).setdefault(message, [0, placeholder_len])
            else:
                # Overwrite path: refresh recorded length too, else predict_as_anchor
                # admits the anchor by a stale length and mis-slices the shorter tensor.
                anchor_store[ph_id][message].update(entry)
                state.anchor_len_dict.setdefault(ph_id, {})[message] = [placeholder_len, accumulate_len]
                gbucket = state.global_anchor_info.setdefault(ph_id, {})
                if message in gbucket:
                    gbucket[message][1] = placeholder_len
                else:
                    gbucket[message] = [0, placeholder_len]
            accumulate_len += placeholder_len
        return anchor_store

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def rotate_tensor(
        self,
        tensor: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        unsqueeze_dim: int = 1,
    ) -> torch.Tensor:
        """Apply RoPE rotation using provided cos/sin tables."""
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        return (tensor * cos) + (self._rotate_half(tensor) * sin)

    def apply_rotary_pos_emb(
        self,
        ph_cache: DynamicCache,
        offset: int,
        drop_num: int = 0,
        consume: bool = False,
        owner: Optional[DynamicCache] = None,
    ) -> DynamicCache:
        """Rotate placeholder cache keys by absolute offset (with optional drop). """
        rotate_emb = self.llm.model.model.rotary_emb
        src_keys = ph_cache.key_cache
        src_values = ph_cache.value_cache
        cur = src_keys[0].shape[-2]
        s = min(drop_num, cur) if drop_num and drop_num > 0 else 0
        seq = cur - s
        position_ids = (
            torch.ones(seq, dtype=torch.long).unsqueeze(0).to(self.llm.model.device) * offset
        )
        # cos/sin from the sliced sample key (matches the original, which sliced first)
        cos, sin = rotate_emb(src_keys[0][..., s:, :], position_ids)

        new_ph_cache = type(ph_cache)()
        new_ph_cache.key_cache = []
        new_ph_cache.value_cache = []
        for _li in range(len(src_keys)):
            key, value = src_keys[_li], src_values[_li]
            new_ph_cache.key_cache.append(
                self.rotate_tensor(key[..., s:, :], cos, sin).contiguous()
            )
            new_ph_cache.value_cache.append(value[..., s:, :].clone())
            if consume:
                src_keys[_li] = None
                src_values[_li] = None
                if owner is not None:
                    # ``ph_cache`` is a VIEW over ``owner`` (the hop's corrected cache), so
                    # nulling the view frees nothing -- the owner holds the storage.
                    for _lst in ("key_cache", "value_cache", "_buf_k", "_buf_v"):
                        _seq = getattr(owner, _lst, None)
                        if _seq is not None and _li < len(_seq):
                            _seq[_li] = None
        for attr in ("offloading", "only_non_sliding", "prefetch_stream", "layer_class_to_replicate"):
            if hasattr(ph_cache, attr):
                setattr(new_ph_cache, attr, getattr(ph_cache, attr))
        _set_seen_tokens(new_ph_cache, seq)
        return new_ph_cache

    def fetch_shared_cache(
        self,
        ph_id: str,
        message: str,
    ) -> Tuple[DynamicCache, Dict[str, torch.Tensor], int]:
        """Retrieve shared KV cache and ids for a placeholder given message context."""
        shared_memory = self.llm._shared_kv_cache_memory

        if "user_question" in ph_id:
            return (
                shared_memory["input"][message][-1],
                shared_memory["input_ids"][message][-1],
                shared_memory["input_drop_num"][message][-1],
            )

        type_str, node_id, *rest = ph_id.split("_")
        is_current = (rest and rest[0] == "current")

        key_prefix = "condition" if type_str == "condition" else "response"
        slot_idx = -1 if is_current else -2

        node_memory = shared_memory[node_id]

        def _get_slot(bucket_key: str):
            bucket = node_memory.get(bucket_key, {})
            values = bucket.get(message)
            if not values:
                return None
            try:
                return values[slot_idx]
            except IndexError:
                return None

        ph_cache = _get_slot(key_prefix)
        ph_cache_ids = _get_slot(f"{key_prefix}_ids")
        drop_num = _get_slot(f"{key_prefix}_drop_num")

        if ph_cache is None:
            raise RuntimeError(
                f"fetch_shared_cache: placeholder {ph_id} for message='{message}' not found."
            )

        return ph_cache, ph_cache_ids, drop_num

    @staticmethod
    def trim_token_ids(ids_dict: Dict[str, torch.Tensor], drop_num: int) -> Dict[str, torch.Tensor]:
        if drop_num == 0:
            return ids_dict
        return {
            key: None if value is None else value[:, drop_num:]
            for key, value in ids_dict.items()
        }

    def _span_consumable(self, ph_id: Any, message: str) -> bool:
        """Is this hop the LAST reader of ``ph_id``'s bucket for ``message``?"""
        ph_id = str(ph_id or "")
        if "user_question" in ph_id:
            return True              # replaced per hop under chaining: dead after use
        if not _consume_responses_enabled():
            return False
        if not hasattr(self, "_ph_reads"):
            self._ph_reads = {}
        seen = self._ph_reads.get((message, ph_id), 0) + 1
        self._ph_reads[(message, ph_id)] = seen
        # Conservative by construction: a dense hop reads a bucket without folding it, so it never
        # increments here and the count can only UNDER-shoot.
        return seen >= _ph_reader_count(type(self.llm), ph_id)

    def update_kv_cache_segment(
        self,
        request_uid: str,
        message: str,
        m: Dict[str, Any],
        anchors_for_ph: List[Dict],
    ) -> Tuple[int, List[Dict[str, Any]], Dict[str, torch.Tensor]]:
        """Emit ph + pf segment DESCRIPTORS for a kv_reuse placeholder — nothing is rotated, copied, or corrected here. """
        ph_cache, pf_kv = m["ph_cache"], m["pf_kv"]
        # Defensive device normalization: a dense hop parks bucket caches on CPU and the restore
        # paths are mode/chaining-specific.
        _dev = self.llm.model.device
        if ph_cache.key_cache and ph_cache.key_cache[0].device != _dev:
            _to_device(ph_cache, _dev)
        if pf_kv.key_cache and pf_kv.key_cache[0].device != _dev:
            _to_device(pf_kv, _dev)
        drop = int(m["drop_num"])
        ph_len = int(ph_cache.key_cache[0].shape[-2]) - drop
        pf_len = int(pf_kv.key_cache[0].shape[-2])
        ph_offset = m["start"] - m["drop_num"] + m["offset_before"]
        pf_offset = m["offset_after"]
        node_id = self.llm.node_id

        # CONSUME INVARIANT: only anchors that carry THIS node's delta are usable.
        anchors_for_ph = [a for a in anchors_for_ph if f"{node_id}_ph_key_delta" in a]

        # Per-segment RoPE: ones(seq)*offset gives the same R(offset) at every
        # position, so store one [B,1,D] cos/sin and broadcast.
        ph_cos, ph_sin = self._segment_rope(1, ph_offset, ph_cache.key_cache[0])
        pf_cos, pf_sin = self._segment_rope(1, pf_offset, pf_kv.key_cache[0])

        # Weights from the RAW ph base, rotated + channel-selected per layer.
        cache_entry = self._resolve_anchor_weights(
            request_uid, m["ph_id"], message, ph_cache, anchors_for_ph, 1.0,
            placeholder_len=ph_len, drop=drop, cos=ph_cos, sin=ph_sin,
        )

        ph_desc = dict(
            key_src=ph_cache.key_cache, value_src=ph_cache.value_cache,
            # Consumable = this hop is the LAST reader of that bucket, so its layers are dead once
            # folded into the corrected cache.
            consumable=self._span_consumable(m.get("ph_id", ""), message),
            drop=drop, out_len=ph_len, cos=ph_cos, sin=ph_sin,
            key_field=None, value_field=None, anchors=None, anchor_index=None, w_key=None, w_value=None,
            cover_len=None,
        )
        pf_desc = dict(
            key_src=pf_kv.key_cache, value_src=pf_kv.value_cache,
            drop=0, out_len=pf_len, cos=pf_cos, sin=pf_sin,
            key_field=None, value_field=None, anchors=None, anchor_index=None, w_key=None, w_value=None,
            cover_len=None,
        )
        if cache_entry is not None and cache_entry.get("anchor_index_prefix"):
            # Per-segment anchor sets (selective anchoring): ph and pf each carry their
            # own selection (a flat [V] list, or a per-layer [L][k] list-of-lists).
            if _ph_delta_enabled(m.get("ph_id", "")):
                ph_desc.update(
                    key_field=f"{node_id}_ph_key_delta", value_field=f"{node_id}_ph_value_delta",
                    anchors=anchors_for_ph, anchor_index=cache_entry["anchor_index_placeholder"],
                    w_key=cache_entry["weights_key_for_placeholder"],
                    w_value=cache_entry["weights_value_for_placeholder"],
                    cover_len=cache_entry.get("cover_len"),
                )
            if _pf_delta_enabled():
                pf_desc.update(
                    key_field=f"{node_id}_pf_key_delta", value_field=f"{node_id}_pf_value_delta",
                    anchors=anchors_for_ph, anchor_index=cache_entry["anchor_index_prefix"],
                    w_key=cache_entry["weights_key_for_prefix"],
                    w_value=cache_entry["weights_value_for_prefix"],
                )
            # else: pf_desc stays rotate-only (its delta fields were never built).
        seg_token_ids = concat(self.trim_token_ids(m["ph_cache_ids"], drop), m["pf_ids"])
        return m["idx"], [ph_desc, pf_desc], seg_token_ids

    def process_anchor(
        self,
        message: str,
        m: Dict[str, Any],
    ) -> Tuple[int, DynamicCache, Dict[str, torch.Tensor]]:
        """Rotate and concatenate a single segment for dense_prefill mode."""
        new_ph, new_pf = self._rotate_segment_caches(m)
        seg_cache = new_ph.concat_([new_pf])
        seg_token_ids = concat(self.trim_token_ids(m["ph_cache_ids"], m["drop_num"]), m["pf_ids"])

        return m["idx"], seg_cache, seg_token_ids

    def process_anchor_tokens(
        self,
        message: str,
        m: Dict[str, Any],
    ) -> Tuple[int, Dict[str, torch.Tensor]]:
        """Stage-2 dense: build ONLY the rotated token layout for a segment — the base cache is never rotated or concatenated. """
        seg_token_ids = concat(self.trim_token_ids(m["ph_cache_ids"], m["drop_num"]), m["pf_ids"])
        return m["idx"], seg_token_ids
