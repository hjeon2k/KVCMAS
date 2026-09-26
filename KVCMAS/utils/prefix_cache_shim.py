"""True cross-request prefix caching for the EFFICIENCY harness only. """
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import torch

_FLAG = "EFF_TRUE_PREFIX_CACHE"


def enabled() -> bool:
    """Default ON. """
    return os.environ.get(_FLAG, "1").strip().lower() in ("1", "true", "yes", "on")


class _PrefixEntry:
    # NOTE: __slots__ must list EVERY attribute __init__ assigns.
    __slots__ = ("common", "seen", "cache", "length", "hits", "prefix_ids", "decided",
                 "shrinks", "disabled", "stable_for")

    def __init__(self) -> None:
        # Running intersection of every prompt seen for THIS AGENT, not a pairwise LCP of adjacent
        # calls.
        self.common: Optional[torch.Tensor] = None
        self.seen: int = 0
        self.cache: Optional[Any] = None             # DynamicCache over the shared prefix
        self.length: int = 0
        self.hits: int = 0
        self.prefix_ids: Optional[torch.Tensor] = None
        # Learning is one-shot: once decided, stop intersecting and stop re-measuring,
        # whether a prefix was worth caching.
        self.decided: bool = False
        self.shrinks: int = 0
        self.disabled: bool = False
        # Consecutive observations that did NOT shrink the intersection. Commit on
        # stability, never on a call count -- see PrefixCacheShim.stable_needed.
        self.stable_for: int = 0


class PrefixCacheShim:
    """Wraps ``model.generate`` to serve an exact, unchanging token prefix from cache."""

    def __init__(self, model, tokenizer=None, min_prefix: int = 32, stable_needed: int = 3,
                 key_window: int = 512):
        self.model = model
        self.tokenizer = tokenizer
        self.min_prefix = int(min_prefix)
        # Commit only once the intersection has held steady this many consecutive observations.
        self.stable_needed = max(2, int(stable_needed))
        # Window used to identify an agent.
        self.key_window = int(key_window)
        self._entries: Dict[Any, _PrefixEntry] = {}
        self._orig_generate = None
        self.stats = {"learned": 0, "hits": 0, "skipped_has_cache": 0, "misses": 0,
                      "shrinks": 0}

    # -- lifecycle ---------------------------------------------------------------
    def install(self) -> "PrefixCacheShim":
        if self._orig_generate is not None:
            return self
        self._orig_generate = self.model.generate
        shim = self

        def generate(*args, **kwargs):
            return shim._generate(*args, **kwargs)

        self.model.generate = generate
        return self

    def uninstall(self) -> None:
        if self._orig_generate is not None:
            self.model.generate = self._orig_generate
            self._orig_generate = None
        self._entries.clear()

    # -- core --------------------------------------------------------------------
    def _key(self, input_ids: torch.Tensor) -> Any:
        """One entry per AGENT, keyed by a window that lies inside its system prompt. """
        n = min(self.key_window, int(input_ids.shape[-1]))
        return hash(tuple(input_ids[0, :n].tolist()))

    def _generate(self, *args, **kwargs):
        orig = self._orig_generate
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        # A cache that already HOLDS tokens owns its own prefix handling: CacheBlend's fused cache
        # and our streaming base segments have the prefix inside them.
        _pkv = kwargs.get("past_key_values")
        _pkv_len = 0
        if _pkv is not None:
            try:
                _pkv_len = int(_pkv.get_seq_length())
            except Exception:
                _pkv_len = -1                      # unknown: treat as populated, stay out
        if _pkv is not None and _pkv_len != 0:
            self.stats["skipped_has_cache"] += 1
            return orig(*args, **kwargs)
        if not isinstance(input_ids, torch.Tensor) or input_ids.dim() != 2 or input_ids.shape[0] != 1:
            self.stats["misses"] += 1
            return orig(*args, **kwargs)

        # Declared-prefix mode.
        declared = int(os.environ.get("SYSTEM_PROMPT", "0") or 0)
        if declared > 0 and int(input_ids.shape[-1]) > declared:
            entry = self._entries.setdefault(("declared", declared), _PrefixEntry())
            if entry.cache is None and not entry.disabled:
                self._build(entry, input_ids[0, :declared])
                self.stats["learned"] += 1
                print(f"[PREFIX-CACHE] declared prefix = {declared} tokens "
                      f"(SYSTEM_PROMPT; no inference)", flush=True)
            if entry.cache is not None:
                if self._matches(entry, input_ids):
                    kwargs = dict(kwargs)
                    seeded = self._clone(entry.cache)
                    if _pkv is not None:
                        try:
                            from KVCMAS.utils.prealloc_cache import to_prealloc
                            seeded = to_prealloc(
                                seeded,
                                extra_tokens=int(input_ids.shape[-1]) - entry.length + 512)
                        except Exception:
                            pass
                    kwargs["past_key_values"] = seeded
                    self.stats["hits"] += 1
                    entry.hits += 1
                    return orig(*args, **kwargs)
                # The declared prefix is not invariant, i.e. the layout is not what we were
                # told. Report it and fall back to inference rather than silently degrading.
                print(f"[PREFIX-CACHE] WARNING declared prefix of {declared} tokens is not "
                      f"invariant across prompts; falling back to inference", flush=True)
                self.stats["shrinks"] += 1
                entry.cache = None
                entry.disabled = True

        entry = self._entries.setdefault(self._key(input_ids), _PrefixEntry())

        if entry.cache is not None and int(input_ids.shape[-1]) > entry.length:
            if not self._matches(entry, input_ids):
                # A committed prefix that stops being a prefix means the learned head was not
                # actually invariant.
                d = self._lcp(entry.prefix_ids, input_ids[0])
                entry.shrinks += 1
                print(f"[PREFIX-CACHE] WARNING committed prefix ({entry.length} tok) "
                      f"diverges at token {d}; shrink #{entry.shrinks}", flush=True)
                self.stats["shrinks"] += 1
                if entry.shrinks > 2 or d < self.min_prefix:
                    print("[PREFIX-CACHE] WARNING prefix is not invariant for this agent; "
                          "disabling its cache", flush=True)
                    entry.cache = None
                    entry.disabled = True
                else:
                    self._build(entry, input_ids[0, :d])
            if entry.cache is not None and self._matches(entry, input_ids):
                kwargs = dict(kwargs)
                seeded = self._clone(entry.cache)
                if _pkv is not None:
                    # Preserve the caller's prealloc policy: give back a PreallocCache that
                    # already holds the prefix, sized for the prompt remainder + decode.
                    try:
                        from KVCMAS.utils.prealloc_cache import to_prealloc
                        room = int(input_ids.shape[-1]) - entry.length + 512
                        seeded = to_prealloc(seeded, extra_tokens=room)
                    except Exception:
                        pass                        # plain DynamicCache still correct
                kwargs["past_key_values"] = seeded
                self.stats["hits"] += 1
                entry.hits += 1
                return orig(*args, **kwargs)

        if not entry.decided and not entry.disabled:
            cur = input_ids[0].detach().clone()
            if entry.common is None:
                entry.common = cur
                _before = -1                  # first sighting is not "stable"
            else:
                _before = int(entry.common.numel())
                entry.common = entry.common[: self._lcp(entry.common, cur)]
            entry.seen += 1
            if int(entry.common.numel()) == _before:
                entry.stable_for += 1
            else:
                entry.stable_for = 0          # shrank: not converged yet
            if entry.stable_for >= self.stable_needed:
                entry.decided = True
                n = int(entry.common.numel())
                if n >= self.min_prefix:
                    self._build(entry, entry.common)
                    self.stats["learned"] += 1
                    # This length is the headline diagnostic: it is the ONLY exactly cacheable
                    # part of the prompt.
                    print(f"[PREFIX-CACHE] learned prefix = {n} tokens after "
                          f"{entry.seen} calls ({100.0 * n / max(1, int(input_ids.shape[-1])):.1f}% "
                          f"of a {int(input_ids.shape[-1])}-token prompt)", flush=True)
                else:
                    print(f"[PREFIX-CACHE] common prefix only {n} tokens after "
                          f"{entry.seen} calls (< min {self.min_prefix}); nothing to cache",
                          flush=True)
        self.stats["misses"] += 1
        return orig(*args, **kwargs)

    def report(self) -> str:
        learned = [(e.length, e.hits) for e in self._entries.values() if e.cache]
        return (f"[PREFIX-CACHE] entries={len(self._entries)} learned={self.stats['learned']} "
                f"hits={self.stats['hits']} misses={self.stats['misses']} "
                f"skipped_populated_cache={self.stats['skipped_has_cache']} "
                f"shrinks={self.stats['shrinks']} "
                f"prefixes={learned}")

    @staticmethod
    def _lcp(a: torch.Tensor, b: torch.Tensor) -> int:
        n = int(min(a.numel(), b.numel()))
        if n == 0:
            return 0
        eq = (a[:n] == b[:n])
        nz = (~eq).nonzero()
        return int(nz[0].item()) if nz.numel() else n

    def _matches(self, entry: _PrefixEntry, input_ids: torch.Tensor) -> bool:
        L = entry.length
        if int(input_ids.shape[-1]) <= L or entry.prefix_ids is None:
            return False
        return bool(torch.equal(input_ids[0, :L], entry.prefix_ids))

    def _build(self, entry: _PrefixEntry, prefix_ids: torch.Tensor) -> None:
        """One forward over the prefix; its KV becomes the reusable constant."""
        ids = prefix_ids.unsqueeze(0).to(self.model.device)
        with torch.no_grad():
            out = self.model(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                use_cache=True,
                return_dict=True,
                # Only the KV is wanted; without this the direct forward materializes [1,
                # prefix_len, vocab] logits (fp32 after Llama's upcast, ~0.5 GB for a 1k.
                logits_to_keep=1,
            )
        entry.cache = out.past_key_values
        entry.length = int(prefix_ids.numel())
        entry.prefix_ids = prefix_ids.detach().clone()

    @staticmethod
    def _clone(cache) -> Any:
        """generate() appends to the cache it is given, so hand it a private copy. """
        import copy as _copy

        keys: List[torch.Tensor] = getattr(cache, "key_cache", None)
        vals: List[torch.Tensor] = getattr(cache, "value_cache", None)
        if keys is None or vals is None:
            return _copy.deepcopy(cache)
        new = _copy.copy(cache)
        new.key_cache = [k.clone() for k in keys]
        new.value_cache = [v.clone() for v in vals]
        return new


def maybe_install(model, tokenizer=None) -> Optional[PrefixCacheShim]:
    """Install and return the shim when the flag is set, else None."""
    if not enabled():
        return None
    return PrefixCacheShim(model, tokenizer).install()
