"""Chat backends and local HF model with KV reuse and anchor-based prefill. """
from typing import List, Union, Optional, Dict, Any, Tuple
import json
from tenacity import retry, wait_random_exponential, stop_after_attempt
from dotenv import load_dotenv
import os
import time
from pathlib import Path
from time import perf_counter
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
import torch
import threading
import random as _random
import asyncio
import async_timeout
from openai import AsyncOpenAI
import re
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from transformers.cache_utils import DynamicCache
from KVCMAS.llm.format import Message
from KVCMAS.llm.llm import LLM
from KVCMAS.llm.llm_registry import LLMRegistry
from KVCMAS.llm.config import KVCommConfig


class _ConfigCarrier:
    """Minimal stand-in exposing only ``.config`` for the engine's classmethod APIs."""
    __slots__ = ("config",)
    def __init__(self, config: KVCommConfig):
        self.config = config

from KVCMAS.llm.token_ops import *
from KVCMAS.utils.prealloc_cache import PreallocCache, prealloc_enabled, to_prealloc
from KVCMAS.llm.kvcmas_engine import (
    KVCMASEngine,
    _RequestState,
    StreamingCorrectionCache,
    _is_question_span,
    _lkv_chain_delta,
)
from KVCMAS.utils.metrics import GenerationResult
from KVCMAS.utils.log import logger



MINE_API_KEYS = os.getenv('API_KEY')

def _token_reuse_metadata(meta, mode: str, q_has_source: bool = True) -> Dict[str, Any]:
    """Per-hop reuse counters for one agent call (no ratio -- see utils/metrics)."""
    shared = 0
    reusable = 0
    # q_has_source=False identifies the request's SOURCE HOP, and that WHOLE hop leaves the
    # denominator -- not merely its question span.
    is_source_hop = not q_has_source
    for m in meta:
        # `_span_len` is captured when meta is built so the base TENSORS can be released
        # before the harvest; this metric only ever needed the length.
        n = m.get("_span_len")
        if n is None:
            n = int(m["ph_cache"]._seen_tokens) - int(m["drop_num"])
        n = int(n)
        shared += n
        if is_source_hop:
            continue        # source hop: nothing was reusable, so nothing enters the denominator
        reusable += n
    reused = reusable if mode == "kv_reuse" else 0
    recomputed = reusable - reused
    assert reused + recomputed == reusable, "token partition must be exact"
    # Counters only. The reported reuse ratio is the run-level one in utils/metrics;
    # `is_source_hop` keeps the source hop out of its denominator.
    return {
        "is_source_hop": bool(is_source_hop),
        "shared_tokens": int(shared),
        "reusable_tokens": int(reusable),
        "reused_tokens": int(reused),
        "recomputed_tokens": int(recomputed),
    }



def _escape_loguru_markup(text: Optional[str]) -> str:
    """Escape Loguru markup tokens in free-form text."""
    if text is None:
        return ""
    return text.replace("<", "\\<")


_LATENCY_IO_LOCK = threading.Lock()


def _resolve_latency_path(target: Optional[Union[str, Path]]) -> Optional[Path]:
    if target is None:
        return None
    path = Path(target)
    # `.suffix` is not a file/directory test.
    if path.suffix.lower() == ".json" and not path.is_dir():
        return path
    return path / "latency.json"


def _append_latency_record(target: Optional[Union[str, Path]], record: Dict[str, Any]) -> None:
    """Persist a latency record to JSON, tolerating malformed or missing files."""
    path = _resolve_latency_path(target)
    if path is None:
        return
    serializable = {key: value for key, value in record.items() if value is not None}
    with _LATENCY_IO_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        existing: List[Dict[str, Any]] = []
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    loaded = json.load(handle)
                    if isinstance(loaded, list):
                        existing = loaded
            except (json.JSONDecodeError, OSError):
                existing = []
        existing.append(serializable)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(existing, handle, ensure_ascii=False, indent=2)


class _TTFTTracer(StoppingCriteria):
    """Stopping criteria to capture time-to-first-token during generation."""

    def __init__(self, prompt_length: int):
        self.prompt_length = prompt_length
        self.start_time = perf_counter()
        self.ttft: Optional[float] = None

    def reset(self, prompt_length: int) -> None:
        self.prompt_length = prompt_length
        self.start_time = perf_counter()
        self.ttft = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any) -> bool:
        if self.ttft is None and input_ids.shape[-1] > self.prompt_length:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.ttft = perf_counter() - self.start_time
        return False

@retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(3))
async def achat(model: str, msg: List[Dict],):
    """Call an OpenAI-compatible chat endpoint asynchronously."""
    api_kwargs = dict(api_key = MINE_API_KEYS)
    try:
        aclient = AsyncOpenAI(**api_kwargs)
    except Exception as e:
        raise RuntimeError(f"Failed to create the async client: {e}")
    try:
        async with async_timeout.timeout(1000):
            completion = await aclient.chat.completions.create(model=model,messages=msg)
        response_message = completion.choices[0].message.content

        if isinstance(response_message, str):
            prompt = "".join([item['content'] for item in msg])
            return response_message

    except Exception as e:
        raise RuntimeError(f"Failed to complete the async chat request: {e}")    

@LLMRegistry.register('GPTChat')
class GPTChat(LLM):
    """Thin wrapper around OpenAI-style chat completions."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    async def agen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        *,
        request_uid: Optional[str] = None,
        agent_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        agent_role: Optional[str] = None,
    ) -> GenerationResult:
        """Asynchronously generate a response via hosted chat API."""

        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE

        if isinstance(messages, str):
            messages = [Message(role="user", content=messages)]
        response_text = await achat(self.model_name, messages)
        metadata: Dict[str, Any] = {}
        if request_uid:
            metadata["request_uid"] = request_uid
        if agent_id:
            metadata["agent_id"] = agent_id
        if agent_name:
            metadata["agent_name"] = agent_name
        if agent_role:
            metadata["agent_role"] = agent_role
        return GenerationResult(
            text=response_text,
            mode="default",
            ttft=0.0,
            metadata=metadata,
        )

    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Union[List[str], str]:
        """Synchronous generation not implemented for this adapter."""
        pass

def _is_vlm_name(name: str) -> bool:
    """VLM checkpoints routed to the wrapper loader. OneVision only for now: native at
    transformers 4.50.2 with a Qwen2 LM; M-RoPE families are excluded."""
    return "llava-onevision" in (name or "").lower()


@LLMRegistry.register('LLMChat')
class LLMChat(LLM):
    """Local HF model chat with KV reuse and anchor-based dense prefill. """
    _shared_model = None
    _shared_processor = None            # VLM only (AutoProcessor)
    _vlm_image_token_id = None          # id of <image> (151646 for OneVision)
    # B=1 "current image" slot.
    _vlm_current_image = None           # dict(pixel_values=..., image_sizes=...)
    _shared_tokenizer = None
    _model_lock = threading.Lock()
    _THREAD_POOL: ThreadPoolExecutor | None = None
    _THREAD_POOL_WORKERS: int | None = None
    _shared_kv_cache_memory = None
    # NONSHARED baseline "KV holding" (KVCMAS_HOLD_KV=1): each agent keeps its full KV resident on
    # GPU (per node_id).
    _held_kv_cache: Dict[Any, Any] = {}
    _initialization = {}
    _last_config: KVCommConfig | None = None  # for the classmethod finalize_request
    anchors = KVCMASEngine.anchors
    anchor_dict = KVCMASEngine.anchor_dict
    anchor_len_dict = KVCMASEngine.anchor_len_dict
    anchor_info_dict = KVCMASEngine.anchor_info_dict
    weight_dict = KVCMASEngine.weight_dict
    global_anchor_info_dict = KVCMASEngine.global_anchor_info_dict

    _request_lock = KVCMASEngine._request_lock
    _request_states = KVCMASEngine._request_states
    _active_requests = KVCMASEngine._active_requests
    _staged_commits = KVCMASEngine._staged_commits

    def __init__(self, model_name: str, prefix: str = None, config: KVCommConfig | None = None):
        """Create a chat model instance and initialize shared resources. """
        self.model_name = model_name

        self.config = (config or KVCommConfig.from_env()).validate()
        LLMChat._last_config = self.config
        # Surface the effective KVCMAS rank so every run self-documents whether SVD delta
        # compression was active (svd_rank>0) or vanilla KVComm (svd_rank==0).
        logger.opt(colors=True).info(
            "<cyan>[KVCMAS CONFIG]</cyan> model={} svd_rank(ph={}, pf={}) base(key={}, value={}) "
            "(0 = vanilla / no compression) drop_shared_cache={} threshold={} max_anchor_num={} window_size={}",
            model_name, self.config.rank_ph, self.config.rank_pf,
            self.config.rank_base_key, self.config.rank_base_value,
            self.config.drop_shared_cache_on_finalize, self.config.threshold,
            self.config.max_anchor_num, self.config.window_size,
        )
        self._ensure_thread_pool(self.config.thread_pool_workers)
        self.kv_engine = KVCMASEngine(self)

        self.lock = asyncio.Lock()                       


        self._initialize_shared_resources()


        self.tokenizer = LLMChat._shared_tokenizer
        self.model = LLMChat._shared_model
        self._shared_kv_cache_memory = LLMChat._shared_kv_cache_memory
        self._initialization = LLMChat._initialization
        self._chat_markers = self._extract_chat_markers()
        self.default_assistant_prompt = "A: "
        self.base_messages_template: List[Dict[str, str]] = [
            {"role": "system", "content": "{system_prompt}"},
            {"role": "user", "content": "{user_prompt}"},
        ]
        if prefix is not None:
            self._prepare_prefix_template(prefix)

    def _extract_chat_markers(self) -> Dict[str, str]:
        """Parse tokenizer chat template to identify structural markers."""
        template = getattr(self.tokenizer, "chat_template", "") or ""
        markers = {"begin": "", "start": "", "end": "", "eot": ""}
        begin_candidates = ["<|begin_of_text|>", "<s>", getattr(self.tokenizer, "bos_token", "") or ""]
        start_candidates = ["<|start_header_id|>", "<|im_start|>"]
        end_candidates = ["<|end_header_id|>", "<|im_end|>", "\n"]
        eot_candidates = ["<|eot_id|>", "<|im_end|>", getattr(self.tokenizer, "eos_token", "") or ""]

        for token in begin_candidates:
            if token and token in template:
                markers["begin"] = token
                break
        if not markers["begin"]:
            markers["begin"] = begin_candidates[-1]

        for token in start_candidates:
            if token and token in template:
                markers["start"] = token
                break

        for token in end_candidates:
            if token and token in template:
                markers["end"] = token
                break
        if not markers["end"]:
            markers["end"] = ""

        for token in eot_candidates:
            if token and token in template:
                markers["eot"] = token
                break
        if not markers["eot"]:
            markers["eot"] = eot_candidates[-1]

        return markers

    def _prepare_prefix_template(self, prefix: Union[str, List[Dict[str, str]]]) -> None:
        """Normalise various prefix formats into a base messages template."""
        if isinstance(prefix, list):
            self.base_messages_template = prefix
            return
        if isinstance(prefix, dict):
            self.base_messages_template = [prefix]
            return
        if isinstance(prefix, tuple):
            prefix = list(prefix)
        if isinstance(prefix, list) and all(isinstance(item, tuple) and len(item) == 2 for item in prefix):
            self.base_messages_template = [{"role": role, "content": tmpl} for role, tmpl in prefix]
            return
        if isinstance(prefix, str):
            self.default_assistant_prompt = self._extract_assistant_prompt(prefix)
            return
        raise TypeError("Unsupported prefix template type.")

    def _extract_assistant_prompt(self, legacy_prefix: str) -> str:
        """Extract trailing assistant prompt from a legacy text prefix."""
        start = self.start_header_id
        end = self.end_header_id
        if start and end:
            marker = f"{start}assistant{end}\n"
            if marker in legacy_prefix:
                tail = legacy_prefix.split(marker, 1)[-1]
                eot = self.eot_id
                if eot:
                    tail = tail.replace(eot, "")
                return tail
        return legacy_prefix

    @property
    def begin_of_text(self) -> str:
        return self._chat_markers.get("begin", "")

    @property
    def start_header_id(self) -> str:
        return self._chat_markers.get("start", "")

    @property
    def end_header_id(self) -> str:
        return self._chat_markers.get("end", "")

    @property
    def eot_id(self) -> str:
        return self._chat_markers.get("eot", "")

    @staticmethod
    def _normalise_messages(messages: Union[List[Message], List[Dict[str, str]], Dict[str, Any], Tuple[Any, ...], str]) -> List[Dict[str, str]]:
        """Convert mixed message representations into chat dicts."""
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]
        if isinstance(messages, tuple):
            if len(messages) == 2 and all(isinstance(item, str) for item in messages):
                system_prompt, user_prompt = messages
                return [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            return LLMChat._normalise_messages(list(messages))
        if isinstance(messages, dict):
            result: List[Dict[str, str]] = []
            system_prompt = messages.get("system") or messages.get("system_prompt")
            if system_prompt:
                result.append({"role": "system", "content": system_prompt})
            conversation = messages.get("messages") or messages.get("conversation")
            if conversation is not None:
                result.extend(LLMChat._normalise_messages(conversation))
            else:
                if "user" in messages:
                    user_payload = messages["user"]
                    if isinstance(user_payload, list):
                        result.extend(LLMChat._normalise_messages(user_payload))
                    else:
                        result.append({"role": "user", "content": user_payload})
                if "assistant" in messages:
                    assistant_payload = messages["assistant"]
                    if isinstance(assistant_payload, list):
                        result.extend(LLMChat._normalise_messages(assistant_payload))
                    else:
                        result.append({"role": "assistant", "content": assistant_payload})
            return result
        if not isinstance(messages, list):
            raise TypeError("messages must be a string, sequence, or a list of Message/Dict objects.")
        if messages and isinstance(messages[0], Message):
            return [{"role": m.role, "content": m.content} for m in messages]
        normalised: List[Dict[str, str]] = []
        for item in messages:
            if isinstance(item, Message):
                normalised.append({"role": item.role, "content": item.content})
            elif isinstance(item, dict):
                if "role" in item:
                    normalised.append({"role": item["role"], "content": item.get("content", "")})
                else:
                    normalised.extend(LLMChat._normalise_messages(item))
            elif isinstance(item, str):
                normalised.append({"role": "user", "content": item})
            else:
                normalised.extend(LLMChat._normalise_messages(item))
        return normalised

    def _legacy_prompt_from_messages(self, messages: List[Dict[str, str]]) -> str:
        """Fallback prompt renderer when chat_template is unavailable."""
        prompt_parts = [self.begin_of_text or getattr(self.tokenizer, "bos_token", "") or ""]
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            start = self.start_header_id
            end = self.end_header_id
            eot = self.eot_id or getattr(self.tokenizer, "eos_token", "") or ""
            if start and end:
                prompt_parts.append(f"{start}{role}{end}\n{content}{eot}")
            else:
                prompt_parts.append(f"[{role.upper()}]\n{content}{eot}")
        if self.start_header_id and self.end_header_id:
            prompt_parts.append(f"{self.start_header_id}assistant{self.end_header_id}\n")
        else:
            prompt_parts.append("[ASSISTANT]\n")
        return "".join(prompt_parts)

    def _build_chat_inputs(
        self,
        messages: Union[List[Message], List[Dict[str, str]], str],
        assistant_prompt: Optional[str] = None,
        add_generation_prompt: bool = True,
    ) -> Tuple[Dict[str, torch.Tensor], str, int]:
        """Tokenize chat messages and return model inputs, text, and prompt length."""
        normalised = self._normalise_messages(messages)
        assistant_prompt = assistant_prompt or self.default_assistant_prompt
        prompt_text = ""
        try:
            prompt_text = self.tokenizer.apply_chat_template(
                normalised,
                add_generation_prompt=add_generation_prompt,
                tokenize=False,
            ) + assistant_prompt

            tokenized = self.tokenizer.encode(prompt_text, return_tensors="pt", add_special_tokens=False)

            if isinstance(tokenized, dict):
                inputs = tokenized
            else:
                inputs = {
                    "input_ids": tokenized,
                    "attention_mask": torch.ones_like(tokenized),
                }
        except (ValueError, AttributeError, NotImplementedError, TypeError):

            prompt_text = self._legacy_prompt_from_messages(normalised) + assistant_prompt
            inputs = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
        inputs = {
            k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }
        input_length = inputs["input_ids"].shape[-1]
        return inputs, prompt_text, input_length

    def _render_base_messages(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> List[Dict[str, str]]:
        """Render base messages from the current template with provided text."""
        rendered: List[Dict[str, str]] = []
        template = self.base_messages_template or [
            {"role": "system", "content": "{system_prompt}"},
            {"role": "user", "content": "{user_prompt}"},
        ]
        for block in template:
            content_template = block.get("content", "")
            content = content_template.format(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            rendered.append({"role": block.get("role", "user"), "content": content})
        return rendered

    def format_chat_segment(
        self,
        role: str,
        content: str,
        *,
        include_begin: bool = False,
        include_eot: bool = True,
    ) -> str:
        """Render a single chat block for the given role and content."""
        prefix = self.begin_of_text if include_begin else ""
        start = self.start_header_id
        end = self.end_header_id
        eot = self.eot_id if include_eot else ""
        if start and end:
            return f"{prefix}{start}{role}{end}\n{content}{eot}"
        upper_role = role.upper()
        return f"{prefix}[{upper_role}]\n{content}{eot}"

    def tokenize_segment(
        self,
        role: str,
        content: str,
        *,
        include_begin: bool = False,
        include_eot: bool = True,
        add_special_tokens: bool = False,
        return_tensors: Optional[str] = "pt",
    ) -> Dict[str, torch.Tensor]:
        """Tokenize a single chat segment and move tensors to the model device."""
        text = self.format_chat_segment(
            role,
            content,
            include_begin=include_begin,
            include_eot=include_eot,
        )
        tokens = self.tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            return_tensors=return_tensors,
        )
        return {
            k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v
            for k, v in tokens.items()
        }

    def build_prompt(
        self,
        system_prompt: str,
        user_prompt: str,
        assistant_prompt: Optional[str] = None,
        *,
        add_generation_prompt: bool = True,
        return_messages: bool = False,
    ) -> Dict[str, Any]:
        """Create model inputs from system/user prompts and optional assistant suffix."""
        messages = self._render_base_messages(system_prompt, user_prompt)
        inputs, prompt_text, prompt_length = self._build_chat_inputs(
            messages,
            assistant_prompt=assistant_prompt,
            add_generation_prompt=add_generation_prompt,
        )
        result: Dict[str, Any] = {
            "inputs": inputs,
            "prompt_text": prompt_text,
            "prompt_length": prompt_length,
        }
        if return_messages:
            result["messages"] = messages
        return result

    @classmethod
    def _ensure_thread_pool(cls, workers: int) -> None:
        """Initialise or resize the shared thread pool used for CPU work."""
        if cls._THREAD_POOL is None or cls._THREAD_POOL_WORKERS != workers:
            if cls._THREAD_POOL is not None:
                cls._THREAD_POOL.shutdown(wait=False)
            cls._THREAD_POOL = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="LLM-chat")
            cls._THREAD_POOL_WORKERS = workers

    @classmethod
    def finalize_request(cls, request_uid: str) -> None:
        proxy = _ConfigCarrier(cls._last_config) if cls._last_config is not None else None
        KVCMASEngine.finalize_request(request_uid, llm=proxy)

    def get_request_state(self, request_uid: str) -> "_RequestState":
        """Return per-request state used by the KV engine."""
        return self.kv_engine.get_request_state(request_uid)

    def _ensure_agent_memory(self, agent_id: str) -> Dict[str, Any]:
        """Return the shared memory slot for a given agent id."""
        return LLMChat._shared_kv_cache_memory.setdefault(agent_id, {})

    def _ensure_global_input_buckets(self) -> Dict[str, Dict[str, Any]]:
        """Ensure the global input buckets exist and return the shared store."""
        store = LLMChat._shared_kv_cache_memory
        store.setdefault("input", {})
        store.setdefault("input_ids", {})
        store.setdefault("input_drop_num", {})
        return store

    def has_prefix_initialized(self, agent_id: str) -> bool:
        """Check if prefix KV has been initialized for an agent."""
        return LLMChat._initialization.get(agent_id, False)

    def has_active_anchor(self, request_uid: str, message: str) -> bool:
        """Determine whether an anchor should trigger dense prefill."""
        state = self.get_request_state(request_uid)
        ph_ids = LLMChat._shared_kv_cache_memory.get(self.node_id, {}).get('placeholder_info', {}).keys()
        for ph_id in ph_ids:
            bucket = state.anchor_dict.setdefault(ph_id, {})
            if bucket.get(message) is True and f'{self.node_id}_ph_key_delta' not in state.anchors.get(ph_id, {}).get(message, {}):
                return True
        return False

    def update_condition_anchor(
        self,
        *,
        request_uid: str,
        owner_agent_id: str,
        message: str,
        content: str,
        prefix_text: str,
        role: str = "user",
        include_begin: bool = True,
        include_eot: bool = False,
        anchor_namespace: Optional[str] = None,
        max_length: int = None,
    ) -> bool:
        """Materialise condition KV cache for another agent and update anchors."""
        state = self.get_request_state(request_uid)
        anchor_key = anchor_namespace or f"condition_{owner_agent_id}_current"

        owner_memory = self._ensure_agent_memory(owner_agent_id)
        condition_bucket = owner_memory.setdefault("condition", {})
        if message in condition_bucket:

            return state.anchor_dict.setdefault(anchor_key, {}).get(message, False)

        token_ids = self.tokenize_segment(
            role=role,
            content=content,
            include_begin=include_begin,
            include_eot=include_eot,
            add_special_tokens=False,
        )
        if "position_ids" not in token_ids:
            position_ids = torch.arange(token_ids["input_ids"].shape[-1]).unsqueeze(0)
            token_ids["position_ids"] = position_ids.to(self.model.device)
        else:
            token_ids["position_ids"] = token_ids["position_ids"].to(self.model.device)
        token_ids["input_ids"] = token_ids["input_ids"].to(self.model.device)
        token_ids["attention_mask"] = token_ids["attention_mask"].to(self.model.device)

        prefix_ids = self.tokenize_segment(
            role=role,
            content=prefix_text,
            include_begin=include_begin,
            include_eot=include_eot,
            add_special_tokens=False,
        )["input_ids"]
        drop_num = prefix_ids.shape[-1]

        if max_length is not None:
            token_ids["input_ids"] = token_ids["input_ids"][:, :drop_num + max_length]
            token_ids["attention_mask"] = token_ids["attention_mask"][:, :drop_num + max_length]
            token_ids["position_ids"] = token_ids["position_ids"][:, :drop_num + max_length]
            
        generated = self.model.generate(
            **self._merge_vlm_inputs(token_ids),
            use_cache=True,
            do_sample=False,
            temperature=None,
            top_p=None,
            max_length=token_ids["input_ids"].shape[-1] + 1,
            return_dict_in_generate=True,
            return_legacy_cache=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        condition_cache = generated.past_key_values


        for key_name, value in (
            ("condition", condition_cache),
            ("condition_ids", token_ids),
            ("condition_drop_num", drop_num),
        ):
            bucket = owner_memory.setdefault(key_name, {})
            bucket.setdefault(message, []).append(value)

        anchor_store = state.anchors.setdefault(anchor_key, {})
        cond_anchor_list = list(anchor_store.values())
        cond_len_bucket = state.anchor_len_dict.setdefault(anchor_key, {})
        anchor_len_list = [
            cond_len_bucket.get(entry_key, [0, 0])
            for entry_key in anchor_store.keys()
        ]
        cond_info_bucket = state.anchor_info_dict.setdefault(anchor_key, {})
        anchor_activated_list = list(cond_info_bucket.values())

        total_prefix_len = 0
        for bucket in state.anchor_len_dict.values():
            total_prefix_len += bucket.get(message, [0, 0])[0]

        prob, anchor_activated_list = self.kv_engine.predict_as_anchor(
            condition_cache.copy().slice_(start=drop_num),
            anchor_kv_cache_list=cond_anchor_list,
            anchor_len_list=anchor_len_list,
            anchor_activated_list=anchor_activated_list,
        )

        cond_flag_bucket = state.anchor_dict.setdefault(anchor_key, {})
        cond_flag_bucket[message] = prob

        global_bucket = state.global_anchor_info.setdefault(anchor_key, {})
        if not prob:
            info_items = list(cond_info_bucket.items())
            for idx, (msg_key, _) in enumerate(info_items):
                cond_info_bucket[msg_key] = anchor_activated_list[idx]
                bucket_entry = global_bucket.setdefault(msg_key, [0, 0])
                bucket_entry[0] = anchor_activated_list[idx]
        else:
            cond_info_bucket[message] = 0
            global_bucket[message] = [
                0,
                condition_cache.get_seq_length() - drop_num,
            ]
        return prob

    # ---- question harvest: the delta base comes from hop 0's own prefill ---------
    def _create_blank_kv_cache(self, batch_size: int, sequence_length: int) -> DynamicCache:
        """Length-only filler cache for a span whose content is never read. """
        cfg = self.model.config
        num_layers = int(cfg.num_hidden_layers)
        num_kv_heads = int(getattr(cfg, "num_key_value_heads", cfg.num_attention_heads))
        head_dim = int(getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads)
        shape = (batch_size, num_kv_heads, sequence_length, head_dim)
        blank = DynamicCache()
        for _ in range(num_layers):
            k1 = torch.full((batch_size, num_kv_heads, 1, head_dim), -1,
                            dtype=self.model.dtype, device=self.model.device)
            v1 = torch.full((batch_size, num_kv_heads, 1, head_dim), -1,
                            dtype=self.model.dtype, device=self.model.device)
            blank.key_cache.append(k1.expand(shape))
            blank.value_cache.append(v1.expand(shape))
        blank._seen_tokens = int(sequence_length)
        return blank

    def _fs_install_blank(self, message: str, *, role: str, user_content: str,
                          prefix_text: str, include_begin: bool, include_eot: bool) -> None:
        """Seed the input bucket with a length-only filler, tagged BLANK. No forward pass."""
        token_ids = self.tokenize_segment(
            role=role, content=user_content, include_begin=include_begin,
            include_eot=include_eot, add_special_tokens=False,
        )
        n = int(token_ids["input_ids"].shape[-1])
        position_ids = token_ids.get("position_ids")
        if position_ids is None:
            position_ids = torch.arange(n, dtype=torch.long).unsqueeze(0)
        token_ids["position_ids"] = position_ids.to(self.model.device)
        drop_num = int(self.tokenize_segment(
            role=role, content=prefix_text, include_begin=include_begin,
            include_eot=include_eot, add_special_tokens=False,
        )["input_ids"].shape[-1])
        buckets = self._ensure_global_input_buckets()
        buckets["input"][message] = [self._create_blank_kv_cache(1, n)]
        buckets["input_ids"][message] = [token_ids]
        buckets["input_drop_num"][message] = [drop_num]
        buckets.setdefault("input_tags", {})[message] = ["BLANK"]

    def _fs_harvest_question(self, message: str, full_kv_cache, placeholder_indices) -> None:
        """Replace the BLANK filler with hop 0's real question KV, re-rotated to 0-base. """
        buckets = self._ensure_global_input_buckets()
        tags = buckets.setdefault("input_tags", {})
        if not _lkv_chain_delta() and (tags.get(message) or ["BLANK"])[-1] == "REAL":
            # STAR: exactly one harvest per request -- the bucket keeps hop 0's version, so every
            # hop's delta is taken against that one fixed base.
            return
        q_span = placeholder_indices.get("user_question")
        if q_span is None:
            return
        q_lo, q_hi = int(q_span[0]), int(q_span[1])
        if q_hi <= q_lo:
            return
        old_ids = buckets["input_ids"][message][-1]
        old_drop = buckets["input_drop_num"][message][-1]
        # Drop the filler before materializing the real span: the filler is a stride-0
        # view, but the assignment order still matters for the real cache's peak.
        buckets["input"][message] = []
        # Rotate from per-layer VIEWS of the live cache rather than from a contiguous copy.
        if not isinstance(full_kv_cache, StreamingCorrectionCache) and hasattr(full_kv_cache, "slice_view"):
            # REPLACE the corrected cache with the new bucket instead of holding both.
            _consume_src = (_lkv_chain_delta()
                            and os.environ.get("KVCMAS_CONSUME_MATERIALIZED", "1").strip().lower()
                            not in ("0", "false", "no", "off"))
            q_cache = self.kv_engine.apply_rotary_pos_emb(
                full_kv_cache.slice_view(start=q_lo, end=q_hi), offset=-q_lo,
                consume=_consume_src, owner=full_kv_cache if _consume_src else None)
        else:
            q_cache = self.kv_engine.apply_rotary_pos_emb(
                full_kv_cache.slice(start=q_lo, end=q_hi), offset=-q_lo, consume=True)
        buckets["input"][message] = [q_cache]
        buckets["input_ids"][message] = [self.kv_engine.trim_token_ids(old_ids, old_drop)]
        buckets["input_drop_num"][message] = [0]
        tags[message] = ["REAL"]

    def update_input_anchor(
        self,
        *,
        request_uid: str,
        agent_id: str,
        message: str,
        user_content: str,
        prefix_text: str,
        role: str = "user",
        include_begin: bool = True,
        include_eot: bool = False,
        anchor_namespace: str = "user_question",
        test_time: bool = False,
    ) -> str:
        """Ensure the user input placeholder cache is ready and choose a strategy."""
        state = self.get_request_state(request_uid)
        shared_mem = LLMChat._shared_kv_cache_memory
        agent_memory = self._ensure_agent_memory(agent_id)
        placeholder_info = agent_memory.get("placeholder_info")
        safe_message = _escape_loguru_markup(message)

        if message in shared_mem.get("input", {}):
            # Reuse only once a real harvest has landed.
            _tags = (shared_mem.get("input_tags") or {}).get(message) or ["BLANK"]
            if _tags[-1] != "REAL":
                return "dense_prefill"
            # KVCMAS: harvest exists, so behave as before -- the anchor bookkeeping below decides
            # dense vs reuse for this hop.
            if float(getattr(self.config, "threshold", 1.0) or 0.0) <= 0.0:
                logger.opt(colors=True).debug(
                    f"<yellow>threshold<=0: forced dense_prefill for '{safe_message}'.</yellow>"
                )
                return "dense_prefill"
            if not placeholder_info:
                logger.opt(colors=True).warning(
                    f"<yellow>No placeholder info found for agent '{agent_id}' while reusing input cache.</yellow>"
                )
                return "kv_reuse"
            placeholder_entries = list(placeholder_info.items())[::-1]

            _ph_order = [pid for pid, _ in placeholder_entries]
            for _i, (ph_id, _) in enumerate(placeholder_entries):
                safe_ph_id = _escape_loguru_markup(ph_id)
                bucket = state.anchor_dict.setdefault(ph_id, {})
                if bucket.get(message):
                    _has = (f'{self.node_id}_ph_key_delta'
                            in state.anchors.get(ph_id, {}).get(message, {}))
                    logger.opt(colors=True).info(
                        "<magenta>[GATE-VETO]</magenta> {}",
                        json.dumps({"agent_id": str(self.node_id),
                                    "decider_ph": str(ph_id),
                                    "decider_index": _i,
                                    "n_placeholders": len(_ph_order),
                                    "verdict": "kv_reuse" if _has else "dense_prefill",
                                    "order": [str(x) for x in _ph_order]}),
                    )

                    if f'{self.node_id}_ph_key_delta' in state.anchors.get(ph_id, {}).get(message, {}):
                        logger.opt(colors=True).debug(
                            f"<green>The message has repeatedly received for message '{safe_message}' at placeholder '{safe_ph_id}'. So we will reuse the KV cache.</green>"
                        )
                        return "kv_reuse"
                    logger.opt(colors=True).debug(
                        f"<yellow>Existing Anchor for message '{safe_message}' at placeholder '{safe_ph_id}'.</yellow>"
                    )
                    return "dense_prefill"
            logger.opt(colors=True).debug(
                f"<green>Reusing KV caches for message '{safe_message}' in all placeholders</green>."
            )
            return "kv_reuse"

        # No standalone encode.
        self._fs_install_blank(message, role=role, user_content=user_content,
                               prefix_text=prefix_text, include_begin=include_begin,
                               include_eot=include_eot)
        logger.opt(colors=True).debug(
            f"<green>KVCMAS: no encode; "
            f"hop 0 runs dense and harvests '{safe_message}'</green>."
        )
        return "dense_prefill"

        token_ids = self.tokenize_segment(
            role=role,
            content=user_content,
            include_begin=include_begin,
            include_eot=include_eot,
            add_special_tokens=False,
        )
        if "position_ids" in token_ids:
            position_ids = token_ids["position_ids"]
        else:
            position_ids = torch.arange(token_ids["input_ids"].shape[-1], dtype=torch.long)
        token_ids["position_ids"] = position_ids.unsqueeze(0).to(self.model.device)
        token_ids["input_ids"] = token_ids["input_ids"].to(self.model.device)
        token_ids["attention_mask"] = token_ids["attention_mask"].to(self.model.device)

        prefix_ids = self.tokenize_segment(
            role=role,
            content=prefix_text,
            include_begin=include_begin,
            include_eot=include_eot,
            add_special_tokens=False,
        )["input_ids"]
        drop_num = prefix_ids.shape[-1]
        if test_time:
            for _ in range(10):
                if _ == 5:
                    torch.cuda.synchronize()
                    start_time = perf_counter()
                output = self.model.generate(
                    **self._merge_vlm_inputs(token_ids),
                    use_cache=True,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    max_length=token_ids["input_ids"].shape[-1] + 1,
                    return_dict_in_generate=True,
                    return_legacy_cache=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            torch.cuda.synchronize()
            end_time = perf_counter()
            logger.opt(colors=True).info(
                f"<cyan>Latency for computing the input kv-cache of {message}: {(end_time - start_time) / 5:.3f} seconds</cyan>"
            )
        else:
            # [MEASUREMENT] The standalone input encode: a forward over the question text alone,
            # whose only product is the canonical question KV the consumers correct against.
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            _enc_t0 = perf_counter()
            output = self.model.generate(
                **self._merge_vlm_inputs(token_ids),
                use_cache=True,
                do_sample=False,
                temperature=None,
                top_p=None,
                max_length=token_ids["input_ids"].shape[-1] + 1,
                return_dict_in_generate=True,
                return_legacy_cache=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self._pending_encode_latency = (
                float(getattr(self, "_pending_encode_latency", 0.0)) + (perf_counter() - _enc_t0)
            )
        input_cache = output.past_key_values

        global_buckets = self._ensure_global_input_buckets()
        # Only the latest entry per message is ever read (fetch_shared_cache uses [-1]); store a
        # single-element list instead of appending so the ~8 GB input.
        global_buckets["input"][message] = [
            input_cache.slice(start=0, end=token_ids["input_ids"].shape[-1])
        ]
        global_buckets["input_ids"][message] = [token_ids]
        global_buckets["input_drop_num"][message] = [drop_num]

        anchor_store = state.anchors.setdefault(anchor_namespace, {})
        input_anchor_list = list(anchor_store.values())
        uq_len_bucket = state.anchor_len_dict.setdefault(anchor_namespace, {})
        anchor_len_list = [
            uq_len_bucket.get(entry_key, [0, 0])
            for entry_key in anchor_store.keys()
        ]
        uq_info_bucket = state.anchor_info_dict.setdefault(anchor_namespace, {})
        anchor_activated_list = list(uq_info_bucket.values())

        accumulate_len = 0
        for bucket in state.anchor_len_dict.values():
            accumulate_len += bucket.get(message, [0, 0])[0]

        prob, anchor_activated_list = self.kv_engine.predict_as_anchor(
            input_cache.slice_view(start=drop_num),  # read-only: view, not an 8 GB clone
            anchor_kv_cache_list=input_anchor_list,
            anchor_len_list=anchor_len_list,
            anchor_activated_list=anchor_activated_list,
            test_time=test_time,
        )
        logger.opt(colors=True).debug(
            f"<magenta>Anchor prediction for input '{safe_message}'</magenta>: {prob}"
        )

        state.anchor_dict.setdefault(anchor_namespace, {})[message] = prob
        global_bucket = state.global_anchor_info.setdefault(anchor_namespace, {})
        if not prob:
            info_items = list(uq_info_bucket.items())
            for idx, (msg_key, _) in enumerate(info_items):
                uq_info_bucket[msg_key] = anchor_activated_list[idx]
                bucket_entry = global_bucket.setdefault(msg_key, [0, 0])
                bucket_entry[0] = anchor_activated_list[idx]
            return "kv_reuse"

        uq_info_bucket[message] = 0
        global_bucket[message] = [
            0,
            input_cache.get_seq_length() - drop_num,
        ]
        return "dense_prefill"

    async def generate_for_agent(
        self,
        *,
        request_uid: str,
        message: str,
        preferred_mode: Optional[str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        min_tokens: Optional[int] = None,
        agent_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        agent_role: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> GenerationResult:
        """Generate a response using the requested strategy with sensible fallbacks."""
        latency_target = output_dir or kwargs.get("output_dir")
        # Per-agent overrides for localizing WHERE reuse damage enters.
        _an = agent_name or ""
        def _agent_listed(var: str) -> bool:
            v = os.environ.get(var, "").strip()
            return bool(v) and any(t and t in _an for t in v.split(","))
        _force_dense = _agent_listed("KVCMAS_DENSE_AGENTS")
        _rotate_only = _agent_listed("KVCMAS_ROTATE_ONLY_AGENTS")
        if preferred_mode == "dense_prefill" or _force_dense:
            mode = "dense_prefill"
        elif self.has_active_anchor(request_uid, message):
            mode = "dense_prefill"
        else:
            mode = "kv_reuse"

        if mode == "dense_prefill":
            return await self.generate_with_dense_prefill(
                message,
                max_tokens=max_tokens,
                temperature=temperature,
                min_tokens=min_tokens,
                max_anchor_num=kwargs.get("max_anchor_num", self.config.max_anchor_num),
                window_length=kwargs.get("window_length", self.config.window_size),
                request_uid=request_uid,
                agent_id=agent_id,
                agent_name=agent_name,
                agent_role=agent_role,
                output_dir=latency_target,
                **kwargs,
            )
        if not _rotate_only:
            return await self.generate_with_kv_reuse(
                message, max_tokens=max_tokens, temperature=temperature, min_tokens=min_tokens,
                request_uid=request_uid, agent_id=agent_id, agent_name=agent_name,
                agent_role=agent_role, output_dir=latency_target, **kwargs,
            )
        # Rotate-only for THIS agent: widen the placeholder skip list so every span it
        # reuses falls back to rotation, then restore -- other agents are unaffected.
        _prev = os.environ.get("KVCMAS_PH_DELTA_SKIP")
        os.environ["KVCMAS_PH_DELTA_SKIP"] = ",".join(filter(None, [_prev, "user_question", "_current"]))
        try:
            return await self.generate_with_kv_reuse(
                message, max_tokens=max_tokens, temperature=temperature, min_tokens=min_tokens,
                request_uid=request_uid, agent_id=agent_id, agent_name=agent_name,
                agent_role=agent_role, output_dir=latency_target, **kwargs,
            )
        finally:
            if _prev is None:
                os.environ.pop("KVCMAS_PH_DELTA_SKIP", None)
            else:
                os.environ["KVCMAS_PH_DELTA_SKIP"] = _prev

    def _map_in_pool(self, fn, iterable, timeout=None):
        pool = LLMChat._THREAD_POOL
        if pool is None:
            raise RuntimeError("Thread pool not initialized")
        task_timeout = timeout or self.config.worker_timeout
        futures = [pool.submit(fn, *args) for args in iterable]
        for fut in as_completed(futures, timeout=task_timeout):
            try:
                yield fut.result(timeout=self.config.worker_timeout)
            except TimeoutError as exc:
                raise TimeoutError("Thread task timeout") from exc
            except Exception as exc:
                raise RuntimeError("Thread task failed") from exc

    def set_id(self, node_id: str, role: str):
        """Bind the chat instance to a graph node id and role."""
        self.node_id = node_id
        self.role = role

        if self.node_id not in LLMChat._shared_kv_cache_memory:
            self._shared_kv_cache_memory[self.node_id] = LLMChat._shared_kv_cache_memory[self.node_id] = {}
            self._initialization[self.node_id] = LLMChat._initialization[self.node_id] = False

    async def prepare_prefix_kv_segments(self, node_id: str, prefix: str, user_prompt: str):
        """Materialize and store prefix KV segments and placeholder indices. """
        messages = self._render_base_messages(prefix, user_prompt)
        _, prompt_text, _ = self._build_chat_inputs(messages, add_generation_prompt=True)
        placeholder_info, token_ids, segments = self.locate_placeholder(prompt_text, return_segments=True)
        
        with torch.no_grad():
            out = self.model.generate(
                **self._merge_vlm_inputs(token_ids),
                use_cache=True,
                do_sample=False,
                temperature=None,
                top_p=None,
                max_length=token_ids['input_ids'].shape[-1] + 1,
                return_dict_in_generate=True,
                return_legacy_cache=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        base_kv = out.past_key_values.slice_(start=0, end=token_ids['input_ids'].shape[-1])                                               
        segment_kv_list = []
        token_id_list = []
        for type_, _, token_id, s, e in segments:

            if type_ == "text":
                seg_kv = base_kv.slice(start=s, end=e)
                segment_kv_list.append(seg_kv)
                token_id_list.append(token_id)
        self._shared_kv_cache_memory[node_id]["prefix"] = LLMChat._shared_kv_cache_memory[node_id]["prefix"] = segment_kv_list          
        self._shared_kv_cache_memory[node_id]["placeholder_info"] = LLMChat._shared_kv_cache_memory[node_id]["placeholder_info"] = placeholder_info
        self._shared_kv_cache_memory[node_id]["token_ids"] = LLMChat._shared_kv_cache_memory[node_id]["token_ids"] = token_id_list            

        self._initialization[node_id] = LLMChat._initialization[node_id] = True

    def _merge_vlm_inputs(self, inputs: Dict[str, Any], reuse: bool = False) -> Dict[str, Any]:
        """Attach pixel inputs iff this prompt actually carries <image> token ids. """
        if LLMChat._vlm_current_image is None:
            return inputs
        if reuse:
            # kv_reuse: the image ids sit INSIDE the cache-covered prefix; generate forwards only
            # the tail, which has no image tokens -- attaching pixel inputs.
            return inputs
        img = LLMChat._vlm_current_image
        tid = img.get("token_id", LLMChat._vlm_image_token_id)
        ids = inputs.get("input_ids")
        if ids is None or tid is None or not (ids == tid).any():
            return inputs
        n_ids = int((ids == tid).sum())
        if n_ids != int(img["n_tokens"]):
            raise RuntimeError(
                f"VLM merge: {n_ids} media ids in prompt but current media expands to "
                f"{img['n_tokens']} -- runner and prompt disagree (stale media slot?)")
        out = dict(inputs)
        for k, v in img["outputs"].items():
            if hasattr(v, "to"):
                out[k] = v.to(self.model.device, self.model.dtype) if v.is_floating_point() else v.to(self.model.device)
            else:
                out[k] = v
        return out

    @classmethod
    def set_current_media(cls, outputs, token_id: int, n_tokens: int) -> None:
        """Generalized B=1 slot (Video-MME P2): `outputs` is the processor's output dict minus input_ids/attention_mask (images: pixel_values+image_sizes."""
        cls._vlm_current_image = (None if outputs is None else
                                  dict(outputs=dict(outputs), token_id=int(token_id),
                                       n_tokens=int(n_tokens)))

    @classmethod
    def set_current_image(cls, pixel_values, image_sizes, n_tokens: int) -> None:
        """Runner-side: install the B=1 current image (None to clear)."""
        cls.set_current_media(
            None if pixel_values is None else
            dict(pixel_values=pixel_values, image_sizes=image_sizes),
            token_id=cls._vlm_image_token_id if cls._vlm_image_token_id is not None else 151646,
            n_tokens=n_tokens or 0)

    def _initialize_shared_resources(self):
        """Lazy-load shared tokenizer/model and shared KV memory storage."""
        with LLMChat._model_lock:
            if LLMChat._shared_model is None and _is_vlm_name(self.model_name):
                # VLM path: the wrapper owns generate() and the vision merge;
                # every KV/rotation/cache mechanism sees only the inner Qwen2ForCausalLM.
                from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
                LLMChat._shared_processor = AutoProcessor.from_pretrained(self.model_name)
                LLMChat._shared_tokenizer = LLMChat._shared_processor.tokenizer
                # The text loader gives every non-Llama checkpoint bfloat16, and OneVision's LM is
                # Qwen2ForCausalLM -- the same class that path deliberately keeps out.
                _vlm_dtype = getattr(torch, os.environ.get("KVCMAS_DTYPE", "float16"))
                LLMChat._shared_model = LlavaOnevisionForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=_vlm_dtype,
                    low_cpu_mem_usage=True,
                    device_map="cuda:0",
                    attn_implementation=os.environ.get("KVCMAS_ATTN_IMPL", "flash_attention_2"),
                )
                LLMChat._vlm_image_token_id = LLMChat._shared_tokenizer.convert_tokens_to_ids("<image>")
                # The wrapper's composite config hides the LM attributes every engine site reads
                # (num_hidden_layers etc.
                _lm = LLMChat._shared_model.language_model
                if not hasattr(LLMChat._shared_model, "model"):
                    LLMChat._shared_model.model = _lm.model
                if not hasattr(LLMChat._shared_model, "lm_head"):
                    LLMChat._shared_model.lm_head = _lm.lm_head
                _tc = LLMChat._shared_model.config.text_config
                for _attr in ("num_hidden_layers", "num_attention_heads",
                              "num_key_value_heads", "hidden_size", "head_dim",
                              "max_position_embeddings", "rope_theta", "vocab_size"):
                    if not hasattr(LLMChat._shared_model.config, _attr) and hasattr(_tc, _attr):
                        setattr(LLMChat._shared_model.config, _attr, getattr(_tc, _attr))
                if LLMChat._shared_tokenizer.pad_token_id is None:
                    LLMChat._shared_tokenizer.pad_token_id = LLMChat._shared_tokenizer.eos_token_id
                    LLMChat._shared_model.config.pad_token_id = LLMChat._shared_tokenizer.eos_token_id
                logger.info("VLM {} loaded (LM: {}).", self.model_name,
                            type(LLMChat._shared_model.language_model).__name__)
            if LLMChat._shared_model is None:
                LLMChat._shared_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                # Flash-Attention-2 requires fp16/bf16 (it rejects fp32), so non-Llama checkpoints
                # (e.g.
                LLMChat._shared_model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if 'llama' in self.model_name else torch.bfloat16,
                    low_cpu_mem_usage=True,
                    device_map="cuda:0",
                    attn_implementation=os.environ.get("KVCMAS_ATTN_IMPL", "flash_attention_2"),
                    trust_remote_code=True,
                )
                if LLMChat._shared_tokenizer.pad_token_id is None:
                    LLMChat._shared_tokenizer.pad_token_id = LLMChat._shared_tokenizer.eos_token_id
                    LLMChat._shared_model.config.pad_token_id = LLMChat._shared_tokenizer.eos_token_id
                logger.info("Model {} loaded and shared across instances.", self.model_name)
            if LLMChat._shared_kv_cache_memory is None:
                LLMChat._shared_kv_cache_memory = {}

    def locate_placeholder(self, original_text, return_segments=False):
        """Locate placeholder token spans in a templated prompt. """

        placeholder_pattern = r'\{((?:agent|condition)_\w+_(?:current|history)|user_question)\}'

        matches = list(re.finditer(placeholder_pattern, original_text))

        last_pos = 0
        segments = []
        placeholder_info = {}
        token_num = 0
        idx_count = 0
        for m in matches:
            start, end = m.span()
            placeholder_inner = m.group(1)
            if last_pos < start:
                txt = original_text[last_pos:start]
                token_id = self.tokenizer(txt, add_special_tokens=False)['input_ids']
                encoding = {}
                encoding['input_ids'] = torch.tensor(token_id).unsqueeze(0).to(self.model.device)
                encoding['attention_mask'] = torch.ones_like(encoding['input_ids']).to(self.model.device)
                if txt.strip():
                    segments.append(("text", txt, encoding, token_num, token_num + len(token_id)))
                    idx_count += 1
                token_num += len(token_id)
            token_id = self.tokenizer(f'{ {placeholder_inner}} ', add_special_tokens=False)['input_ids']
            encoding = {}
            encoding['input_ids'] = torch.tensor(token_id).unsqueeze(0).to(self.model.device)
            encoding['attention_mask'] = torch.ones_like(encoding['input_ids']).to(self.model.device)
            segments.append(("placeholder", placeholder_inner, encoding, token_num, token_num + len(token_id)))
            placeholder_info[placeholder_inner] = [token_num, token_num + len(token_id)]
            token_num += len(token_id)
            idx_count += 1
            last_pos = end

        txt = original_text[last_pos:]
        token_id = self.tokenizer(txt, add_special_tokens=False)['input_ids']
        encoding = {}
        encoding['input_ids'] = torch.tensor(token_id).unsqueeze(0).to(self.model.device)
        encoding['attention_mask'] = torch.ones_like(encoding['input_ids']).to(self.model.device)
        if txt.strip():
            segments.append(("text", txt, encoding, token_num, token_num + len(token_id)))
            token_num += len(token_id)

        segments.sort(key=lambda x: x[-1])
        token_ids = torch.cat([sublist[2]['input_ids'] for sublist in segments], dim=1)
        encoding = {}
        encoding['input_ids'] = token_ids
        encoding['attention_mask'] = torch.ones_like(encoding['input_ids']).to(self.model.device)

        placeholder_info = dict(sorted(placeholder_info.items(), key=lambda x: x[1][0], reverse=True))
        if return_segments:
            return placeholder_info, encoding, segments
        return placeholder_info, encoding

    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Union[List[str], str]:
        pass

    @retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(3))
    async def agen(
        self,
        messages: List[Message] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        return_cache: Optional[bool] = False,
        *,
        min_tokens: Optional[int] = None,
        request_uid: Optional[str] = None,
        agent_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        agent_role: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> GenerationResult:
        async with self.lock:
            if max_tokens is None:
                max_tokens = self.DEFAULT_MAX_TOKENS
            if temperature is None:
                temperature = self.DEFAULT_TEMPERATURE
            # NONSHARED "KV holding" baseline: this agent's context changed, so evict its
            # previously-held KV BEFORE re-prefilling.
            _hold_kv = os.environ.get("KVCMAS_HOLD_KV", "0").strip().lower() in ("1", "true", "yes", "on")
            if _hold_kv:
                _old_kv = LLMChat._held_kv_cache.pop(self.node_id, None)
                del _old_kv
            inputs, prompt_text, prompt_length = self._build_chat_inputs(messages)
            safe_prompt_text = _escape_loguru_markup(prompt_text)
            logger.opt(colors=True).debug(
                "<blue>[PROMPT]</blue> Agent {} Role {} Prompt:\n{}",
                self.node_id,
                self.role,
                safe_prompt_text,
            )
            generation_kwargs = {
                "do_sample": False,
                "temperature": None,
                "top_p": None,
                "max_new_tokens": max_tokens,
                "return_dict_in_generate": True,
                "return_legacy_cache": False,
                "use_cache": True,
                "pad_token_id": self.tokenizer.pad_token_id,
            }
            if min_tokens:
                generation_kwargs["min_new_tokens"] = min_tokens
            ttft_tracer = _TTFTTracer(prompt_length)
            generation_kwargs["stopping_criteria"] = StoppingCriteriaList([ttft_tracer])
            ttft_tracer.reset(prompt_length)
            if prealloc_enabled():
                # [FAIR-IMPL] Uniform decode cache policy.
                generation_kwargs["past_key_values"] = PreallocCache(
                    capacity=int(prompt_length) + int(max_tokens) + 8
                )
            outputs = self.model.generate(**self._merge_vlm_inputs(inputs), **generation_kwargs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            # Hold this agent's full KV cache resident (per node_id) so all N agents' caches
            # coexist — the true no-sharing peak.
            if _hold_kv:
                LLMChat._held_kv_cache[self.node_id] = outputs.past_key_values
            generation_e2e_latency = perf_counter() - ttft_tracer.start_time
            if ttft_tracer.ttft is None:
                ttft_value = 0.0
            else:
                ttft_value = ttft_tracer.ttft
            generated_token_count = outputs.sequences.shape[-1] - prompt_length
            generated_sequence = outputs.sequences[:, prompt_length:]
            response_message = self.tokenizer.decode(
                generated_sequence[0], skip_special_tokens=True
            ).strip()
            safe_response_message = _escape_loguru_markup(response_message)
            logger.opt(colors=True).debug(
                "<blue>[RESPONSE]</blue> Agent {} Role {} Response:\n{}",
                self.node_id,
                self.role,
                safe_response_message,
            )
            metadata: Dict[str, Any] = {}
            if request_uid:
                metadata["request_uid"] = request_uid
            if agent_id:
                metadata["agent_id"] = agent_id
            if agent_name:
                metadata["agent_name"] = agent_name
            if agent_role:
                metadata["agent_role"] = agent_role
            if return_cache:
                metadata["kv_cache"] = outputs.past_key_values
            if output_dir is not None:
                latency_record = {
                    "timestamp": time.time(),
                    "mode": "default",
                    "ttft": float(ttft_value),
                    "e2e_latency": float(generation_e2e_latency),
                    "input_tokens": int(prompt_length),
                    "output_tokens": int(generated_token_count),
                    # Dense path: nothing is reused, by construction. Emitted so both
                    # paths share one schema and the trace extractor needs no branch.
                    "shared_tokens": 0,
                    "reusable_tokens": 0,
                    "reused_tokens": 0,
                    "recomputed_tokens": 0,
                    "is_source_hop": False,
                    "request_uid": request_uid,
                    "agent_id": agent_id,
                    "agent_name": agent_name,
                    "agent_role": agent_role,
                }
                _append_latency_record(output_dir, latency_record)
            return GenerationResult(
                text=response_message,
                mode="default",
                ttft=ttft_value,
                e2e_latency=generation_e2e_latency,
                metadata=metadata,
            )

    @retry(wait=wait_random_exponential(max=1000), stop=stop_after_attempt(1))
    async def generate_with_dense_prefill(
        self,
        messages: List[Message] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        min_tokens: Optional[int] = None,
        max_anchor_num: Optional[int] = 20,
        window_length: Optional[int] = 5,
        request_uid: Optional[str] = None,
        agent_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        agent_role: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> GenerationResult:
        """Generate with dense prefix prefill and optional anchor update."""
        return await self.agen_kvcomm(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            min_tokens=min_tokens,
            request_uid=request_uid,
            mode="dense_prefill",
            max_anchor_num=max_anchor_num,
            window_length=window_length,
            agent_id=agent_id,
            agent_name=agent_name,
            agent_role=agent_role,
            output_dir=output_dir
        )

    @retry(wait=wait_random_exponential(max=1000), stop=stop_after_attempt(1))
    async def generate_with_kv_reuse(
        self,
        messages: List[Message] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        min_tokens: Optional[int] = None,
        request_uid: Optional[str] = None,
        agent_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        agent_role: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> GenerationResult:
        """Generate by reusing existing prefix KV (fast path)."""
        test_time = kwargs.get("test_time", False)
        if test_time:
            return await self.agen_kvcomm_time_test(
                messages=messages,
                max_tokens=max_tokens,
                min_tokens=min_tokens if min_tokens is not None else max_tokens,
                temperature=temperature,
                request_uid=request_uid,
                mode="kv_reuse",
                agent_id=agent_id,
                agent_name=agent_name,
                agent_role=agent_role,
                output_dir=output_dir,
            )
        return await self.agen_kvcomm(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            min_tokens=min_tokens,
            request_uid=request_uid,
            mode="kv_reuse",
            agent_id=agent_id,
            agent_name=agent_name,
            agent_role=agent_role,
            output_dir=output_dir,
        )

    async def agen_kvcomm(
        self,
        messages: List[Message] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        min_tokens: Optional[int] = None,
        request_uid: Optional[str] = None,
        mode: str = "dense_prefill",
        max_anchor_num: int = 20,
        window_length: int = 5,
        agent_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        agent_role: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> GenerationResult:
        """Core KV-aware generation entry. """
        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if request_uid is None:
            raise ValueError("request_uid must be provided for agen_kvcomm.")
        state = self.kv_engine.resolve_request_state(request_uid)
        preprocess_start = perf_counter() if mode == "kv_reuse" else None

        if isinstance(messages, List):
            message = messages[0]
        else:
            message = messages

        # Snapshot BEFORE the harvest at the end of this call flips the tag to REAL: did the
        # shared question span have a real source when THIS call assembled.
        _q_had_source = (
            ((self._shared_kv_cache_memory.get("input_tags") or {}).get(message) or ["REAL"])[-1] == "REAL"
        )

        prefix_store = self._shared_kv_cache_memory[self.node_id]
        prefix_kv_list: List[DynamicCache] = prefix_store.get("prefix", [])
        prefix_token_ids: List[Dict[str, torch.Tensor]] = prefix_store.get("token_ids", [])
        placeholder_info_map = prefix_store.get("placeholder_info")
        if not prefix_kv_list:
            raise RuntimeError(
                "No prefix KV found in shared memory. Make sure you've called prepare_prefix_kv_segments or init_shared_placeholder_prefix_kv."
            )
        if placeholder_info_map is None:
            raise RuntimeError("placeholder_info missing in shared KV cache memory.")

        # kv_reuse builds the base as a list of segment descriptors that REFERENCE
        # prefix_kv_list (no copy / no merged cache); dense doesn't use a base cache.
        merged_prefix_token_ids = prefix_token_ids[0].copy()

        placeholder_entries = list(placeholder_info_map.items())[::-1]

        meta: List[Dict[str, Any]] = []
        ph_id_list: List[str] = []
        cum_offset = 0
        ph_cum_len = 0
        for idx, ((ph_id, (start, end)), pf_kv, pf_token_id) in enumerate(
            zip(placeholder_entries, prefix_kv_list[1:], prefix_token_ids[1:])
        ):
            ph_cache, ph_cache_ids, drop_num = self.kv_engine.fetch_shared_cache(ph_id, message)
            real_len = ph_cache._seen_tokens - drop_num
            templ_len = end - start
            delta_len = real_len - templ_len
            meta.append(
                {
                    "idx": idx,
                    "ph_id": ph_id,
                    "start": start,
                    "end": end,
                    "drop_num": drop_num,
                    "delta": delta_len,
                    "offset_before": cum_offset,
                    "offset_after": cum_offset + delta_len,
                    "ph_cache": ph_cache,
                    # Captured now so the base tensors can be released before the harvest:
                    # `_token_reuse_metadata` is the only later consumer and it needs the length.
                    "_span_len": int(ph_cache._seen_tokens) - int(drop_num),
                    "ph_cache_ids": ph_cache_ids,
                    "pf_kv": pf_kv,
                    "pf_ids": pf_token_id,
                    "cum_len": ph_cum_len,
                }
            )
            cum_offset += delta_len
            ph_cum_len += real_len
            ph_id_list.append(ph_id)

        placeholder_indices: Dict[str, Tuple[int, int]] = {}
        for m in meta:
            start = m["start"] + m["offset_before"]
            placeholder_indices[m["ph_id"]] = (
                start,
                start + m["ph_cache"]._seen_tokens - m["drop_num"],
            )

        base_segments: List[Dict[str, Any]] = []

        if mode == "dense_prefill":
            # Stage-2: token-only assembly — the rotated/merged base cache is NEVER built.
            tasks = [(message, m) for m in meta]
            results_sorted = sorted(
                self._map_in_pool(self.kv_engine.process_anchor_tokens, tasks, timeout=30),
                key=lambda x: x[0],
            )
            seg_ids_list = [r[1] for r in results_sorted]
            merged_prefix_token_ids = concat_(merged_prefix_token_ids, seg_ids_list)
            del results_sorted, seg_ids_list
        elif mode == "kv_reuse":
            anchors_for_node = state.anchors
            tasks = [
                (request_uid, message, m, list(anchors_for_node.get(m["ph_id"], {}).values()))
                for m in meta
            ]
            results_sorted = sorted(
                self._map_in_pool(self.kv_engine.update_kv_cache_segment, tasks, timeout=30),
                key=lambda x: x[0],
            )
            # Build the base as segment descriptors that REFERENCE the shared base
            # (prefix_kv_list[0] + the raw placeholder/prefix segments) — no copy, no merged.
            base_segments = [dict(
                key_src=prefix_kv_list[0].key_cache, value_src=prefix_kv_list[0].value_cache,
                drop=0, out_len=int(prefix_kv_list[0].key_cache[0].shape[-2]),
                cos=None, sin=None, key_field=None, value_field=None,
                anchors=None, anchor_index=None, w_key=None, w_value=None,
            )]
            for r in results_sorted:
                base_segments.extend(r[1])  # [ph_desc, pf_desc]
            seg_ids_list = [r[2] for r in results_sorted]
            merged_prefix_token_ids = concat_(merged_prefix_token_ids, seg_ids_list)
            del results_sorted, seg_ids_list
        else:
            raise ValueError(f"Unsupported mode '{mode}' for agen_kvcomm.")

        if mode == "dense_prefill":
            prefix_token_length = merged_prefix_token_ids["input_ids"].shape[-1]
        else:
            prefix_token_length = sum(int(s["out_len"]) for s in base_segments)
        input_length = merged_prefix_token_ids["input_ids"].shape[-1]
        if input_length != prefix_token_length:
            logger.warning(
                "prefix_token_length: {} merged_length: {}",
                prefix_token_length,
                input_length,
            )
            raise RuntimeError("prefix_token_length != merged_prefix_token_ids['input_ids'].shape[-1]")

        if "position_ids" in merged_prefix_token_ids:
            merged_prefix_token_ids["position_ids"] = (
                torch.arange(input_length).unsqueeze(0).to(self.model.device)
            )

        generation_kwargs: Dict[str, Any] = {
            "max_length": max_tokens + prefix_token_length,
            "do_sample": False,
            "temperature": None,
            "top_p": None,
            "return_legacy_cache": False,
            "return_dict_in_generate": True,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if min_tokens:
            generation_kwargs["min_new_tokens"] = min_tokens

        if mode == "kv_reuse":
            # Drop the last base token (generate re-processes it) by clipping the last segment,
            # then wrap the segment list as a reference-based cache: rotated +.
            base_segments[-1] = {**base_segments[-1], "out_len": base_segments[-1]["out_len"] - 1}
            merged_prefix_kv = StreamingCorrectionCache.from_segments(base_segments)
            # Chaining forces MATERIALISE over streaming decode: hop i+1's delta is taken against
            # a real cache, and decode must not re-assemble a layer on every step.
            if _lkv_chain_delta() and prealloc_enabled():
                # Decode reads the whole reused span at every step, so a StreamingCorrectionCache
                # would re-assemble all layers per step -- the dominant cost.
                _src = merged_prefix_kv
                # Instrumented separately from delta_recon_materialize (which lives INSIDE
                # materialize and only covers the assemble loop).
                _consume = (_lkv_chain_delta()
                            and os.environ.get("KVCMAS_CONSUME_BASE", "1").strip().lower()
                            not in ("0", "false", "no", "off"))
                merged_prefix_kv = to_prealloc(_src.materialize(consume_base=_consume),
                                               extra_tokens=int(max_tokens))
                del _src
            generation_kwargs["past_key_values"] = merged_prefix_kv
        elif mode == "dense_prefill":
            # Peak (warmup): park the raw base segments on CPU during the re-prefill; they are
            # only needed AFTER, for the streamed delta.
            _park_out = False
            for m in meta:
                if _park_out or _is_question_span(m["ph_id"]):
                    m["ph_cache"] = m["ph_cache"].to("cpu")
                m["pf_kv"] = m["pf_kv"].to("cpu")

        ttft_tracer = _TTFTTracer(prefix_token_length)
        generation_kwargs["stopping_criteria"] = StoppingCriteriaList([ttft_tracer])
        ttft_tracer.reset(prefix_token_length)
        preprocess_latency = 0.0
        _charged = False
        if preprocess_start is not None:
            preprocess_latency = max(0.0, perf_counter() - preprocess_start)
            _charged = True
        # Fold in the standalone input encode measured in update_input_anchor.
        _enc_pending = float(getattr(self, "_pending_encode_latency", 0.0))
        if _enc_pending > 0.0:
            preprocess_latency += _enc_pending
            self._pending_encode_latency = 0.0
            _charged = True
        # The last un-instrumented region on the reuse hop.
        outputs = self.model.generate(**self._merge_vlm_inputs(merged_prefix_token_ids, reuse=(mode == "kv_reuse")), **generation_kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        generation_e2e_latency = perf_counter() - ttft_tracer.start_time
        if ttft_tracer.ttft is None:
            generation_ttft = 0.0
        else:
            generation_ttft = ttft_tracer.ttft
        ttft_value = generation_ttft
        if _charged:
            ttft_value += preprocess_latency

        full_kv_cache = outputs.past_key_values
        generated_sequences = outputs.sequences
        del outputs  # release the generate object's hold on the KV cache so del full frees it
        # Peak-memory (warmup): pull the short response cache out first so the large generate
        # cache can be freed before the delta phase.
        response_kv_cache = full_kv_cache.slice(start=prefix_token_length)
        anchor_create_latency = 0.0

        if mode == "dense_prefill":
            # Stage-2 single-use: stream the per-placeholder delta from full_kv_cache (real)
            # against the RAW base segments.
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            _anc_t0 = perf_counter()
            self.kv_engine.set_anchor_streaming_raw(
                request_uid,
                message,
                ph_id_list,
                full_kv_cache,
                meta,
                placeholder_indices,
                max_anchor_num=max_anchor_num,
                window_length=window_length,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            anchor_create_latency = perf_counter() - _anc_t0
            # The harvest runs ALONGSIDE set_anchor_streaming_raw above rather than instead of it
            # -- the anchors still get built, and the harvested span becomes.
            self._fs_harvest_question(message, full_kv_cache, placeholder_indices)
            del full_kv_cache
            # Real cache freed — bring the base back to GPU for the STEADY consume (done here in
            # warmup, off the measured STEADY path.
            _chained = _lkv_chain_delta()
            _park_out = False
            for m in meta:
                _is_q = _is_question_span(m["ph_id"])
                if _chained and _is_q:
                    m["ph_cache"] = None
                elif _park_out or _is_q:
                    m["ph_cache"] = m["ph_cache"].to(self.model.device)
                m["pf_kv"] = m["pf_kv"].to(self.model.device)
        else:
            # CHAIN, steady state: this hop reused, so it produced its corrected cache without
            # ever running set_anchor -- and the next hop's delta must be taken.
            if _lkv_chain_delta():
                # Release the OLD base FIRST.
                for _m in meta:
                    _m["ph_cache"] = None
                self._fs_harvest_question(message, full_kv_cache, placeholder_indices)
            del full_kv_cache  # kv_reuse: response already extracted above
        response_kv_cache = self.kv_engine.apply_rotary_pos_emb(
            response_kv_cache,
            offset=-prefix_token_length,
        )

        mem = LLMChat._shared_kv_cache_memory[self.node_id]
        resp = mem.setdefault("response", {})
        resp_ids = mem.setdefault("response_ids", {})
        resp_drop = mem.setdefault("response_drop_num", {})

        seq = generated_sequences
        response_tokens = seq[:, prefix_token_length:-1]
        attn_len = response_tokens.size(1)
        response_mask = torch.ones(seq.size(0), attn_len, device=self.model.device)

        current_key = f"agent_{self.node_id}_current"
        anchor_bucket = state.anchors.setdefault(current_key, {})
        anchor_len_bucket = state.anchor_len_dict.setdefault(current_key, {})
        anchor_info_bucket = state.anchor_info_dict.setdefault(current_key, {})
        response_anchor_list = list(anchor_bucket.values())
        anchor_len_list = [
            anchor_len_bucket.get(kk, [0, 0])
            for kk in anchor_bucket.keys()
        ]
        anchor_active_list: List[int] = list(anchor_info_bucket.values())

        resp.setdefault(message, []).append(response_kv_cache)
        resp_ids.setdefault(message, []).append(
            {
                "input_ids": response_tokens,
                "attention_mask": response_mask,
            }
        )
        resp_drop.setdefault(message, []).append(0)

        accumulate_len = 0
        for key in state.anchor_len_dict.keys():
            bucket = state.anchor_len_dict.get(key, {})
            length_entry = bucket.get(message, [0, 0])
            accumulate_len += length_entry[0]

        prob, anchor_active_list = self.kv_engine.predict_as_anchor(
            response_kv_cache,
            anchor_kv_cache_list=response_anchor_list,
            anchor_len_list=anchor_len_list,
            anchor_activated_list=anchor_active_list,
        )
        safe_message = _escape_loguru_markup(message)
        logger.opt(colors=True).debug(
            f"<magenta>Agent {self.node_id} Role {self.role} Message {safe_message} Response Anchor Prediction: {prob}</magenta>",
        )
        state.anchor_dict.setdefault(current_key, {})[message] = prob

        if not prob:
            global_bucket = state.global_anchor_info.setdefault(current_key, {})
            info_items = list(anchor_info_bucket.items())
            for idx, (msg_key, _) in enumerate(info_items):
                anchor_info_bucket[msg_key] = anchor_active_list[idx]
                bucket_entry = global_bucket.setdefault(msg_key, [0, 0])
                bucket_entry[0] = anchor_active_list[idx]

        response_message = self.tokenizer.decode(
            generated_sequences[0, prefix_token_length:],
            skip_special_tokens=True,
        )
        prompt_preview = self.tokenizer.decode(
            merged_prefix_token_ids["input_ids"][0]
        )
        safe_prompt_preview = _escape_loguru_markup(prompt_preview)
        safe_response_message = _escape_loguru_markup(response_message)
        logger.opt(colors=True).debug(
            "<blue>[PROMPT:{mode}]</blue> Agent {} Role {} Prompt:\n{}",
            self.node_id,
            self.role,
            safe_prompt_preview,
            mode=mode,
        )
        logger.opt(colors=True).debug(
            "<blue>[RESPONSE:{mode}]</blue> Agent {} Role {} Response:\n{}",
            self.node_id,
            self.role,
            safe_response_message,
            mode=mode,
        )

        metadata: Dict[str, Any] = {
            "placeholder_ids": ph_id_list,
            # Exact per-placeholder span lengths, so prefill can be split into (media/question |
            # observations | system+glue) by addition instead of a subtraction.
            "placeholder_spans": {str(_m["ph_id"]): int(_m.get("_span_len") or 0) for _m in meta},
        }
        metadata.update(_token_reuse_metadata(meta, mode, q_has_source=_q_had_source))
        if _charged:
            metadata["preprocess_latency"] = preprocess_latency
            metadata["generation_ttft"] = generation_ttft
            metadata["input_encode_latency"] = float(_enc_pending)
        if request_uid:
            metadata["request_uid"] = request_uid
        if agent_id:
            metadata["agent_id"] = agent_id
        if agent_name:
            metadata["agent_name"] = agent_name
        if agent_role:
            metadata["agent_role"] = agent_role
        generated_token_count = generated_sequences.shape[-1] - prefix_token_length
        # e2e = [preprocess: matching + delta materialize (reuse hops)] + generation
        #       + [anchor construction: SVD (dense hops)].  TTFT carries only the first two.
        e2e_latency = (generation_e2e_latency + (preprocess_latency if _charged else 0.0)
                       + anchor_create_latency)
        metadata["anchor_create_latency"] = float(anchor_create_latency)
        latency_record = {
            "timestamp": time.time(),
            "mode": mode,
            "ttft": float(ttft_value),
            "generation_ttft": float(generation_ttft),
            "preprocess_latency": float(preprocess_latency) if _charged else None,
            "input_encode_latency": float(_enc_pending),
            "phases": None,
            "e2e_latency": float(e2e_latency),
            "anchor_create_latency": float(anchor_create_latency),
            "input_tokens": int(prefix_token_length),
            "output_tokens": int(generated_token_count),
            "request_uid": request_uid,
            "agent_id": agent_id,
            "agent_name": agent_name,
            "agent_role": agent_role,
            "message": str(message) if message is not None else None,
            "placeholder_ids": ph_id_list,
            # Exact per-placeholder span lengths, so prefill can be split into (media/question |
            # observations | system+glue) by addition instead of a subtraction.
            "placeholder_spans": {str(_m["ph_id"]): int(_m.get("_span_len") or 0) for _m in meta},
            # Per-HOP reuse partition.
            "shared_tokens": int(metadata.get("shared_tokens", 0)),
            "reusable_tokens": int(metadata.get("reusable_tokens", 0)),
            "reused_tokens": int(metadata.get("reused_tokens", 0)),
            "recomputed_tokens": int(metadata.get("recomputed_tokens", 0)),
            "is_source_hop": bool(metadata.get("is_source_hop", False)),
        }
        _append_latency_record(output_dir, latency_record)
        return GenerationResult(
            text=response_message,
            mode=mode,
            ttft=ttft_value,
            e2e_latency=e2e_latency,
            metadata=metadata,
        )

    async def agen_kvcomm_time_test(
        self,
        messages: List[Message] = None,
        max_tokens: Optional[int] = None,
        min_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        request_uid: Optional[str] = None,
        mode: str = "dense_prefill",
        max_anchor_num: int = 20,
        window_length: int = 5,
        agent_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        agent_role: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> GenerationResult:
        """Core KV-aware generation entry. """
        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if request_uid is None:
            raise ValueError("request_uid must be provided for agen_kvcomm.")
        min_tokens = max_tokens if min_tokens is None else min_tokens
        state = self.kv_engine.resolve_request_state(request_uid)
        preprocess_start = perf_counter() if mode == "kv_reuse" else None

        if isinstance(messages, List):
            message = messages[0]
        else:
            message = messages

        prefix_store = self._shared_kv_cache_memory[self.node_id]
        prefix_kv_list: List[DynamicCache] = prefix_store.get("prefix", [])
        prefix_token_ids: List[Dict[str, torch.Tensor]] = prefix_store.get("token_ids", [])
        placeholder_info_map = prefix_store.get("placeholder_info")
        if not prefix_kv_list:
            raise RuntimeError(
                "No prefix KV found in shared memory. Make sure you've called prepare_prefix_kv_segments or init_shared_placeholder_prefix_kv."
            )
        if placeholder_info_map is None:
            raise RuntimeError("placeholder_info missing in shared KV cache memory.")

        # kv_reuse builds the base as a list of segment descriptors that REFERENCE
        # prefix_kv_list (no copy / no merged cache); dense doesn't use a base cache.
        merged_prefix_token_ids = prefix_token_ids[0].copy()

        placeholder_entries = list(placeholder_info_map.items())[::-1]

        meta: List[Dict[str, Any]] = []
        ph_id_list: List[str] = []
        cum_offset = 0
        ph_cum_len = 0
        for idx, ((ph_id, (start, end)), pf_kv, pf_token_id) in enumerate(
            zip(placeholder_entries, prefix_kv_list[1:], prefix_token_ids[1:])
        ):
            ph_cache, ph_cache_ids, drop_num = self.kv_engine.fetch_shared_cache(ph_id, message)
            real_len = ph_cache._seen_tokens - drop_num
            templ_len = end - start
            delta_len = real_len - templ_len
            meta.append(
                {
                    "idx": idx,
                    "ph_id": ph_id,
                    "start": start,
                    "end": end,
                    "drop_num": drop_num,
                    "delta": delta_len,
                    "offset_before": cum_offset,
                    "offset_after": cum_offset + delta_len,
                    "ph_cache": ph_cache,
                    # Captured now so the base tensors can be released before the harvest:
                    # `_token_reuse_metadata` is the only later consumer and it needs the length.
                    "_span_len": int(ph_cache._seen_tokens) - int(drop_num),
                    "ph_cache_ids": ph_cache_ids,
                    "pf_kv": pf_kv,
                    "pf_ids": pf_token_id,
                    "cum_len": ph_cum_len,
                }
            )
            cum_offset += delta_len
            ph_cum_len += real_len
            ph_id_list.append(ph_id)

        if mode == "dense_prefill":
            tasks = [(message, m) for m in meta]
            results = list(
                self._map_in_pool(self.kv_engine.process_anchor, tasks, timeout=30)
            )
        elif mode == "kv_reuse":
            anchors_for_node = state.anchors
            tasks = [
                (
                    request_uid,
                    message,
                    m,
                    list(anchors_for_node.get(m["ph_id"], {}).values()),
                )
                for m in meta
            ]
            results = list(self._map_in_pool(self.kv_engine.update_kv_cache_segment, tasks, timeout=30))
        else:
            raise ValueError(f"Unsupported mode '{mode}' for agen_kvcomm.")

        results_sorted = sorted(results, key=lambda x: x[0])

        placeholder_indices: Dict[str, Tuple[int, int]] = {}
        for m in meta:
            start = m["start"] + m["offset_before"]
            placeholder_indices[m["ph_id"]] = (
                start,
                start + m["ph_cache"]._seen_tokens - m["drop_num"],
            )

        base_segments: List[Dict[str, Any]] = []
        if mode == "kv_reuse":
            # Reference-based base segments (no copy / no merged cache) — the
            # StreamingCorrectionCache rotates + corrects + assembles per layer.
            base_segments = [dict(
                key_src=prefix_kv_list[0].key_cache, value_src=prefix_kv_list[0].value_cache,
                drop=0, out_len=int(prefix_kv_list[0].key_cache[0].shape[-2]),
                cos=None, sin=None, key_field=None, value_field=None,
                anchors=None, anchor_index=None, w_key=None, w_value=None,
            )]
            for r in results_sorted:
                base_segments.extend(r[1])  # [ph_desc, pf_desc]
        seg_ids_list = [r[2] for r in results_sorted]
        merged_prefix_token_ids = concat_(merged_prefix_token_ids, seg_ids_list)
        del results, results_sorted, seg_ids_list
        if mode == "dense_prefill":
            # Free the standalone per-placeholder base caches: they were only the source for
            # merged_prefix_kv (the rotated base the delta uses).
            for _m in meta:
                _m["ph_cache"] = None
                _m["pf_kv"] = None

        if mode == "dense_prefill":
            prefix_token_length = merged_prefix_token_ids["input_ids"].shape[-1]
        else:
            prefix_token_length = sum(int(s["out_len"]) for s in base_segments)
        input_length = merged_prefix_token_ids["input_ids"].shape[-1]
        if input_length != prefix_token_length:
            logger.warning(
                "prefix_token_length: {} merged_length: {}",
                prefix_token_length,
                input_length,
            )
            raise RuntimeError("prefix_token_length != merged_prefix_token_ids['input_ids'].shape[-1]")

        if "position_ids" in merged_prefix_token_ids:
            merged_prefix_token_ids["position_ids"] = (
                torch.arange(input_length).unsqueeze(0).to(self.model.device)
            )

        generation_kwargs: Dict[str, Any] = {
            "max_length": max_tokens + prefix_token_length,
            "min_new_tokens": min_tokens,
            "do_sample": False,
            "temperature": None,
            "top_p": None,
            "return_legacy_cache": False,
            "return_dict_in_generate": True,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

        if mode == "kv_reuse":
            base_segments[-1] = {**base_segments[-1], "out_len": base_segments[-1]["out_len"] - 1}
            merged_prefix_kv = StreamingCorrectionCache.from_segments(base_segments)
            # Chaining forces MATERIALISE over streaming decode: hop i+1's delta is taken against
            # a real cache, and decode must not re-assemble a layer on every step.
            if _lkv_chain_delta() and prealloc_enabled():
                # Decode reads the whole reused span at every step, so a StreamingCorrectionCache
                # would re-assemble all layers per step -- the dominant cost.
                _src = merged_prefix_kv
                # Instrumented separately from delta_recon_materialize (which lives INSIDE
                # materialize and only covers the assemble loop).
                _consume = (_lkv_chain_delta()
                            and os.environ.get("KVCMAS_CONSUME_BASE", "1").strip().lower()
                            not in ("0", "false", "no", "off"))
                merged_prefix_kv = to_prealloc(_src.materialize(consume_base=_consume),
                                               extra_tokens=int(max_tokens))
                del _src
            generation_kwargs["past_key_values"] = merged_prefix_kv

        preprocess_latency = 0.0
        if preprocess_start is not None:
            torch.cuda.synchronize()
            preprocess_latency = max(0.0, perf_counter() - preprocess_start)
        torch.cuda.synchronize()
        ttft_tracer = _TTFTTracer(prefix_token_length)
        generation_kwargs["stopping_criteria"] = StoppingCriteriaList([ttft_tracer])
        ttft_tracer.reset(prefix_token_length)
        # The last un-instrumented region on the reuse hop.
        outputs = self.model.generate(**self._merge_vlm_inputs(merged_prefix_token_ids, reuse=(mode == "kv_reuse")), **generation_kwargs)
        torch.cuda.synchronize()
        if mode == "kv_reuse" and preprocess_start is not None:
            kvcomm_end_to_end_latency = perf_counter() - ttft_tracer.start_time
            kvcomm_ttft_value = ttft_tracer.ttft + preprocess_latency
            logger.opt(colors=True).info(
                f"<green>Agent {self.node_id} Role {self.role} Message {_escape_loguru_markup(message)} KVCMAS E2E Latency: {kvcomm_end_to_end_latency:.4f}s TTFT: {kvcomm_ttft_value:.4f}s (Preprocess: {preprocess_latency:.4f}s)</green>",
            )
        full_kv_cache = outputs.past_key_values

        generation_kwargs.pop("past_key_values", None)
        torch.cuda.synchronize()
        ttft_tracer = _TTFTTracer(prefix_token_length)
        generation_kwargs["stopping_criteria"] = StoppingCriteriaList([ttft_tracer])
        ttft_tracer.reset(prefix_token_length)
        _ = self.model.generate(**self._merge_vlm_inputs(merged_prefix_token_ids, reuse=(mode == "kv_reuse")), **generation_kwargs)
        torch.cuda.synchronize()
        dense_end_to_end_latency = perf_counter() - ttft_tracer.start_time
        dense_prefill_ttft = ttft_tracer.ttft
        logger.opt(colors=True).info(
            f"<cyan>Agent {self.node_id} Role {self.role} Message {_escape_loguru_markup(message)} Dense Prefill E2E Latency: {dense_end_to_end_latency:.4f}s TTFT: {dense_prefill_ttft:.4f}s</cyan>",
        )
        if mode == "kv_reuse" and preprocess_start is not None and kvcomm_ttft_value > 0:
            logger.opt(colors=True).info(
                f"<green>Agent {self.node_id} Role {self.role} Message {_escape_loguru_markup(message)} KVCMAS is {dense_prefill_ttft / kvcomm_ttft_value:.2f}x faster than Dense Prefill in TTFT</green>",
            )
            ttft_value = kvcomm_ttft_value
        else:
            ttft_value = dense_prefill_ttft
        response_kv_cache = full_kv_cache.slice_(start=prefix_token_length)
        response_kv_cache = self.kv_engine.apply_rotary_pos_emb(
            response_kv_cache,
            offset=-prefix_token_length,
        )

        mem = LLMChat._shared_kv_cache_memory[self.node_id]
        resp = mem.setdefault("response", {})
        resp_ids = mem.setdefault("response_ids", {})
        resp_drop = mem.setdefault("response_drop_num", {})

        seq = outputs.sequences
        response_tokens = seq[:, prefix_token_length:-1]
        attn_len = response_tokens.size(1)
        response_mask = torch.ones(seq.size(0), attn_len, device=self.model.device)

        current_key = f"agent_{self.node_id}_current"
        anchor_bucket = state.anchors.setdefault(current_key, {})
        anchor_len_bucket = state.anchor_len_dict.setdefault(current_key, {})
        anchor_info_bucket = state.anchor_info_dict.setdefault(current_key, {})
        response_anchor_list = list(anchor_bucket.values())
        anchor_len_list = [
            anchor_len_bucket.get(kk, [0, 0])
            for kk in anchor_bucket.keys()
        ]
        anchor_active_list: List[int] = list(anchor_info_bucket.values())

        resp.setdefault(message, []).append(response_kv_cache)
        resp_ids.setdefault(message, []).append(
            {
                "input_ids": response_tokens,
                "attention_mask": response_mask,
            }
        )
        resp_drop.setdefault(message, []).append(0)

        accumulate_len = 0
        for key in state.anchor_len_dict.keys():
            bucket = state.anchor_len_dict.get(key, {})
            length_entry = bucket.get(message, [0, 0])
            accumulate_len += length_entry[0]

        prob, anchor_active_list = self.kv_engine.predict_as_anchor(
            response_kv_cache,
            anchor_kv_cache_list=response_anchor_list,
            anchor_len_list=anchor_len_list,
            anchor_activated_list=anchor_active_list,
            test_time=True,
        )
        safe_message = _escape_loguru_markup(message)
        logger.opt(colors=True).debug(
            f"<magenta>Agent {self.node_id} Role {self.role} Message {safe_message} Response Anchor Prediction: {prob}</magenta>",
        )
        state.anchor_dict.setdefault(current_key, {})[message] = prob

        if not prob:
            global_bucket = state.global_anchor_info.setdefault(current_key, {})
            info_items = list(anchor_info_bucket.items())
            for idx, (msg_key, _) in enumerate(info_items):
                anchor_info_bucket[msg_key] = anchor_active_list[idx]
                bucket_entry = global_bucket.setdefault(msg_key, [0, 0])
                bucket_entry[0] = anchor_active_list[idx]

        response_message = self.tokenizer.decode(
            outputs.sequences[0, prefix_token_length:],
            skip_special_tokens=True,
        )
        prompt_preview = self.tokenizer.decode(
            merged_prefix_token_ids["input_ids"][0]
        )
        safe_prompt_preview = _escape_loguru_markup(prompt_preview)
        safe_response_message = _escape_loguru_markup(response_message)
        logger.opt(colors=True).debug(
            "<blue>[PROMPT:{mode}]</blue> Agent {} Role {} Prompt:\n{}",
            self.node_id,
            self.role,
            safe_prompt_preview,
            mode=mode,
        )
        logger.opt(colors=True).debug(
            "<blue>[RESPONSE:{mode}]</blue> Agent {} Role {} Response:\n{}",
            self.node_id,
            self.role,
            safe_response_message,
            mode=mode,
        )

        metadata: Dict[str, Any] = {
            "placeholder_ids": ph_id_list,
            # Exact per-placeholder span lengths, so prefill can be split into (media/question |
            # observations | system+glue) by addition instead of a subtraction.
            "placeholder_spans": {str(_m["ph_id"]): int(_m.get("_span_len") or 0) for _m in meta},
        }
        if preprocess_start is not None:
            metadata["preprocess_latency"] = preprocess_latency
            metadata["generation_ttft"] = ttft_value - preprocess_latency
        if request_uid:
            metadata["request_uid"] = request_uid
        if agent_id:
            metadata["agent_id"] = agent_id
        if agent_name:
            metadata["agent_name"] = agent_name
        if agent_role:
            metadata["agent_role"] = agent_role
        generated_token_count = outputs.sequences.shape[-1] - prefix_token_length
        if mode == "kv_reuse":
            latency_record = {
                "timestamp": time.time(),
                "mode": mode,
                "ttft": float(ttft_value),
                "generation_ttft": float(metadata["generation_ttft"]) if "generation_ttft" in metadata else None,
                "preprocess_latency": float(preprocess_latency) if preprocess_start is not None else None,
                "dense_prefill_ttft": float(dense_prefill_ttft),
                "kvcomm_end_to_end_latency": float(kvcomm_end_to_end_latency),
                "dense_end_to_end_latency": float(dense_end_to_end_latency),
                "ttft_ratio_dense_over_kvcomm": float(dense_prefill_ttft / ttft_value) if ttft_value > 0 else None,
                "input_tokens": int(prefix_token_length),
                "output_tokens": int(generated_token_count),
                "request_uid": request_uid,
                "agent_id": agent_id,
                "agent_name": agent_name,
                "agent_role": agent_role,
                "message": str(message) if message is not None else None,
                "placeholder_ids": ph_id_list,
                "placeholder_spans": {str(_m["ph_id"]): int(_m.get("_span_len") or 0) for _m in meta},
            # Exact per-placeholder span lengths, so prefill can be split into (media/question |
            # observations | system+glue) by addition instead of a subtraction.
            "placeholder_spans": {str(_m["ph_id"]): int(_m.get("_span_len") or 0) for _m in meta},
            }
        else:
            latency_record = {
                "timestamp": time.time(),
                "mode": mode,
                "ttft": float(ttft_value),
                "generation_ttft": float(metadata["generation_ttft"]) if "generation_ttft" in metadata else None,
                "preprocess_latency": float(preprocess_latency) if preprocess_start is not None else None,
                "dense_prefill_ttft": float(dense_prefill_ttft),
                "dense_end_to_end_latency": float(dense_end_to_end_latency),
                "input_tokens": int(prefix_token_length),
                "output_tokens": int(generated_token_count),
                "request_uid": request_uid,
                "agent_id": agent_id,
                "agent_name": agent_name,
                "agent_role": agent_role,
                "message": str(message) if message is not None else None,
                "placeholder_ids": ph_id_list,
                "placeholder_spans": {str(_m["ph_id"]): int(_m.get("_span_len") or 0) for _m in meta},
            # Exact per-placeholder span lengths, so prefill can be split into (media/question |
            # observations | system+glue) by addition instead of a subtraction.
            "placeholder_spans": {str(_m["ph_id"]): int(_m.get("_span_len") or 0) for _m in meta},
            }
        _append_latency_record(output_dir, latency_record)
        tt = kvcomm_end_to_end_latency if mode == "kv_reuse" and preprocess_start is not None else dense_end_to_end_latency
        return GenerationResult(
            text=response_message,
            mode=mode,
            ttft=ttft_value,
            e2e_latency=tt,
            metadata=metadata,
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['model']
        del state['tokenizer']
        del state['lock']
        del state['_shared_kv_cache_memory']
        del state['_initialization']
        return state

    def __setstate__(self, state):

        self.__dict__.update(state)
        self.tokenizer = LLMChat._shared_tokenizer
        self.model = LLMChat._shared_model
        self._shared_kv_cache_memory = LLMChat._shared_kv_cache_memory
        self._initialization = LLMChat._initialization
        self.lock = asyncio.Lock()

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        state = self.__getstate__()
        copied_state = copy.deepcopy(state, memo)
        node_id = copied_state.get('node_id', None)
        role = copied_state.get('role', None)
        if node_id is not None:
            if node_id in LLMChat._shared_kv_cache_memory:
                original_cache = LLMChat._shared_kv_cache_memory[node_id]
                LLMChat._shared_kv_cache_memory[node_id] = {
                    "prefix": original_cache.get("prefix"),
                    "placeholder_info": original_cache.get("placeholder_info"),
                    "token_ids": original_cache.get("token_ids"),
                    "input": {},
                    "response": {},
                    "response_ids": {},
                    "condition": {},
                    "condition_ids": {},
                    "input_drop_num": {},
                    "response_drop_num": {},
                    "condition_drop_num": {},
                }
                LLMChat.weight_dict = {}
            result.set_id(node_id, role)
        result.__setstate__(copied_state)
        return result
