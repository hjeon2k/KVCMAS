import os
from typing import List,Any,Dict
import re
import asyncio

import torch
from KVCMAS.graph.node import Node
from KVCMAS.agents.agent_registry import AgentRegistry
from KVCMAS.llm.llm_registry import LLMRegistry
from KVCMAS.llm.gpt_chat import LLMChat
from KVCMAS.prompt.prompt_set_registry import PromptSetRegistry
from KVCMAS.llm.config import KVCommConfig
from time import perf_counter
from KVCMAS.utils.log import logger

import random as _random
import zlib

GENERATED_TEXT = int(os.getenv('GENERATED_TEXT', '128'))
RETRIEVED_TEXT = int(os.getenv('RETRIEVED_TEXT', '0'))
INTER_AGENT_TEXT = int(os.getenv('INTER_AGENT_TEXT', '0'))

# Segment layout of the CopyMachine efficiency trajectory: synthetic padding, so each
# axis moves exactly one segment.
_SYMBOLS = ("Δ", "Ω")          # Delta / Omega: exactly one token each


def _pad(n_tokens: int, salt: int = 0) -> str:
    """Deterministic filler of exactly ``n_tokens`` tokens (empty when n <= 0)."""
    if n_tokens <= 0:
        return ""
    rng = _random.Random(salt)
    return " ".join(rng.choice(_SYMBOLS) for _ in range(n_tokens))


_OVERHEAD_CACHE: Dict[int, int] = {}


def _template_overhead(tokenizer) -> int:
    """Tokens the chat template inserts ahead of the user content. """
    key = id(tokenizer)
    if key in _OVERHEAD_CACHE:
        return _OVERHEAD_CACHE[key]
    overhead = 0
    try:
        def enc(user):
            return tokenizer.apply_chat_template(
                [{"role": "system", "content": ""}, {"role": "user", "content": user}],
                tokenize=True, add_generation_prompt=True,
            )
        a, b = enc(_SYMBOLS[0]), enc(_SYMBOLS[1])
        n = 0
        for x, y in zip(a, b):
            if x != y:
                break
            n += 1
        overhead = n
    except Exception:
        overhead = 0                      # unknown: do not compensate, just report below
    _OVERHEAD_CACHE[key] = overhead
    return overhead


def _system_prompt_body(tokenizer) -> str:
    """pf0 body sized so (template overhead + body) == SYSTEM_PROMPT tokens exactly."""
    target = int(os.getenv("SYSTEM_PROMPT", "1024"))
    body_len = max(0, target - _template_overhead(tokenizer))
    return _pad(body_len, salt=1_000_003)


@AgentRegistry.register('CopyMachine')
class CopyMachine(Node):
    def __init__(
        self,
        id: str | None = None,
        role: str = None,
        domain: str = "",
        llm_name: str = "",
        llm_config: KVCommConfig | None = None,
    ):
        super().__init__(id, "CopyMachine" ,domain, llm_name)
        prefix = ""

        self.llm = LLMRegistry.get(llm_name, prefix=prefix, llm_config=llm_config)
        self.prompt_set = PromptSetRegistry.get(domain)
        self.role = self.prompt_set.get_role() if role is None else role
        self.llm.set_id(self.id, self.role)
        self.constraint = self.prompt_set.get_analyze_constraint(self.role)

    async def _process_inputs(
        self,
        raw_inputs:Dict[str,str],
        spatial_info:Dict[str,Dict],
        temporal_info:Dict[str,Dict],
        mode: str = "allow_kv_reuse",
        **kwargs,
    ) -> Dict[str, Any]:
        """Prepare prompts, optionally populate anchors, and return mode hints."""
        if mode == "allow_kv_reuse":
            request_uid = raw_inputs.get("_request_uid") or kwargs.get("request_uid")
            if request_uid is None:
                raise ValueError("request_uid is required for request-scoped anchor updates.")

            preferred_mode = "kv_reuse"
            agent_memory = self.llm._ensure_agent_memory(self.id)

            if self.llm.has_prefix_initialized(self.id) and "placeholder_info" in agent_memory:
                
                # No lead-in: ph0 is the task alone in the declared layout, so the encode must
                # cover the task alone too.
                prefix_text = kwargs.get('prefix', "")
                user_content = prefix_text + raw_inputs['task']
                preferred_mode = self.llm.update_input_anchor(
                    request_uid=request_uid,
                    agent_id=self.id,
                    message=raw_inputs['task'],
                    user_content=user_content,
                    prefix_text=prefix_text,
                )
                logger.opt(colors=True).info(
                    "<green>[MODE]</green> Agent {} ({}) mode: {}",
                    self.id,
                    self.role,
                    preferred_mode,
                )
                return {"preferred_mode": preferred_mode, "early_response": None}

            # Declared layout, template form: the placeholders are what every method uses
            # to locate spans, so they must appear verbatim and in order.
            system_prompt = _system_prompt_body(self.llm.tokenizer)
            user_prompt = self._compose(
                "{user_question}",
                [(aid, info['output'] if len(info['output']) > 0
                       else "{agent_" + aid + "_current}")
                 for aid, info in spatial_info.items()],
                [(aid, info['output'] if len(info['output']) > 0
                       else "{agent_" + aid + "_history}")
                 for aid, info in temporal_info.items()],
            )
            await self.llm.prepare_prefix_kv_segments(self.id, system_prompt, user_prompt)
            return {"preferred_mode": preferred_mode, "early_response": None}


        # Same declared layout, concrete form (NonShared / dense).
        system_prompt = _system_prompt_body(self.llm.tokenizer)
        user_prompt = self._compose(
            raw_inputs['task'],
            [(aid, info['output']) for aid, info in spatial_info.items()],
            [(aid, info['output']) for aid, info in temporal_info.items()],
        )
        return {"system_prompt": system_prompt, "user_prompt": user_prompt}

    def _private_context(self) -> str:
        """`priv`: this agent's retrieved context. Frozen per agent, never shared."""
        if RETRIEVED_TEXT <= 0:
            return ""
        return _pad(RETRIEVED_TEXT, salt=zlib.crc32(str(self.id).encode()))

    def _compose(self, question: str, spatial: List[Any], temporal: List[Any]) -> str:
        """ph0, priv, then (pf_i glue, ph_i peer output) pairs -- no prose, nothing undeclared. """
        parts = [question, self._private_context()]
        for i, (_aid, out) in enumerate(list(spatial) + list(temporal)):
            glue = _pad(INTER_AGENT_TEXT, salt=100 + i)
            if glue:
                parts.append(glue)
            parts.append(out)
        return " ".join(p for p in parts if p)

    def _execute(self, input:Dict[str,str],  spatial_info:Dict[str,Dict], temporal_info:Dict[str,Dict],**kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """

        inputs = asyncio.run(
            self._process_inputs(
                input,
                spatial_info,
                temporal_info,
                mode="default",
                **kwargs,
            )
        )
        system_prompt = inputs["system_prompt"]
        user_prompt = inputs["user_prompt"]
        message = [{'role':'system','content':system_prompt},{'role':'user','content':user_prompt}]
        response = self.llm.gen(message)
        return response

    async def _async_execute(self, input:Dict[str,str],  spatial_info:Dict[str,Dict], temporal_info:Dict[str,Dict], mode: str = "default", **kwargs):
        """Handle asynchronous execution across different KV cache modes."""
        if mode == "default":
            request_uid = input.get("_request_uid")
            inputs = await self._process_inputs(
                input,
                spatial_info,
                temporal_info,
                mode=mode,
                **kwargs,
            )
            system_prompt = inputs["system_prompt"]
            user_prompt = inputs["user_prompt"]
            message = [{'role':'system','content':system_prompt},{'role':'user','content':user_prompt}]
            result = await self.llm.agen(
                message,
                max_tokens=GENERATED_TEXT,
                min_tokens=GENERATED_TEXT,
                request_uid=request_uid,
                agent_id=self.id,
                agent_name=self.agent_name,
                agent_role=self.role,
                output_dir=kwargs.get("output_dir"),
            )
            return result

        assert mode == "allow_kv_reuse", f"Unsupported async execution mode: {mode}"
        request_uid = input.get("_request_uid") or kwargs.get("request_uid")
        if request_uid is None:
            raise ValueError("request_uid is required for request-scoped anchor updates.")

        mode_data = await self._process_inputs(
            input,
            spatial_info,
            temporal_info,
            mode=mode,
            **kwargs,
        )
        preferred_mode = mode_data["preferred_mode"]
        result = await self.llm.generate_for_agent(
            request_uid=request_uid,
            message=input['task'],
            preferred_mode=preferred_mode,
            output_dir=kwargs.get("output_dir"),
            agent_id=self.id,
            agent_name=self.agent_name,
            agent_role=self.role,
            max_tokens=GENERATED_TEXT,
            min_tokens=GENERATED_TEXT,
        )
        return input['task'], result
