import os
from typing import Any,Dict
import re
import asyncio

import torch
from KVCMAS.graph.node import Node
from KVCMAS.agents.agent_registry import AgentRegistry
from KVCMAS.llm.llm_registry import LLMRegistry
from KVCMAS.llm.gpt_chat import LLMChat
from KVCMAS.prompt.prompt_set_registry import PromptSetRegistry
from KVCMAS.tools.search.wiki import search_wiki_main
from KVCMAS.llm.config import KVCommConfig
from KVCMAS.utils.log import logger
WIKI_TOKEN_LENGTH = int(os.environ.get('WIKI_TOKEN_LENGTH', 1024))

def find_strings_between_pluses(text):
    return re.findall(r'@+([^@]+?)@+', text)

def _analyze_tail(prompt_set, role: str) -> str:
    """Per-domain instruction restated at the GENERATION point, optionally per ROLE. """
    fn = getattr(prompt_set, "get_analyze_tail", None)
    if fn is None:
        return ""
    try:
        return fn(role)
    except TypeError:
        return fn()


# Retrieval span: the Knowledgeable Expert emits @entities@, only the Wiki Searcher
# receives the lookup.
def _retrieval_pair(prompt_set):
    """(producer_role, consumer_role, prefix_text, max_tokens) for the retrieval stage."""
    fn = getattr(prompt_set, "get_retrieval_stage", None)
    if fn is None:
        return ("Knowledgeable Expert", "Wiki Searcher",
                "The key entities of the problem are explained in Wikipedia as follows:",
                WIKI_TOKEN_LENGTH)
    return fn()


def _is_retrieval_edge(prompt_set, my_role, peer_role) -> bool:
    stage = _retrieval_pair(prompt_set)
    if stage is None:          # the domain has no lookup stage (Video-MME answers from the video)
        return False
    producer, consumer, _, _ = stage
    return my_role == consumer and peer_role == producer


async def _retrieve(prompt_set, peer_output: str, task: str):
    """Run the domain's lookup for the queries the producer emitted."""
    queries = find_strings_between_pluses(peer_output)
    fn = getattr(prompt_set, "retrieve", None)
    if fn is not None:
        out = fn(queries, task)
        return list(out) if out else []
    try:
        return await search_wiki_main(queries)
    except Exception:
        return []


@AgentRegistry.register('AnalyzeAgent')
class AnalyzeAgent(Node):
    """Research/analysis agent that can search wiki and compose context."""
    def __init__(
        self,
        id: str | None = None,
        role: str = None,
        domain: str = "",
        llm_name: str = "",
        llm_config: KVCommConfig | None = None,
    ):
        super().__init__(id, "AnalyzeAgent" ,domain, llm_name)
        prefix = ""

        self.llm = LLMRegistry.get(llm_name, prefix=prefix, llm_config=llm_config)
        self.prompt_set = PromptSetRegistry.get(domain)
        self.role = self.prompt_set.get_role() if role is None else role
        self.llm.set_id(self.id, self.role)
        self.constraint = self.prompt_set.get_analyze_constraint(self.role)
        self.wiki_summary = ""

    async def _process_inputs(
        self,
        raw_inputs:Dict[str,str],
        spatial_info:Dict[str,Dict],
        temporal_info:Dict[str,Dict],
        mode: str = "default",
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
                for agent_id, info in spatial_info.items():
                    if _is_retrieval_edge(self.prompt_set, self.role, info['role']):
                        _, _, _rprefix, _rmax = _retrieval_pair(self.prompt_set)
                        source_memory = self.llm._ensure_agent_memory(agent_id)
                        if raw_inputs['task'] in source_memory.get('condition', {}):
                            continue
                        wiki = await _retrieve(self.prompt_set, info['output'], raw_inputs['task'])
                        if wiki:
                            summary = _rprefix + "\n".join(wiki)
                        else:
                            summary = _rprefix + "\nNo available information.\n"
                        self.wiki_summary = summary
                        self.llm.update_condition_anchor(
                            request_uid=request_uid,
                            owner_agent_id=agent_id,
                            message=raw_inputs['task'],
                            content=summary,
                            prefix_text=_rprefix + "\n",
                            max_length=_rmax,
                        )

                prefix_text = kwargs.get('prefix', "The task is:")
                user_content = prefix_text + raw_inputs['task']
                preferred_mode = self.llm.update_input_anchor(
                    request_uid=request_uid,
                    agent_id=self.id,
                    message=raw_inputs['task'],
                    user_content=user_content,
                    prefix_text=prefix_text,
                )
                logger.opt(colors=True).info(
                    "<green>[MODE]</green> Task: {} Agent {} ({}) mode: {}",
                    raw_inputs["task"],
                    self.id,
                    self.role,
                    preferred_mode,
                )
                return {"preferred_mode": preferred_mode, "early_response": None}

            system_prompt = f"{self.constraint}"
            user_input = "{user_question}"
            user_prompt = f"The task is: {user_input}\n"
            spatial_str = ""
            temporal_str = ""
            for agent_id, info in spatial_info.items():
                agent_role = info['role']
                agent_output = info['output'] if len(info['output']) > 0 else "{agent_" + agent_id + "_current}"
                if _is_retrieval_edge(self.prompt_set, self.role, agent_role):
                    condition_output = "{condition_" + agent_id + "_current}"
                    user_prompt += _retrieval_pair(self.prompt_set)[2] + f" {condition_output}"
                spatial_str += f"Agent {agent_id}, role is {agent_role}, output is:\n\n {agent_output}\n\n"
            for agent_id, info in temporal_info.items():
                agent_role = info['role']
                agent_output = info['output'] if len(info['output']) > 0 else "{agent_" + agent_id + "_history}"
                temporal_str += f"Agent {agent_id}, role is {agent_role}, output is:\n\n {agent_output}\n\n"

            if spatial_str:
                user_prompt += (
                    "At the same time, the outputs of other agents are as follows:\n\n"
                    f"{spatial_str} \n\n"
                )
            if temporal_str:
                user_prompt += (
                    "In the last round of dialogue, the outputs of other agents were: \n\n"
                    f"{temporal_str}"
                )
            # Optional per-domain instruction restated at the GENERATION point.
            user_prompt += _analyze_tail(self.prompt_set, self.role)
            await self.llm.prepare_prefix_kv_segments(self.id, system_prompt, user_prompt)
            return {"preferred_mode": preferred_mode, "early_response": None}


        system_prompt = f"{self.constraint}"
        user_prompt = f"The task is: {raw_inputs['task']}\n"
        spatial_str = ""
        temporal_str = ""
        for id, info in spatial_info.items():
            if _is_retrieval_edge(self.prompt_set, self.role, info['role']):
                _, _, _rprefix, _rmax = _retrieval_pair(self.prompt_set)
                wiki = await _retrieve(self.prompt_set, info['output'], raw_inputs['task'])
                if len(wiki):
                    self.wiki_summary = ".\n".join(wiki)
                    token_ids = self.llm.tokenizer(self.wiki_summary, return_tensors="pt", add_special_tokens=False)
                    token_ids = {k: v[:, :_rmax].to(self.llm.model.device) for k, v in token_ids.items() if isinstance(v, torch.Tensor)}
                    self.wiki_summary = self.llm.tokenizer.decode(token_ids['input_ids'][0], skip_special_tokens=True)
                    user_prompt += f"{_rprefix}{self.wiki_summary}"
            spatial_str += f"Agent {id}, role is {info['role']}, output is:\n\n {info['output']}\n\n"
        for id, info in temporal_info.items():
            temporal_str += f"Agent {id}, role is {info['role']}, output is:\n\n {info['output']}\n\n"

        user_prompt += f"At the same time, the outputs of other agents are as follows:\n\n{spatial_str} \n\n" if len(spatial_str) else ""
        user_prompt += f"In the last round of dialogue, the outputs of other agents were: \n\n{temporal_str}" if len(temporal_str) else ""
        user_prompt += _analyze_tail(self.prompt_set, self.role)   # see reuse path
        return {"system_prompt": system_prompt, "user_prompt": user_prompt}

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
                max_tokens=self.llm.DEFAULT_MAX_TOKENS,
                request_uid=request_uid,
                agent_id=self.id,
                agent_name=self.agent_name,
                agent_role=self.role,
                output_dir=kwargs.get("output_dir"),
            )
            if self.wiki_summary != "":
                self.wiki_summary = ""
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
        )
        if self.wiki_summary != "":
            self.wiki_summary = ""
        return input['task'], result
