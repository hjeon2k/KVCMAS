from abc import ABC, abstractmethod
from typing import List, Optional
import os
from KVCMAS.llm.format import Message
from KVCMAS.utils.metrics import GenerationResult


GENERATED_TEXT = int(os.getenv('GENERATED_TEXT', '0'))
class LLM(ABC):
    """Abstract base class for text-generating language models. """

    DEFAULT_MAX_TOKENS = 512
    DEFAULT_TEMPERATURE = 1.0
    DEFUALT_NUM_COMPLETIONS = 1
    DEFAULT_BANDWIDTH = 3072
    DEFAULT_MIN_TOKENS = GENERATED_TEXT

    @abstractmethod
    async def agen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        ) -> GenerationResult:
        """Asynchronously generate a completion. """
        pass

    @abstractmethod
    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        ) -> GenerationResult:
        """Synchronously generate a completion. """
        pass
