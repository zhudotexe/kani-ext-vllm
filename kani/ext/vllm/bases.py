import abc
import logging
import warnings

import torch
from kani import AIFunction, ChatMessage, PromptPipeline
from kani.engines import BaseEngine
from kani.utils.warnings import deprecated

log = logging.getLogger(__name__)


class VLLMBase(BaseEngine, abc.ABC):
    """Base class for vLLM engines with local tokenization/prompt building."""

    def __init__(
        self, tokenizer, max_context_size: int, prompt_pipeline: PromptPipeline[str | list[int] | torch.Tensor]
    ):
        self.tokenizer = tokenizer
        self.max_context_size = max_context_size
        self.pipeline = prompt_pipeline

        # deprecated
        self._token_reserve = 0

    def build_prompt(
        self, messages: list[ChatMessage], functions: list[AIFunction] | None = None
    ) -> str | list[int] | torch.Tensor:
        """
        Given the list of messages from kani, build either a single string representing the prompt for the model,
        or build the token tensor.

        The default behaviour is to call the supplied pipeline.
        """
        if self.pipeline is None:
            raise NotImplementedError(
                "You must pass a prompt_pipeline to the HuggingEngine to use it as a non-abstract class."
            )
        prompt = self.pipeline(messages, functions)
        log.debug(f"BUILT PROMPT: {prompt}")
        return prompt

    async def prompt_len(self, messages, functions=None, **kwargs) -> int:
        prompt = self.build_prompt(messages, functions)
        # tokenize it ourselves in order to capture special tokens correctly
        if isinstance(prompt, list):
            prompt_toks = prompt
        elif isinstance(prompt, torch.Tensor):
            prompt_toks = prompt[0].tolist()
        else:
            prompt_toks = self.tokenizer.encode(prompt, add_special_tokens=False)
        return len(prompt_toks)

    # ==== deprecated ====
    @property
    @deprecated("Use prompt_len instead")
    def token_reserve(self):
        # infer the token reserve from the pipeline
        if self._token_reserve == 0 and self.pipeline:
            self._token_reserve = self._infer_token_reserve()
        return self._token_reserve

    def _infer_token_reserve(self):
        """If token_reserve is not set and we have a pipeline, infer it."""
        prompt = self.pipeline.execute([], for_measurement=True)
        if isinstance(prompt, list):
            return len(prompt)
        if isinstance(prompt, torch.Tensor):
            return len(prompt[0])
        # prompt str to tokens
        tokenized = self.tokenizer.encode(prompt)
        return len(tokenized)

    @deprecated("Use prompt_len instead")
    def message_len(self, message: ChatMessage) -> int:
        # default concrete base behaviour:
        if self.pipeline is None:
            raise NotImplementedError(
                "You must pass a prompt_pipeline to the VLLMEngine to use it as a non-abstract class."
            )
        prompt = self.pipeline.execute([message], for_measurement=True)
        if isinstance(prompt, list):
            return len(prompt)
        if isinstance(prompt, torch.Tensor):
            return len(prompt[0])
        # prompt str to tokens
        tokenized = self.tokenizer.encode(prompt)
        return len(tokenized)

    @deprecated("Use prompt_len instead")
    def function_token_reserve(self, functions: list[AIFunction]) -> int:
        if not functions:
            return 0
        # default concrete base behaviour:
        if self.pipeline is None:
            raise NotImplementedError(
                "You must pass a prompt_pipeline to the VLLMEngine to use it as a non-abstract class."
            )
        prompt = self.pipeline.execute([], functions, for_measurement=True)
        if isinstance(prompt, list):
            toklen = len(prompt)
        elif isinstance(prompt, torch.Tensor):
            toklen = len(prompt[0])
        else:
            # prompt str to tokens
            tokenized = self.tokenizer.encode(prompt)
            toklen = len(tokenized)

        # warn if there are functions but no tokens
        if toklen == 0:
            warnings.warn(
                "Functions were given to the model, but the function prompt returned 0 tokens! This model may not"
                " support function calling, or you may need to implement"
                f" `{type(self).__name__}.function_token_reserve()`."
            )

        return toklen
