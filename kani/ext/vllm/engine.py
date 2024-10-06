import logging
import uuid
import warnings

import transformers
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams

from kani import AIFunction, ChatMessage, PromptPipeline
from kani.engines import BaseEngine, Completion
from kani.engines.huggingface.chat_template_pipeline import ChatTemplatePromptPipeline

log = logging.getLogger(__name__)


class VLLMEngine(BaseEngine):
    def __init__(
        self,
        model_id: str,
        max_context_size: int = None,
        prompt_pipeline: PromptPipeline[str] = None,
        *,
        model_load_kwargs: dict = None,
        **hyperparams,
    ):
        """
        :param model_id: The ID of the model to load from HuggingFace.
        :param max_context_size: The context size of the model. If not given, will be set from the model's config.
        :param prompt_pipeline: The pipeline to translate a list of kani ChatMessages into the model-specific chat
            format (see :class:`.PromptPipeline`).
        :param model_load_kwargs: Additional arguments to pass to ``AsyncEngineArgs()``.
        :param hyperparams: Additional arguments to supply the model during generation.
        """

        if model_load_kwargs is None:
            model_load_kwargs = {}

        engine_args = AsyncEngineArgs(model=model_id, max_model_len=max_context_size, **model_load_kwargs)
        engine = AsyncLLMEngine.from_engine_args(engine_args)

        self.model = engine
        self.tokenizer = engine.engine.get_tokenizer()
        self.max_context_size = max_context_size
        self.hyperparams = hyperparams

        # load the pipeline
        if prompt_pipeline is None:
            if isinstance(self.tokenizer, transformers.PreTrainedTokenizerBase):
                prompt_pipeline = ChatTemplatePromptPipeline(self.tokenizer)
            else:
                raise ValueError(
                    "There is no chat template associated with this model (tokenizer loaded from a non-HF source)."
                    " Please provide a prompt_pipeline."
                )
        self.pipeline = prompt_pipeline

        # token counting stuff
        # try and infer max context size from the model config if not specified
        if self.max_context_size is None:
            self.max_context_size = engine.engine.get_model_config().max_model_len

        log.debug(f"Inferred max context size: {self.max_context_size}")

        if self.max_context_size is None:
            raise ValueError(
                "Could not infer the model's max context size from the config. Please pass the `max_context_size` arg."
            )
        elif self.max_context_size > 1e20:
            warnings.warn(
                f"The inferred max context size of this model is extremely large ({self.max_context_size}). This"
                " may mean that the model has not configured their model_max_len correctly (or you are still using"
                " my code in 2050). Please pass the `max_context_size` arg to use the correct model size."
            )

        # infer the token reserve from the pipeline
        if self.token_reserve == 0 and self.pipeline:
            self.token_reserve = self._infer_token_reserve()

    def _infer_token_reserve(self):
        """If token_reserve is not set and we have a pipeline, infer it."""
        prompt = self.pipeline.execute([], for_measurement=True)
        # prompt str to tokens
        tokenized = self.tokenizer.encode(prompt)
        return len(tokenized)

    def message_len(self, message: ChatMessage) -> int:
        # default concrete base behaviour:
        if self.pipeline is None:
            raise NotImplementedError(
                "You must pass a prompt_pipeline to the VLLMEngine to use it as a non-abstract class."
            )
        prompt = self.pipeline.execute([message], for_measurement=True)
        # prompt str to tokens
        tokenized = self.tokenizer.encode(prompt)
        return len(tokenized)

    def function_token_reserve(self, functions: list[AIFunction]) -> int:
        if not functions:
            return 0
        # default concrete base behaviour:
        if self.pipeline is None:
            raise NotImplementedError(
                "You must pass a prompt_pipeline to the VLLMEngine to use it as a non-abstract class."
            )
        prompt = self.pipeline.execute([], functions, for_measurement=True)
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

    def build_prompt(self, messages: list[ChatMessage], functions: list[AIFunction] | None = None) -> str:
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

    async def predict(
        self,
        messages: list[ChatMessage],
        functions: list[AIFunction] | None = None,
        *,
        decode_kwargs: dict = None,  # to prevent HF compat things from breaking the call to .generate()
        **hyperparams,
    ) -> Completion:
        """
        Given the current context of messages and available functions, get the next predicted chat message from the LM.

        :param messages: The messages in the current chat context. ``sum(message_len(m) for m in messages)`` is
            guaranteed to be less than max_context_size.
        :param functions: The functions the LM is allowed to call.
        :param hyperparams: Any additional parameters to pass to AsyncLLMEngine.generate(). (See
            https://github.com/vllm-project/vllm/blob/main/vllm/engine/async_llm_engine.py)
        """
        prompt = self.build_prompt(messages, functions)
        kwargs = {
            "sampling_params": SamplingParams(max_tokens=None),
            "request_id": str(uuid.uuid4()),
            **self.hyperparams,
            **hyperparams,
        }

        # run it through the model
        # generation from vllm api entrypoint
        final_output = None
        async for request_output in self.model.generate(prompt, **kwargs):
            final_output = request_output

        assert final_output is not None
        content = final_output.outputs[0].text
        input_len = len(final_output.prompt_token_ids)
        output_len = len(final_output.outputs[0].token_ids)
        return Completion(
            ChatMessage.assistant(content),
            prompt_tokens=input_len,
            completion_tokens=output_len,
        )
