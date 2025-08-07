import asyncio
import logging
import uuid
import warnings

import torch
import transformers
from kani import AIFunction, ChatMessage, PromptPipeline, model_specific
from kani.engines import Completion
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams, TokensPrompt

from .bases import VLLMBase

log = logging.getLogger(__name__)


class VLLMEngine(VLLMBase):
    def __init__(
        self,
        model_id: str,
        max_context_size: int = None,
        prompt_pipeline: PromptPipeline[str | torch.Tensor] = None,
        *,
        model_load_kwargs: dict = None,
        chat_template_kwargs: dict = None,
        **hyperparams,
    ):
        """
        :param model_id: The ID of the model to load from HuggingFace.
        :param max_context_size: The context size of the model. If not given, will be set from the model's config.
        :param prompt_pipeline: The pipeline to translate a list of kani ChatMessages into the model-specific chat
            format (see :class:`.PromptPipeline`).
        :param model_load_kwargs: Additional arguments to pass to ``AsyncEngineArgs()``.
        :param chat_template_kwargs: The keyword arguments to pass to ``tokenizer.apply_chat_template`` if using a chat
            template prompt pipeline.
        :param hyperparams: Additional arguments to supply the model during generation.
        """
        if model_load_kwargs is None:
            model_load_kwargs = {}
        if chat_template_kwargs is None:
            chat_template_kwargs = {}

        engine_args = AsyncEngineArgs(model=model_id, max_model_len=max_context_size, **model_load_kwargs)
        engine = AsyncLLMEngine.from_engine_args(engine_args)

        tokenizer = engine.engine.get_tokenizer()
        self.model = engine
        self.hyperparams = hyperparams

        # load the pipeline
        if prompt_pipeline is None:
            if isinstance(tokenizer, transformers.PreTrainedTokenizerBase):
                # try and load a manual impl, or default to chat template if not available
                prompt_pipeline = model_specific.prompt_pipeline_for_hf_model(
                    model_id, self.tokenizer, chat_template_kwargs=chat_template_kwargs
                )
            else:
                raise ValueError(
                    "There is no chat template associated with this model (tokenizer loaded from a non-HF source)."
                    " Please provide a prompt_pipeline."
                )

        super().__init__(tokenizer=tokenizer, max_context_size=max_context_size, prompt_pipeline=prompt_pipeline)

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
        if decode_kwargs is None:
            decode_kwargs = dict(skip_special_tokens=True)
        prompt = self.build_prompt(messages, functions)
        request_id = str(uuid.uuid4())
        kwargs = {
            "sampling_params": SamplingParams(max_tokens=None),
            "request_id": request_id,
            **self.hyperparams,
            **hyperparams,
        }

        # tokenize it ourselves in order to capture special tokens correctly
        prompt_toks = self.tokenizer(prompt, add_special_tokens=False)
        input_len = len(prompt_toks.input_ids)
        prompt_toks = TokensPrompt(prompt_token_ids=prompt_toks.input_ids)

        # run it through the model
        # generation from vllm api entrypoint
        final_output = None
        try:
            async for request_output in self.model.generate(prompt_toks, **kwargs):
                final_output = request_output
        except (asyncio.CancelledError, KeyboardInterrupt):
            # if something cancels our task, make sure we tell vLLM to stop generating too
            await self.model.abort(request_id)
            raise

        assert final_output is not None
        # decode to tokens
        # the completion shouldn't include the prompt or stop token
        content = self.tokenizer.decode(final_output.outputs[0].token_ids, **decode_kwargs).strip()
        output_len = len(final_output.outputs[0].token_ids)
        log.debug(f"COMPLETION ({input_len=}, {output_len=}): {content}")
        return Completion(
            ChatMessage.assistant(content),
            prompt_tokens=input_len,
            completion_tokens=output_len,
        )

    async def close(self):
        self.model.shutdown_background_loop()
        self.model = None
