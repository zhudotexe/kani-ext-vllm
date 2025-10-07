import logging

import torch
from kani import PromptPipeline, model_specific
from kani.ai_function import AIFunction
from kani.engines import Completion
from kani.models import ChatMessage
from openai import AsyncOpenAI
from transformers import AutoTokenizer

from .bases import VLLMBase
from .vllm_server import VLLMServer

log = logging.getLogger(__name__)


class VLLMServerEngine(VLLMBase):
    """
    Like the VLLMEngine, but uses an HTTP server and the OpenAI client with the client handling chat
    translation/tokenization.

    Will only work for models with HF tokenizers. Replace ``model_load_kwargs`` with ``vllm_args`` and move any
    ``sampling_params`` to top-level kwargs.

    Useful for serving multiple models in parallel.
    """

    def __init__(
        self,
        model_id: str,
        max_context_size: int,
        prompt_pipeline: PromptPipeline[str | torch.Tensor] = None,
        *,
        timeout: int = 600,
        vllm_args: dict = None,
        vllm_host: str = "127.0.0.1",
        vllm_port: int = None,
        chat_template_kwargs: dict = None,
        **hyperparams,
    ):
        r"""
        :param model_id: The ID of the model to load from HuggingFace.
        :param max_context_size: The context size of the model.
        :param prompt_pipeline: A kani PromptPipeline to use.
        :param vllm_args: See https://docs.vllm.ai/en/stable/cli/serve.html.
            Underscores will be converted to hyphens, dashes will be added, and values of True will be present.
            (e.g. ``{"enable_auto_tool_choice": True, "tool_call_parser": "mistral"}`` becomes
            ``--enable-auto-tool-choice --tool-call-parser mistral``\ .)
        :param vllm_host: The host to bind the vLLM server to. Defaults to localhost.
        :param vllm_port: The port to bind the vLLM server to. Defaults to a random free port.
        :param chat_template_kwargs: The keyword arguments to pass to ``tokenizer.apply_chat_template`` if using a chat
            template prompt pipeline.
        :param hyperparams: Additional arguments to supply the model during generation, see
            https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#completions-api_1.
        """
        if chat_template_kwargs is None:
            chat_template_kwargs = {}

        self.model_id = model_id
        self.hyperparams = hyperparams

        self.server = VLLMServer(model_id=model_id, vllm_args=vllm_args, host=vllm_host, port=vllm_port)
        self.client = AsyncOpenAI(
            base_url=f"http://127.0.0.1:{self.server.port}/v1",
            api_key="<the library wants this but it isn't needed>",
            timeout=timeout,
        )

        # load the pipeline
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if prompt_pipeline is None:
            # try and load a manual impl, or default to chat template if not available
            prompt_pipeline = model_specific.prompt_pipeline_for_hf_model(
                model_id, tokenizer, chat_template_kwargs=chat_template_kwargs
            )
        super().__init__(tokenizer=tokenizer, max_context_size=max_context_size, prompt_pipeline=prompt_pipeline)

    # ===== main =====
    async def predict(
        self,
        messages: list[ChatMessage],
        functions: list[AIFunction] | None = None,
        *,
        decode_kwargs: dict = None,  # to prevent HF compat things from breaking the call to .generate()
        **hyperparams,
    ) -> Completion:
        if decode_kwargs is None:
            decode_kwargs = {}
        decode_kwargs.setdefault("skip_special_tokens", False)

        prompt = self.build_prompt(messages, functions)
        kwargs = {
            "max_tokens": self.max_context_size,  # setting this to None causes a 500 for some reason
            **self.hyperparams,
            **hyperparams,
        }

        # tokenize it ourselves in order to capture special tokens correctly
        if isinstance(prompt, list):
            prompt_toks = prompt
        elif isinstance(prompt, torch.Tensor):
            prompt_toks = prompt[0].tolist()
        else:
            prompt_toks = self.tokenizer.encode(prompt, add_special_tokens=False)

        # run it through the model
        await self.server.wait_for_healthy()
        # generation from vllm api entrypoint
        completion = await self.client.completions.create(
            model=self.model_id, prompt=prompt_toks, extra_body=decode_kwargs, **kwargs
        )
        content = completion.choices[0].text.strip()

        input_len = completion.usage.prompt_tokens
        output_len = completion.usage.completion_tokens
        log.debug(f"COMPLETION ({input_len=}, {output_len=}): {content}")
        return Completion(
            ChatMessage.assistant(content),
            prompt_tokens=input_len,
            completion_tokens=output_len,
        )

    async def close(self):
        self.server.close()
