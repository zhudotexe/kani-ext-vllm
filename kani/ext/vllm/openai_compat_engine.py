import logging
from typing import AsyncIterable

from kani import AIFunction, ChatMessage
from kani.engines.base import BaseCompletion
from kani.engines.openai import OpenAIEngine
from kani.engines.openai.translation import ChatCompletion
from openai import AsyncOpenAI

from .vllm_server import VLLMServer

log = logging.getLogger(__name__)


class VLLMOpenAIEngine(OpenAIEngine):
    """
    Like the VLLMEngine, but uses an HTTP server and the OpenAI client with the vLLM server handling chat
    translation/tokenization.

    Useful for fast iteration for models with tool parsers already implemented by vLLM and/or multimodal models.
    """

    def __init__(
        self,
        model_id: str,
        max_context_size: int,
        *,
        timeout: int = 600,
        vllm_args: dict = None,
        vllm_host: str = "127.0.0.1",
        vllm_port: int = None,
        **hyperparams,
    ):
        r"""
        :param model_id: The ID of the model to load from HuggingFace.
        :param max_context_size: The context size of the model.
        :param vllm_args: See https://docs.vllm.ai/en/stable/cli/serve.html.
            Underscores will be converted to hyphens, dashes will be added, and values of True will be present.
            (e.g. ``{"enable_auto_tool_choice": True, "tool_call_parser": "mistral"}`` becomes
            ``--enable-auto-tool-choice --tool-call-parser mistral``\ .)
        :param vllm_host: The host to bind the vLLM server to. Defaults to localhost.
        :param vllm_port: The port to bind the vLLM server to. Defaults to a random free port.
        :param hyperparams: Additional arguments to supply the model during generation, see
            https://docs.vllm.ai/en/stable/serving/openai_compatible_server.html#chat-api_1.
        """
        # launch the server, create a client pointing to it, then pass to the OpenAIEngine
        self.server = VLLMServer(model_id=model_id, vllm_args=vllm_args, host=vllm_host, port=vllm_port)

        openai_client = AsyncOpenAI(
            base_url=f"http://127.0.0.1:{self.server.port}/v1",
            api_key="<the library wants this but it isn't needed>",
            timeout=timeout,
        )
        super().__init__(model=model_id, max_context_size=max_context_size, client=openai_client, **hyperparams)

    # OpenAIEngine patches
    def _load_tokenizer(self):
        return None

    # ===== main =====
    async def prompt_len(self, messages, functions=None, **kwargs) -> int:
        await self.server.wait_for_healthy()
        # make an HTTP request to the vLLM server to figure it out
        local_kwargs, translated_messages, tool_specs = self._prepare_request(
            messages, functions, intent="vllm.tokenize"
        )
        payload = {"model": self.model, "messages": translated_messages, "tools": tool_specs, **(local_kwargs | kwargs)}
        resp = await self.server.http.post("/tokenize", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["count"]

    async def predict(
        self, messages: list[ChatMessage], functions: list[AIFunction] | None = None, **hyperparams
    ) -> ChatCompletion:
        await self.server.wait_for_healthy()
        return await super().predict(messages, functions, **hyperparams)

    async def stream(
        self, messages: list[ChatMessage], functions: list[AIFunction] | None = None, **hyperparams
    ) -> AsyncIterable[str | BaseCompletion]:
        await self.server.wait_for_healthy()
        async for elem in super().stream(messages, functions, **hyperparams):
            yield elem

    async def close(self):
        await self.server.close()
