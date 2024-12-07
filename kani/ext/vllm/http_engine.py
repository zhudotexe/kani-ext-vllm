import json
import logging
import socket
import subprocess
import time
from typing import AsyncIterable

import httpx
import jinja2
import torch
from kani.ai_function import AIFunction
from kani.engines import BaseEngine, Completion
from kani.engines.huggingface.chat_template_pipeline import ChatTemplatePromptPipeline
from kani.engines.openai.translation import (
    ChatCompletion,
    openai_tc_to_kani_tc,
    translate_functions,
    translate_messages,
)
from kani.models import ChatMessage, ChatRole
from openai import AsyncOpenAI

log = logging.getLogger(__name__)


class VLLMServerEngine(BaseEngine):
    """
    Like the VLLMEngine, but uses an HTTP server and the OpenAI client.
    Useful for serving multiple models in parallel or models which have defined special vLLM support.
    """

    def __init__(
        self,
        model_id: str,
        max_context_size: int,
        vllm_args: dict = None,
        **hyperparams,
    ):
        r"""
        :param model_id: The ID of the model to load from HuggingFace.
        :param max_context_size: The context size of the model.
        :param vllm_args: See
            https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#command-line-arguments-for-the-server.
            Underscores will be converted to hyphens, dashes will be added, and values of True will be present.
            (e.g. ``{"enable_auto_tool_choice": True, "tool_call_parser": "mistral"}`` becomes
            ``--enable-auto-tool-choice --tool-call-parser mistral``\ .)

            .. note::
                the host will always be localhost, and the port will always be randomly chosen
        :param hyperparams: Additional arguments to supply the model during generation, see
            https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#extra-parameters-for-chat-completions-api.
        """
        if vllm_args is None:
            vllm_args = {}

        self.model_id = model_id
        self.max_context_size = max_context_size
        self.hyperparams = hyperparams

        # launch the server
        port = str(_get_free_port())
        _vargs = [
            "vllm",
            "serve",
            model_id,
            "--host",
            "127.0.0.1",
            "--port",
            port,
            *_kwargs_to_cli(vllm_args),
        ]
        log.info(f"Launching vLLM server with following command: {_vargs}")
        self.server = subprocess.Popen(_vargs)
        self.client = AsyncOpenAI(base_url=f"http://127.0.0.1:{port}/v1")
        self.http = httpx.Client(base_url=f"http://127.0.0.1:{port}")  # todo tokenization should be async

        self._wait_for_healthy_server()

        # load the pipeline
        self.tokenizer = HTTPTokenizerCompat(self.model_id, self.http)
        self.pipeline = ChatTemplatePromptPipeline(self.tokenizer)

        # infer the token reserve from tokenization
        if self.token_reserve == 0:
            self.token_reserve = self._infer_token_reserve()

    # ===== server shenanigans =====
    def _wait_for_healthy_server(self):
        healthy = False
        while not healthy:
            try:
                log.debug("Checking for healthy server...")
                resp = self.http.get("/health")
                resp.raise_for_status()
            except httpx.HTTPError as e:
                log.debug("Unhealthy server, waiting for 5 seconds...", exc_info=e)
                time.sleep(5)
                continue
            else:
                healthy = resp.status_code == 200

    # ===== main =====
    def _infer_token_reserve(self):
        """If token_reserve is not set and we have a pipeline, infer it."""
        prompt = self.pipeline.execute([], for_measurement=True)
        if isinstance(prompt, torch.Tensor):
            return len(prompt[0])
        return self.tokenizer.len(prompt)

    def message_len(self, message: ChatMessage) -> int:
        prompt = self.pipeline.execute([message], for_measurement=True)
        if isinstance(prompt, torch.Tensor):
            return len(prompt[0])
        return self.tokenizer.len(prompt)

    def function_token_reserve(self, functions: list[AIFunction]) -> int:
        if not functions:
            return 0

        # todo this is kind of hacky - since vllm doesn't expose function tokenization we'll just get the tokenization
        # of the json schema since that is usually an overestimate
        tool_specs = translate_functions(functions)
        json_schema = json.dumps(tool_specs)
        return self.tokenizer.len(json_schema)

    async def predict(
        self, messages: list[ChatMessage], functions: list[AIFunction] | None = None, **hyperparams
    ) -> ChatCompletion:
        if functions:
            tool_specs = translate_functions(functions)
        else:
            tool_specs = None
        # translate to openai spec - group any tool messages together and ensure all free ToolCall IDs are bound
        translated_messages = translate_messages(messages)
        # make API call
        completion = await self.client.chat.completions.create(
            model=self.model_id, messages=translated_messages, tools=tool_specs, **self.hyperparams, **hyperparams
        )
        # translate into Kani spec and return
        kani_cmpl = ChatCompletion(openai_completion=completion)
        return kani_cmpl

    async def stream(
        self, messages: list[ChatMessage], functions: list[AIFunction] | None = None, **hyperparams
    ) -> AsyncIterable[str | Completion]:
        if functions:
            tool_specs = translate_functions(functions)
        else:
            tool_specs = None
        # translate to openai spec - group any tool messages together and ensure all free ToolCall IDs are bound
        translated_messages = translate_messages(messages)
        # make API call
        stream = await self.client.chat.completions.create(
            model=self.model_id,
            messages=translated_messages,
            tools=tool_specs,
            stream=True,
            stream_options={"include_usage": True},
            **self.hyperparams,
            **hyperparams,
        )

        # save requested tool calls and content as streamed
        content_chunks = []
        tool_call_partials = {}  # index -> tool call
        usage = None

        # iterate over the stream and yield/save
        async for chunk in stream:
            # save usage if present
            if chunk.usage is not None:
                usage = chunk.usage

            if not chunk.choices:
                continue

            # process content delta
            delta = chunk.choices[0].delta

            # yield content
            if delta.content is not None:
                content_chunks.append(delta.content)
                yield delta.content

            # tool calls are partials, save a mapping to the latest state and we'll translate them later once complete
            if delta.tool_calls:
                # each tool call can have EITHER the function.name/id OR function.arguments
                for tc in delta.tool_calls:
                    if tc.id is not None:
                        tool_call_partials[tc.index] = tc
                    else:
                        partial = tool_call_partials[tc.index]
                        partial.function.arguments += tc.function.arguments

        # construct the final completion with streamed tool calls
        content = None if not content_chunks else "".join(content_chunks)
        tool_calls = [openai_tc_to_kani_tc(tc) for tc in sorted(tool_call_partials.values(), key=lambda c: c.index)]
        msg = ChatMessage(role=ChatRole.ASSISTANT, content=content, tool_calls=tool_calls)

        # token counting
        if usage:
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens
        else:
            prompt_tokens = completion_tokens = 0
        yield Completion(message=msg, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)

    async def close(self):
        self.server.terminate()


class HTTPTokenizerCompat:
    """Duck-typed tokenizer to provide ChatTemplatePromptPipeline compatibility."""

    def __init__(self, model_id, http):
        self.model_id = model_id
        self.http = http

    # http
    def _request_tokenize_str(self, q):
        resp = self.http.post("/tokenize", json={"model": self.model_id, "prompt": q})
        resp.raise_for_status()
        data = resp.json()
        return data

    def _request_tokenize_msg(self, messages, add_special_tokens=True, add_generation_prompt=True, **_):
        resp = self.http.post(
            "/tokenize",
            json={
                "model": self.model_id,
                "messages": messages,
                "add_special_tokens": add_special_tokens,
                "add_generation_prompt": add_generation_prompt,
            },
        )
        resp.raise_for_status()
        return resp.json()

    # compat
    def encode(self, prompt: str) -> list[int]:
        data = self._request_tokenize_str(prompt)
        return data["tokens"]

    def len(self, prompt: str) -> int:
        data = self._request_tokenize_str(prompt)
        return data["count"]

    def apply_chat_template(self, messages: list[ChatMessage], **kwargs) -> list[int]:
        oai_messages = translate_messages(messages) if messages else []
        try:
            data = self._request_tokenize_msg(oai_messages, **kwargs)
        except httpx.HTTPError as e:
            # hack to make ChatTemplatePromptPipeline fallback
            raise jinja2.TemplateError("This is a hacky reraise; see above error.") from e
        return data["tokens"]


# ==== utils ====
def _kwargs_to_cli(args: dict) -> list[str]:
    out = []
    for k, v in args.items():
        if v is False:
            continue
        k = f"--{k.replace('-', '_')}"
        out.append(k)
        if v is not True:
            out.append(str(v))

    return out


def _get_free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    port = s.getsockname()[1]
    s.close()
    return port
