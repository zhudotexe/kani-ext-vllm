import logging
import socket
import subprocess
import time

import httpx
import torch
import transformers
from kani import PromptPipeline
from kani.ai_function import AIFunction
from kani.engines import Completion
from kani.engines.huggingface.chat_template_pipeline import ChatTemplatePromptPipeline
from kani.models import ChatMessage
from openai import AsyncOpenAI
from transformers import AutoTokenizer

from .bases import VLLMBase

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
        self.client = AsyncOpenAI(
            base_url=f"http://127.0.0.1:{port}/v1",
            api_key="<the library wants this but it isn't needed>",
            timeout=timeout,
        )
        self.http = httpx.Client(base_url=f"http://127.0.0.1:{port}")  # todo tokenization should be async

        _wait_for_healthy_server(self.http)

        # load the pipeline
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        # load the pipeline
        if prompt_pipeline is None:
            if isinstance(tokenizer, transformers.PreTrainedTokenizerBase):
                prompt_pipeline = ChatTemplatePromptPipeline(tokenizer)
            else:
                raise ValueError(
                    "There is no chat template associated with this model (tokenizer loaded from a non-HF source)."
                    " Please provide a prompt_pipeline."
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

        prompt = self.build_prompt(messages, functions)
        kwargs = {
            "max_tokens": None,
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
        self.server.terminate()


# ==== utils ====
def _wait_for_healthy_server(http: httpx.Client):
    healthy = False
    while not healthy:
        try:
            log.debug("Checking for healthy server...")
            resp = http.get("/health")
            resp.raise_for_status()
        except httpx.HTTPError as e:
            log.debug("Unhealthy server, waiting for 5 seconds...", exc_info=e)
            time.sleep(5)
            continue
        else:
            healthy = resp.status_code == 200


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
