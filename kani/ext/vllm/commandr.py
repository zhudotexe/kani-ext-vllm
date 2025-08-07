import logging
import uuid
from collections.abc import AsyncIterable

from kani import AIFunction, ChatMessage, ChatRole
from kani.engines import Completion
from kani.model_specific.cohere import CommandRMixin, function_prompt, tool_call_formatter
from vllm import SamplingParams

from .engine import VLLMEngine

log = logging.getLogger(__name__)


class CommandRVLLMEngine(CommandRMixin, VLLMEngine):
    """Implementation of Command R (35B) and Command R+ (104B) using vllm.

    Model IDs:

    - ``CohereForAI/c4ai-command-r-v01``
    - ``CohereForAI/c4ai-command-r-plus``

    **GPU Support**

    By default, the CommandREngine loads the model on GPU if CUDA is detected on your system. To override the device
    the model is loaded on, pass ``device="cpu|cuda"`` to the constructor.

    **Usage**

    .. code-block:: python

        engine = CommandRVLLMEngine("CohereForAI/c4ai-command-r-v01")
        ai = KaniWithFunctions(engine)

    **Configuration**

    Command R has many configurations that enable function calling and/or RAG, and it is poorly documented exactly
    how certain prompts affect the model. In this implementation, we default to the Cohere-supplied "preamble" if
    function definitions are supplied, and assume that we pass every generated function call and results each turn.

    When generating the result of a tool call turn, this implementation does NOT request the model to generate
    citations by default (unlike the Cohere API). You can enable citations by setting the ``rag_prompt_instructions``
    parameter to ``DEFAULT_RAG_INSTRUCTIONS_ACC`` or ``DEFAULT_RAG_INSTRUCTIONS_FAST`` (imported from
    ``kani.model_specific.cohere``).

    See the constructor's available parameters for more information.

    .. seealso:: https://huggingface.co/CohereForAI/c4ai-command-r-v01
    """

    token_reserve = 200  # generous reserve due to large ctx size and weird 3-mode prompt

    def __init__(self, model_id: str = "CohereForAI/c4ai-command-r-v01", *args, **kwargs):
        """
        :param model_id: The ID of the model to load from HuggingFace.
        :param max_context_size: The context size of the model (defaults to Command R's size of 128k).
        :param device: The hardware device to use. If not specified, uses CUDA if available; otherwise uses CPU.
        :param tokenizer_kwargs: Additional arguments to pass to ``AutoTokenizer.from_pretrained()``.
        :param model_load_kwargs: Additional arguments to pass to ``AutoModelForCausalLM.from_pretrained()``.
        :param tool_prompt_include_function_calls: Whether to include previous turns' function calls or just the model's
            answers when it is the model's generation turn and the last message is not FUNCTION.
        :param tool_prompt_include_function_results: Whether to include the results of previous turns' function calls in
            the context when it is the model's generation turn and the last message is not FUNCTION.
        :param tool_prompt_instructions: The system prompt to send just before the model's generation turn that includes
            instructions on the format to generate tool calls in. Generally you shouldn't change this.
        :param rag_prompt_include_function_calls: Whether to include previous turns' function calls or just the model's
            answers when it is the model's generation turn and the last message is FUNCTION.
        :param rag_prompt_include_function_results: Whether to include the results of previous turns' function calls in
            the context when it is hte model's generation turn and the last message is FUNCTION.
        :param rag_prompt_instructions: The system prompt to send just before the model's generation turn that includes
            instructions on the format to generate the result in. Can be None to only generate a model turn. Defaults
            to ``None`` to for maximum interoperability between models. Options:

            - ``from kani.model_specific.cohere import DEFAULT_RAG_INSTRUCTIONS_ACC``
            - ``from kani.model_specific.cohere import DEFAULT_RAG_INSTRUCTIONS_FAST``
            - ``None`` (default)
            - another user-supplied string
        :param hyperparams: Additional arguments to supply the model during generation.
        """
        kwargs.setdefault("max_context_size", 128000)
        super().__init__(model_id, *args, **kwargs)

    # ==== token counting ====
    def message_len(self, message: ChatMessage) -> int:
        # prompt str to tokens
        if message.text:
            tokenized = self.tokenizer.encode(message.text, add_special_tokens=False)
        else:
            tokenized = 0

        # worst-case function calls if we have them
        if self._tool_prompt_include_function_calls and message.role == ChatRole.ASSISTANT:
            func_body = tool_call_formatter(message)
            tokenized = self.tokenizer.encode(func_body, add_special_tokens=False)
            return len(tokenized) + 3

        # <results></results>
        if message.role == ChatRole.FUNCTION:
            return len(tokenized) + 12

        return len(tokenized) + 3

    def function_token_reserve(self, functions: list[AIFunction]) -> int:
        # include the additional default system prompt tokens here plus the directly_answer tool
        default_prompt_tokens = 325
        function_text = "\n\n".join(map(function_prompt, functions))
        function_tokens = len(self.tokenizer.encode(function_text, add_special_tokens=False))
        return function_tokens + default_prompt_tokens

    # ==== generate ====
    async def _generate(self, prompt, *, parse_functions, **hyperparams):
        """Generate and return a completion (may be a directly_answer call)."""
        kwargs = {
            "sampling_params": SamplingParams(max_tokens=None),
            "request_id": str(uuid.uuid4()),
            **self.hyperparams,
            **hyperparams,
        }

        prompt_toks = self.tokenizer(prompt, add_special_tokens=False)
        input_len = len(prompt_toks.input_ids)

        # run it through the model
        # generation from vllm api entrypoint
        final_output = None
        async for request_output in self.model.generate(prompt, **kwargs):
            final_output = request_output

        assert final_output is not None
        content = final_output.outputs[0].text
        output_len = len(final_output.outputs[0].token_ids)
        return self._parse_completion(content, parse_functions, prompt_tokens=input_len, completion_tokens=output_len)

    async def _stream(self, prompt, **hyperparams) -> AsyncIterable[str | Completion]:
        """Low-level stream yielder (kind of weird duplicated code but it's ok)"""
        kwargs = {
            "sampling_params": SamplingParams(max_tokens=None),
            "request_id": str(uuid.uuid4()),
            **self.hyperparams,
            **hyperparams,
        }

        prompt_toks = self.tokenizer(prompt, add_special_tokens=False)
        input_len = len(prompt_toks.input_ids)

        # run it through the model
        # generation from vllm api entrypoint
        last_generation = ""
        last_output = None
        async for request_output in self.model.generate(prompt, **kwargs):
            chunk = request_output.outputs[0].text
            yield chunk.removeprefix(last_generation)
            last_generation = chunk
            last_output = request_output
        output_len = len(last_output.outputs[0].token_ids)
        yield Completion(ChatMessage.assistant(last_generation), prompt_tokens=input_len, completion_tokens=output_len)

    async def predict(
        self,
        messages: list[ChatMessage],
        functions: list[AIFunction] | None = None,
        **hyperparams,
    ) -> Completion:
        prompt = self.build_prompt(messages, functions)
        completion = await self._generate(prompt, parse_functions=functions is not None, **hyperparams)
        cmd_r_tc_info = self._toolcall_info(completion.message.tool_calls)

        # if the model generated multiple calls that happen to include a directly_answer, remove the directly_answer
        completion.message.tool_calls = cmd_r_tc_info.filtered_tool_calls
        # if tool says directly answer, call again with the rag pipeline (but no result)
        if cmd_r_tc_info.is_directly_answer:
            log.debug("GOT DIRECTLY_ANSWER, REPROMPTING RAG...")
            prompt = self._build_prompt_rag(messages)
            log.debug(f"RAG PROMPT: {prompt}")
            pre_prompt_tokens = completion.prompt_tokens
            pre_completion_tokens = completion.completion_tokens
            completion = await self._generate(prompt, parse_functions=functions is not None, **hyperparams)
            completion._prompt_tokens += pre_prompt_tokens
            completion._completion_tokens += pre_completion_tokens
        # otherwise don't touch it
        return completion

    async def stream(
        self,
        messages: list[ChatMessage],
        functions: list[AIFunction] | None = None,
        **hyperparams,
    ) -> AsyncIterable[str | Completion]:
        # if we have functions things get weird
        # if we have tools and the last turn is not FUNCTION, no-stream the first round to get the Action
        if functions and not (messages and messages[-1].role == ChatRole.FUNCTION):
            prompt = self.build_prompt(messages, functions)
            completion = await self._generate(prompt, parse_functions=functions is not None, **hyperparams)
            cmd_r_tc_info = self._toolcall_info(completion.message.tool_calls)

            # if tool says directly answer, stream with the rag pipeline (but no result)
            if cmd_r_tc_info.is_directly_answer:
                pre_prompt_tokens = completion.prompt_tokens
                pre_completion_tokens = completion.completion_tokens
                log.debug("GOT DIRECTLY_ANSWER, REPROMPTING RAG...")
                prompt = self._build_prompt_rag(messages)
                log.debug(f"RAG PROMPT: {prompt}")
                async for elem in self._stream(prompt, **hyperparams):
                    if isinstance(elem, Completion):
                        # ensure we count the tokens from the first step too
                        yield Completion(
                            elem.message,
                            prompt_tokens=pre_prompt_tokens + elem.prompt_tokens,
                            completion_tokens=pre_completion_tokens + elem.completion_tokens,
                        )
                    else:
                        yield elem
            # if the model generated multiple calls that happen to include a directly_answer, remove the directly_answer
            # then yield as normal
            else:
                completion.message.tool_calls = cmd_r_tc_info.filtered_tool_calls
                if completion.message.text:
                    yield completion.message.text
                yield completion
        # otherwise stream as normal
        else:
            async for elem in super().stream(messages, functions, **hyperparams):
                yield elem
