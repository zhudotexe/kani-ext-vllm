# kani-ext-vllm

This Kani extension repository adds 3 engines for using vLLM to deploy LLMs on local hardware.

vLLM is an LLM deployment platform optimized for GPU memory efficiency and throughput. This extension adds Kani engines
to use vLLM engines in offline mode, manage a vLLM server, or connect to an existing vLLM server depending on the 
use case.

To install this package, you can install it from PyPI:

```shell
$ pip install kani-ext-vllm
```

Alternatively, you can install it using the git source:

```shell
$ pip install git+https://github.com/zhudotexe/kani-ext-vllm.git@main
```

See https://docs.vllm.ai/en/latest/index.html for more information on vLLM.

## Usage

This package provides 3 main methods of serving models with vLLM:

- Offline mode
- vLLM-Native API mode
- OpenAI-Compatible API mode

These are generally equivalent, but offer slightly different options for each mode:

| **Mode**   | **Communication** | **Multiple Parallel Models?** | **Prompt Template/Parsing** | **Best For**                                                  |
|------------|-------------------|-------------------------------|-----------------------------|---------------------------------------------------------------|
| Offline    | Local             | No                            | kani                        | Low-level control over the model                              |
| vLLM API   | HTTP              | Yes                           | kani                        | Running multiple different models in parallel                 |
| OpenAI API | HTTP              | Yes                           | vLLM                        | Fast iteration and testing multiple models; multimodal models |

### Offline Mode

```python
from kani import Kani, chat_in_terminal
from kani.ext.vllm import VLLMEngine

engine = VLLMEngine(model_id="meta-llama/Meta-Llama-3-8B-Instruct")
ai = Kani(engine)
chat_in_terminal(ai)
```

### vLLM-Native API Mode

The API mode can be used to connect to an existing running vLLM server or to start a managed vLLM server.

**Connecting to a Running Server**

```python
from kani import Kani, chat_in_terminal
from kani.ext.vllm import VLLMServerEngine

engine = VLLMServerEngine(
    model_id="meta-llama/Meta-Llama-3-8B-Instruct",
    vllm_host="127.0.0.1",
    vllm_port=8000,
    use_managed_server=False,
)
ai = Kani(engine)
chat_in_terminal(ai)
```

**Managed Server**

> [!NOTE]
> The vLLM server will be started on a random free port. It will not be exposed to the wider internet (i.e, it binds to
> localhost).

```python
from kani import Kani, chat_in_terminal
from kani.ext.vllm import VLLMServerEngine

engine = VLLMServerEngine(model_id="meta-llama/Meta-Llama-3-8B-Instruct")
ai = Kani(engine)
chat_in_terminal(ai)
```

### OpenAI-Compatible API Mode

**Connecting to a Running Server**

```python
from kani import Kani, chat_in_terminal
from kani.ext.vllm import VLLMOpenAIEngine

engine = VLLMOpenAIEngine(
    model_id="meta-llama/Meta-Llama-3-8B-Instruct",
    vllm_host="127.0.0.1",
    vllm_port=8000,
    use_managed_server=False,
)
ai = Kani(engine)
chat_in_terminal(ai)
```

**Managed Server**

> [!NOTE]
> The vLLM server will be started on a random free port. It will not be exposed to the wider internet (i.e, it binds to
> localhost).

```python
from kani import Kani, chat_in_terminal
from kani.ext.vllm import VLLMOpenAIEngine

engine = VLLMOpenAIEngine(model_id="meta-llama/Meta-Llama-3-8B-Instruct")
ai = Kani(engine)
chat_in_terminal(ai)
```

### Using Multiple GPUs

For multi-GPU support (probably needed), add `model_load_kwargs={"tensor_parallel_size": 4}`. Replace "4" with the
number of GPUs you have available.

> [!NOTE]
> If you are loading in an API mode, use `vllm_args={"tensor_parallel_size": 4}` instead.

## Examples

### Offline Mode

```python
from kani.ext.vllm import VLLMEngine
from vllm import SamplingParams

model = VLLMEngine(
    model_id="mistralai/Mistral-Small-Instruct-2409",
    model_load_kwargs={"tensor_parallel_size": 2, "tokenizer_mode": "auto"},
    sampling_params=SamplingParams(temperature=0, max_tokens=2048),
)
```

### vLLM-Native API Mode

```python
from kani.ext.vllm import VLLMServerEngine

model = VLLMServerEngine(
    model_id="mistralai/Mistral-Small-Instruct-2409",
    vllm_args={"tensor_parallel_size": 2, "tokenizer_mode": "auto"},
    # note that these should not be wrapped in SamplingParams!
    temperature=0,
    max_tokens=2048,
)
```

See https://docs.vllm.ai/en/stable/serving/openai_compatible_server.html#completions-api_1 for a list of valid decoding
parameters that can be specified in the engine constructor.

See https://docs.vllm.ai/en/stable/cli/serve/ for a list of valid arguments to `vllm_args`.

### OpenAI-Compatible API Mode

```python
from kani.ext.vllm import VLLMOpenAIEngine

model = VLLMOpenAIEngine(
    model_id="Qwen/Qwen3-Omni-30B-A3B-Instruct",
    vllm_args={"tensor_parallel_size": 2, "allowed_local_media_path": "/"},
    # note that these should not be wrapped in SamplingParams!
    temperature=0,
    max_tokens=2048,
)
```

See https://docs.vllm.ai/en/stable/serving/openai_compatible_server.html#chat-api_1 for a list of valid decoding
parameters that can be specified in the engine constructor.

See https://docs.vllm.ai/en/stable/cli/serve/ for a list of valid arguments to `vllm_args`.
