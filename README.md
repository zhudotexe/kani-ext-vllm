# kani-ext-vllm

This repository adds the VLLMEngine.

This package is considered provisional and maintained on a best-effort basis.

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

This package provides 2 main methods of serving models with vLLM: offline mode (preferred), and API mode.
These are generally equivalent, and differ only in how Kani communicates with vLLM workers.

Generally, you should use offline mode unless you need to load multiple models in parallel.

### Offline Mode

```python
from kani import Kani, chat_in_terminal
from kani.ext.vllm import VLLMEngine

engine = VLLMEngine(model_id="meta-llama/Meta-Llama-3-8B-Instruct")
ai = Kani(engine)
chat_in_terminal(ai)
```

### API Mode

> [!NOTE]
> Using offline mode is preferred unless you need to load multiple models in parallel.

> [!NOTE]
> The vLLM server will be started on a random free port. It will not be exposed to the wider internet (i.e, it binds to
> localhost).

When loading a model in API mode, the model's context length can not be read from the configuration, so you must pass
the `max_context_len`.

```python
from kani import Kani, chat_in_terminal
from kani.ext.vllm import VLLMServerEngine

engine = VLLMServerEngine(model_id="meta-llama/Meta-Llama-3-8B-Instruct", max_context_len=128000)
ai = Kani(engine)
chat_in_terminal(ai)
```

### Using Multiple GPUs

For multi-GPU support (probably needed), add `model_load_kwargs={"tensor_parallel_size": 4}`. Replace "4" with the
number of GPUs you have available.

> [!NOTE]
> If you are loading in API mode, use `vllm_args={"tensor_parallel_size": 4}` instead.

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

### API Mode

```python
from kani.ext.vllm import VLLMServerEngine

model = VLLMServerEngine(
    model_id="mistralai/Mistral-Small-Instruct-2409",
    max_context_len=32000,
    vllm_args={"tensor_parallel_size": 2, "tokenizer_mode": "auto"},
    # note that these should not be wrapped in SamplingParams!
    temperature=0,
    max_tokens=2048,
)
```
