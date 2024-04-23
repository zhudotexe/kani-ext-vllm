# kani-ext-vllm

This repository adds the VLLMEngine.

This package is considered provisional and maintained on a best-effort basis. As such, it will not be released on
PyPI.

To install this package, you must install it using the git source:

```shell
$ pip install git+https://github.com/zhudotexe/kani-ext-vllm.git@main
```

See https://docs.vllm.ai/en/latest/index.html for more information on vLLM.

## Usage

```python
from kani import Kani, chat_in_terminal
from kani.prompts.impl import LLAMA3_PIPELINE
from kani.ext.vllm import VLLMEngine

engine = VLLMEngine(model_id="meta-llama/Meta-Llama-3-8B-Instruct", prompt_pipeline=LLAMA3_PIPELINE)
ai = Kani(engine)
chat_in_terminal(ai)
```

### Command R

Command R's HF
impl [does not support the full 128k ctx length](https://huggingface.co/CohereForAI/c4ai-command-r-v01/discussions/12).
Cohere [recommends using vLLM](https://huggingface.co/CohereForAI/c4ai-command-r-v01/discussions/32), so here we are.

```python
from kani import Kani, chat_in_terminal
from kani.ext.vllm import CommandRVLLMEngine

engine = CommandRVLLMEngine(model_id="CohereForAI/c4ai-command-r-v01")
ai = Kani(engine)
chat_in_terminal(ai)
```
