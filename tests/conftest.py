import logging
import os
import random

import pytest
from vllm import SamplingParams

from kani import Kani
from kani.ext.vllm import VLLMEngine, VLLMOpenAIEngine, VLLMServerEngine

if os.getenv("KANI_DEBUG"):
    logging.basicConfig(level=logging.DEBUG)


MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
# random prompts
PROMPTS = [
    "Tell me about the Boeing 737.",
    "Without using the Shinkansen, how do I get from Oku-Tama to Komagome?",
    "Help me come up with a new magic item for D&D called the Blade of Kani.",
    "How do I set up vLLM?",
    "Please output as many of the letter 'a' as possible.",
    "How many 'a's are in the word 'strawberry'?",
]
SEED = random.randint(0, 99999)


# define engines to test with
@pytest.fixture(scope="session")
async def offline_engine():
    model = VLLMEngine(
        model_id=MODEL_ID,
        max_context_size=8192,
        model_load_kwargs={"seed": SEED, "gpu_memory_utilization": 0.3},
        sampling_params=SamplingParams(temperature=0, max_tokens=2048),
    )
    yield model
    await model.close()
    del model


@pytest.fixture(scope="session")
async def api_engine():
    model = VLLMServerEngine(
        model_id=MODEL_ID,
        max_context_size=8192,
        vllm_args={"seed": SEED, "gpu_memory_utilization": 0.3},
        vllm_port=31415,
        timeout=3000,
        temperature=0,
        max_tokens=2048,
    )
    yield model
    await model.close()
    del model


@pytest.fixture(scope="session")
async def openai_engine():
    model = VLLMOpenAIEngine(
        model_id=MODEL_ID,
        max_context_size=8192,
        vllm_args={"seed": SEED, "gpu_memory_utilization": 0.3},
        vllm_port=31416,
        timeout=3000,
        temperature=0,
        max_tokens=2048,
    )
    yield model
    await model.close()
    del model


# define helpers to call the engines with
async def infer_with_engine(engine, prompt, stream=False):
    ai = Kani(engine)
    if stream:
        msg = await ai.chat_round_stream(prompt)
        resp = msg.text
    else:
        resp = await ai.chat_round_str(prompt)
    return resp
