"""
Ensure that the tokens generated by the offline vLLM engine are the same as the tokens in the API mode.

We do this by just spinning up one of each and seeing what it outputs. This runs on CPU, so we use a tiny model.
"""

from kani import Kani
from kani.ext.vllm import VLLMEngine, VLLMServerEngine
from vllm import SamplingParams

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"


async def offline():
    model = VLLMEngine(
        model_id=MODEL_ID,
        max_context_size=8192,
        model_load_kwargs={"seed": 31415},
        sampling_params=SamplingParams(temperature=0, max_tokens=2048),
    )
    ai = Kani(model)

    resp = await ai.chat_round_str("Tell me about the Boeing 737.")
    await model.close()
    del model
    return resp


async def api():
    model = VLLMServerEngine(
        model_id=MODEL_ID,
        max_context_size=8192,
        vllm_args={"seed": 31415},
        temperature=0,
        max_tokens=2048,
    )
    ai = Kani(model)

    resp = await ai.chat_round_str("Tell me about the Boeing 737.")
    await model.close()
    del model
    return resp


async def test_equivalence():
    resp1 = await offline()
    print(f"OFFLINE: {resp1}")
    resp2 = await api()
    print(f"API: {resp2}")
    assert resp1 == resp2
