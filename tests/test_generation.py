import random

from kani import Kani, print_stream
from kani.ext.vllm import VLLMEngine, VLLMOpenAIEngine, VLLMServerEngine
from vllm import SamplingParams

from tests.conftest import MODEL_ID, PROMPTS, SEED

# @pytest.mark.parametrize("engine", [lf("offline_engine"), lf("api_engine"), lf("openai_engine")])
# @pytest.mark.parametrize("stream", [False, True])
# async def test_generation(engine, stream):
#     prompt = random.choice(PROMPTS)
#     ai = Kani(engine)
#     if stream:
#         stream = ai.chat_round_stream(prompt)
#         await print_stream(stream)
#         resp = (await stream.message()).text
#     else:
#         resp = await ai.chat_round_str(prompt)
#         print(f"{type(engine).__name__}: {resp}")
#     assert resp


async def _query_engine(engine, stream):
    prompt = random.choice(PROMPTS)
    ai = Kani(engine)
    if stream:
        stream = ai.chat_round_stream(prompt)
        await print_stream(stream, prefix=f"{type(engine).__name__} (streaming): ")
        resp = (await stream.message()).text
    else:
        resp = await ai.chat_round_str(prompt)
        print(f"{type(engine).__name__} (no streaming): {resp}")
    assert resp


async def test_offline_engine():
    engine = VLLMEngine(
        model_id=MODEL_ID,
        max_context_size=8192,
        model_load_kwargs={"seed": SEED},
        sampling_params=SamplingParams(temperature=0, max_tokens=2048),
    )
    await _query_engine(engine, stream=False)
    await _query_engine(engine, stream=True)
    await engine.close()


async def test_api_engine():
    engine = VLLMServerEngine(
        model_id=MODEL_ID,
        max_context_size=8192,
        vllm_args={"seed": SEED},
        vllm_port=31415,
        timeout=3000,
        temperature=0,
        max_tokens=2048,
    )
    await _query_engine(engine, stream=False)
    await _query_engine(engine, stream=True)
    await engine.close()


async def test_openai_engine():
    engine = VLLMOpenAIEngine(
        model_id=MODEL_ID,
        max_context_size=8192,
        vllm_args={"seed": SEED},
        vllm_port=31416,
        timeout=3000,
        temperature=0,
        max_tokens=2048,
    )
    await _query_engine(engine, stream=False)
    await _query_engine(engine, stream=True)
    await engine.close()
