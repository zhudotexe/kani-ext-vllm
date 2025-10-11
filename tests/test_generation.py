import random

import pytest
from pytest_lazy_fixtures import lf

from kani import Kani, print_stream
from tests.conftest import PROMPTS


@pytest.mark.parametrize("engine", [lf("offline_engine"), lf("api_engine"), lf("openai_engine")])
@pytest.mark.parametrize("stream", [False, True])
async def test_generation(engine, stream):
    prompt = random.choice(PROMPTS)
    ai = Kani(engine)
    if stream:
        stream = ai.chat_round_stream(prompt)
        await print_stream(stream)
        resp = (await stream.message()).text
    else:
        resp = await ai.chat_round_str(prompt)
        print(f"{type(engine).__name__}: {resp}")
    assert resp
