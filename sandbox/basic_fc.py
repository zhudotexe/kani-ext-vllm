import asyncio
import enum
import logging
from typing import Annotated

from kani import AIParam, Kani, ai_function
from kani.ext.vllm import VLLMEngine
from kani.model_specific.gpt_oss import GPTOSSParser
from vllm import SamplingParams


class Unit(enum.Enum):
    FAHRENHEIT = "fahrenheit"
    CELSIUS = "celsius"


class MyKani(Kani):
    @ai_function()
    def get_weather(
        self,
        location: Annotated[str, AIParam(desc="The city and state, e.g. San Francisco, CA")],
        unit: Unit,
    ):
        """Get the current weather in a given location."""
        # call some weather API, or just mock it for this example
        degrees = 72 if unit == Unit.FAHRENHEIT else 22
        return f"Weather in {location}: Sunny, {degrees} degrees {unit.value}."


def get_engines():
    model = VLLMEngine(
        model_id="openai/gpt-oss-120b",
        max_context_size=128000,
        model_load_kwargs={"tensor_parallel_size": 8},
        sampling_params=SamplingParams(temperature=0.6, top_p=1, max_tokens=None),
    )
    yield GPTOSSParser(model)


async def main():
    for engine in get_engines():
        print(engine)
        ai = MyKani(engine)
        async for msg in ai.full_round("What's the weather in Tokyo and SF?"):
            print(msg)
        # await chat_in_terminal_async(ai, verbose=True, stream=False)
        await engine.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    asyncio.run(main())
