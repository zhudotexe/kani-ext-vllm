"""
What is each LLM's most likely conversation?
"""

import asyncio
import sys
import uuid

from vllm import AsyncEngineArgs, SamplingParams
from vllm.v1.engine.async_llm import AsyncLLM


async def main(model_id):
    engine_args = AsyncEngineArgs(model=model_id)
    engine = AsyncLLM.from_engine_args(engine_args)

    request_id = str(uuid.uuid4())
    sampling_params = SamplingParams(
        top_k=1,
        max_tokens=2048,
        ignore_eos=True,
    )
    final_output = None
    try:
        async for request_output in engine.generate(prompt="", sampling_params=sampling_params, request_id=request_id):
            final_output = request_output
    except (asyncio.CancelledError, KeyboardInterrupt):
        # if something cancels our task, make sure we tell vLLM to stop generating too
        await engine.abort(request_id)
        raise

    print(final_output.outputs[0].text)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
