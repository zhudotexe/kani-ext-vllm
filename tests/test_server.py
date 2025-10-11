import asyncio

from kani.ext.vllm.vllm_server import VLLMServer
from tests.conftest import MODEL_ID


async def test_server_startup():
    # the server should be able to start in < 5 minutes
    server = VLLMServer(model_id=MODEL_ID, vllm_args={"gpu_memory_utilization": 0.1})
    await asyncio.wait_for(server.wait_for_healthy(), timeout=300)
    # and the model should be available
    resp = await server.http.get("/v1/models")
    data = resp.json()
    print(data)
    assert data["data"][0]["id"] == MODEL_ID
    await server.close()
