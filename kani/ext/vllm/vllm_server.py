import asyncio
import logging
import socket
import subprocess

import httpx

log = logging.getLogger(__name__)


class VLLMServer:
    """Class to manage a vLLM server instance."""

    def __init__(self, model_id: str, vllm_args: dict = None, *, host: str = "127.0.0.1", port: int = None):
        """
        Launch a vLLM server instance serving the given model ID with the given CLI arguments.

        Optionally, the host and port can be overridden, but defaults to localhost and a random free port.
        """
        if vllm_args is None:
            vllm_args = {}
        # launch the server
        if port is None:
            port = str(get_free_port())
        else:
            port = str(port)
        self.port = port
        self.host = host
        _vargs = [
            "vllm",
            "serve",
            model_id,
            "--host",
            host,
            "--port",
            port,
            *kwargs_to_cli(vllm_args),
        ]
        log.info(f"Launching vLLM server with following command: {_vargs}")
        self.process = subprocess.Popen(_vargs)
        self.http = httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}")
        self._server_healthy = False

    async def wait_for_healthy(self, route="/health"):
        """Wait until a GET request to the given route returns a 2XX response code."""
        # we can early return if the server has been healthy once, hopefully
        if self._server_healthy:
            return
        log.info("Waiting until vLLM server is healthy...")
        i = 0
        while not self._server_healthy:
            i += 1
            try:
                log.debug(f"Checking for healthy server (request {i})...")
                resp = await self.http.get(route)
                resp.raise_for_status()
            except httpx.HTTPError as e:
                log.debug(f"Unhealthy server (request {i}), waiting for 5 seconds...", exc_info=e)
                if i % 10 == 0 and i > 1:
                    log.warning(f"vLLM server is still unhealthy after {i} health checks!")
                await asyncio.sleep(5)
                continue
            else:
                self._server_healthy = 200 <= resp.status_code < 300

    def close(self):
        self.process.terminate()


# ==== utils ====
def kwargs_to_cli(args: dict) -> list[str]:
    """
    Convert vLLM engine-style kwargs to CLI-style kwargs.

    Example: {"tensor_parallel_size": 3} -> ["--tensor-parallel-size", "3"]
    """
    out = []
    for k, v in args.items():
        if v is False:
            continue
        k = f"--{k.replace('-', '_')}"
        out.append(k)
        if v is not True:
            out.append(str(v))

    return out


def get_free_port() -> int:
    """Return a random free port on the host."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    port = s.getsockname()[1]
    s.close()
    return port
