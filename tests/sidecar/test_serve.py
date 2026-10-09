import logging
import threading
import time

import httpx
import uvicorn

from inspect_sentinel.sidecar import Handler
from inspect_sentinel.sidecar._proxy_endpoint import proxy_app
from inspect_sentinel.sidecar._serve import _HealthCheckFilter, _quiet_health_checks
from tests.sidecar._traffic import weather_rule


class _Lines(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(record.getMessage())


def test_health_checks_stay_out_of_the_access_log() -> None:
    # a real server, so the test reads the access log as uvicorn writes it
    server = uvicorn.Server(uvicorn.Config(proxy_app(Handler(weather_rule())), port=0))
    access = logging.getLogger("uvicorn.access")
    lines = _Lines()
    access.addHandler(lines)
    _quiet_health_checks()
    _quiet_health_checks()
    serving = threading.Thread(target=server.run)
    serving.start()
    try:
        while not server.started:
            time.sleep(0.01)
        port = server.servers[0].sockets[0].getsockname()[1]
        with httpx.Client(base_url=f"http://127.0.0.1:{port}") as client:
            checks = [client.get("/health"), client.get("/health?probe=1")]
            other = [client.post("/", content=b"{}"), client.get("/?next=/health")]
    finally:
        server.should_exit = True
        serving.join()
        access.removeHandler(lines)
        added = [
            kept for kept in access.filters if isinstance(kept, _HealthCheckFilter)
        ]
        for kept in added:
            access.removeFilter(kept)

    assert [answered.status_code for answered in checks] == [200, 200]
    assert [answered.status_code for answered in other] == [400, 405]
    # asking twice for quiet adds one filter
    assert len(added) == 1
    assert len(lines.lines) == 2
    assert '"POST / HTTP/1.1" 400' in lines.lines[0]
    assert '"GET /?next=/health HTTP/1.1" 405' in lines.lines[1]
