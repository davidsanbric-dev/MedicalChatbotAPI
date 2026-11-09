from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request

from app.core import rate_limit
from app.core.rate_limit import SlidingWindowLimiter, client_ip


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def make_request(peer="10.0.0.9", forwarded=None):
    headers = [(b"x-forwarded-for", forwarded.encode())] if forwarded else []
    return Request({"type": "http", "headers": headers, "client": (peer, 1234)})


def test_sliding_window_blocks_then_recovers():
    clock = FakeClock()
    limiter = SlidingWindowLimiter(2, 60, "slow down", clock=clock)
    for _ in range(2):
        assert limiter.retry_after("a") == 0
        limiter.hit("a")
    assert limiter.retry_after("a") == 60
    assert limiter.retry_after("b") == 0  # other clients are unaffected
    clock.now += 45
    assert limiter.retry_after("a") == 15
    clock.now += 15
    assert limiter.retry_after("a") == 0


def test_zero_limit_disables():
    limiter = SlidingWindowLimiter(0, 60, "off", clock=FakeClock())
    for _ in range(100):
        limiter.hit("a")
    assert limiter.retry_after("a") == 0


def test_sweep_forgets_idle_clients():
    clock = FakeClock()
    limiter = SlidingWindowLimiter(5, 60, "x", clock=clock)
    limiter.hit("a")
    clock.now += 61
    limiter.sweep()
    assert limiter._hits == {}


def test_client_ip_ignores_forwarded_header_without_trusted_proxies():
    request = make_request(forwarded="6.6.6.6")
    assert client_ip(request, trusted_hops=0) == "10.0.0.9"


def test_client_ip_reads_through_trusted_proxies():
    # browser 1.2.3.4 -> Traefik 172.18.0.2 -> nginx 10.0.0.9 -> API
    request = make_request(forwarded="1.2.3.4, 172.18.0.2")
    assert client_ip(request, trusted_hops=2) == "1.2.3.4"


def test_client_ip_ignores_forged_entries_left_of_the_proxies():
    request = make_request(forwarded="6.6.6.6, 1.2.3.4, 172.18.0.2")
    assert client_ip(request, trusted_hops=2) == "1.2.3.4"


def test_client_ip_with_shorter_chain_falls_back_to_leftmost():
    request = make_request(forwarded="1.2.3.4")
    assert client_ip(request, trusted_hops=2) == "1.2.3.4"


def test_limit_chat_returns_429_with_retry_after(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(rate_limit, "_per_client", [SlidingWindowLimiter(2, 60, "per client", clock=clock)])
    monkeypatch.setattr(rate_limit, "_global", SlidingWindowLimiter(3, 86400, "global", clock=clock))

    app = FastAPI()

    @app.post("/chat/", dependencies=[Depends(rate_limit.limit_chat)])
    async def chat():
        return {"ok": True}

    client = TestClient(app)
    assert [client.post("/chat/").status_code for _ in range(2)] == [200, 200]

    blocked = client.post("/chat/")
    assert blocked.status_code == 429
    assert blocked.json() == {"detail": "per client"}
    assert blocked.headers["Retry-After"] == "60"

    # A rejected request uses no quota: after the window only the global cap
    # (3 per day, 2 used) remains, so exactly one more request gets through.
    clock.now += 60
    assert client.post("/chat/").status_code == 200
    blocked = client.post("/chat/")
    assert blocked.status_code == 429
    assert blocked.json() == {"detail": "global"}
