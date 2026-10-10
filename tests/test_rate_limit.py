
import pytest
from fastapi import HTTPException

import app.rate_limit as rate_limit


def test_requests_within_limit_are_allowed(monkeypatch):
    monkeypatch.setattr(rate_limit, "RATE_LIMIT", 2)
    rate_limit.REQUEST_COUNTS.clear()

    rate_limit.enforce_rate_limit("team-alpha")
    rate_limit.enforce_rate_limit("team-alpha")


def test_excess_requests_are_rejected(monkeypatch):
    monkeypatch.setattr(rate_limit, "RATE_LIMIT", 2)
    rate_limit.REQUEST_COUNTS.clear()

    rate_limit.enforce_rate_limit("team-alpha")
    rate_limit.enforce_rate_limit("team-alpha")

    with pytest.raises(HTTPException) as exc_info:
        rate_limit.enforce_rate_limit("team-alpha")

    assert exc_info.value.status_code == 429


def test_tenants_have_separate_limits(monkeypatch):
    monkeypatch.setattr(rate_limit, "RATE_LIMIT", 1)
    rate_limit.REQUEST_COUNTS.clear()

    rate_limit.enforce_rate_limit("team-alpha")
    rate_limit.enforce_rate_limit("team-beta")

    with pytest.raises(HTTPException) as exc_info:
        rate_limit.enforce_rate_limit("team-alpha")

    assert exc_info.value.status_code == 429
