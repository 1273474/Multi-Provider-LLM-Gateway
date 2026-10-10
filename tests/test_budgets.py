
import pytest

import app.budgets as budgets


@pytest.fixture(autouse=True)
def reset_budget_state():
    budgets.TENANT_USAGE.clear()
    budgets.RESERVATIONS.clear()


def test_cost_calculation():
    budgets.MODEL_PRICING["test/model"] = (
        1.0, 2.0
    )

    cost = budgets.calculate_cost(
        "test/model",
        input_tokens=1000,
        output_tokens=500,
    )

    assert cost == pytest.approx(0.002)


def test_reservation_respects_budget(monkeypatch):
    monkeypatch.setattr(
        budgets, "DEFAULT_BUDGET_USD", 0.01
    )
    budgets.MODEL_PRICING["test/model"] = (
        1.0, 1.0
    )

    reservation = budgets.reserve_budget(
        tenant_id="team-alpha",
        models=["test/model"],
        messages=[],
        max_tokens=5000,
    )

    assert reservation in budgets.RESERVATIONS

    with pytest.raises(budgets.BudgetExceededError):
        budgets.reserve_budget(
            tenant_id="team-alpha",
            models=["test/model"],
            messages=[],
            max_tokens=10000,
        )


def test_actual_cost_is_recorded():
    budgets.MODEL_PRICING["test/model"] = (
        1.0, 2.0
    )

    reservation = budgets.reserve_budget(
        tenant_id="team-alpha",
        models=["test/model"],
        messages=[],
        max_tokens=100,
    )

    budgets.settle_budget(
        reservation_id=reservation,
        model_key="test/model",
        input_tokens=1000,
        output_tokens=500,
    )

    usage = budgets.TENANT_USAGE["team-alpha"]

    assert usage["spent"] == pytest.approx(0.002)
    assert usage["reserved"] == pytest.approx(0.0)


def test_tenants_have_separate_budgets(monkeypatch):
    monkeypatch.setattr(
        budgets, "DEFAULT_BUDGET_USD", 0.01
    )
    budgets.MODEL_PRICING["test/model"] = (
        1.0, 1.0
    )

    budgets.reserve_budget(
        tenant_id="team-alpha",
        models=["test/model"],
        messages=[],
        max_tokens=5000,
    )

    # Another tenant has an independent allowance.
    reservation = budgets.reserve_budget(
        tenant_id="team-beta",
        models=["test/model"],
        messages=[],
        max_tokens=5000,
    )

    assert reservation in budgets.RESERVATIONS



import pytest

from app.budgets import (
    TENANT_USAGE,
    RESERVATIONS,
    reserve_budget,
    settle_budget,
    release_budget,
)


def test_settlement_cannot_charge_reservation_twice():
    tenant = "double-settlement-test"
    TENANT_USAGE.pop(tenant, None)

    messages = [
        type("Message", (), {"content": "Hello"})()
    ]

    reservation_id = reserve_budget(
        tenant_id=tenant,
        models=["mock/echo"],
        messages=messages,
        max_tokens=10,
    )

    settle_budget(
        reservation_id=reservation_id,
        model_key="mock/echo",
        input_tokens=5,
        output_tokens=3,
    )

    spent_after_first_settlement = TENANT_USAGE[tenant]["spent"]

    with pytest.raises(KeyError):
        settle_budget(
            reservation_id=reservation_id,
            model_key="mock/echo",
            input_tokens=5,
            output_tokens=3,
        )

    assert TENANT_USAGE[tenant]["spent"] == spent_after_first_settlement


def test_releasing_reservation_twice_is_safe():
    tenant = "double-release-test"
    TENANT_USAGE.pop(tenant, None)

    messages = [
        type("Message", (), {"content": "Hello"})()
    ]

    reservation_id = reserve_budget(
        tenant_id=tenant,
        models=["mock/echo"],
        messages=messages,
        max_tokens=10,
    )

    release_budget(reservation_id)
    release_budget(reservation_id)

    assert reservation_id not in RESERVATIONS
    assert TENANT_USAGE[tenant]["reserved"] == pytest.approx(0.0)


def test_settlement_with_unknown_reservation_fails():
    with pytest.raises(KeyError):
        settle_budget(
            reservation_id="does-not-exist",
            model_key="mock/echo",
            input_tokens=5,
            output_tokens=3,
        )
    
