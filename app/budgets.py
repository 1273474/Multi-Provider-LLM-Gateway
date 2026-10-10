
import math
import os
import time
from threading import Lock
from uuid import uuid4


DEFAULT_BUDGET_USD = float(
    os.getenv("SENTINEL_DEFAULT_BUDGET_USD", "10")
)

# Prices are USD per 1 million tokens:
# (input-token price, output-token price).
# Add the actual prices for your chosen models before using them.
MODEL_PRICING = {
    "mock/echo": (0.0, 0.0),

    # USD per 1 million tokens: (input, output)
    "openai/gpt-4.1-mini": (0.40, 1.60),
    "anthropic/claude-sonnet-4-6": (3.00, 15.00),
    "groq/openai/gpt-oss-20b": (0.075, 0.30),
}

# tenant_id -> {"spent": float, "reserved": float}
TENANT_USAGE = {}

# reservation_id -> (tenant_id, estimated_cost)
RESERVATIONS = {}

_lock = Lock()


class BudgetExceededError(Exception):
    pass


def get_usage(tenant_id: str) -> dict:
    return TENANT_USAGE.setdefault(
        tenant_id,
        {"spent": 0.0, "reserved": 0.0},
    )


def calculate_cost(
    model_key: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    if model_key not in MODEL_PRICING:
        raise ValueError(
            f"Pricing is not configured for {model_key}"
        )

    input_price, output_price = MODEL_PRICING[model_key]

    return (
        input_tokens * input_price
        + output_tokens * output_price
    ) / 1_000_000


def reserve_budget(
    tenant_id: str,
    models: list[str],
    messages,
    max_tokens: int,
) -> str:
    # Rough estimate: about four characters per token.
    total_chars = sum(len(message.content) for message in messages)
    estimated_input_tokens = max(
        1, math.ceil(total_chars / 4)
    )

    # Reserve against the most expensive configured candidate.
    estimated_cost = max(
        calculate_cost(
            model,
            estimated_input_tokens,
            max_tokens,
        )
        for model in models
    )

    with _lock:
        usage = get_usage(tenant_id)

        if (
            usage["spent"]
            + usage["reserved"]
            + estimated_cost
            > DEFAULT_BUDGET_USD
        ):
            raise BudgetExceededError(
                f"Tenant {tenant_id} has insufficient budget"
            )

        reservation_id = str(uuid4())

        usage["reserved"] += estimated_cost
        RESERVATIONS[reservation_id] = (
            tenant_id,
            estimated_cost,
        )

        return reservation_id


def settle_budget(
    reservation_id: str,
    model_key: str,
    input_tokens: int,
    output_tokens: int,
) -> None:
    actual_cost = calculate_cost(
        model_key,
        input_tokens,
        output_tokens,
    )

    with _lock:
        tenant_id, reserved_cost = RESERVATIONS.pop(
            reservation_id
        )

        usage = get_usage(tenant_id)
        usage["reserved"] -= reserved_cost
        usage["spent"] += actual_cost


def release_budget(reservation_id: str) -> None:
    with _lock:
        reservation = RESERVATIONS.pop(
            reservation_id, None
        )

        if reservation is not None:
            tenant_id, reserved_cost = reservation
            usage = get_usage(tenant_id)
            usage["reserved"] -= reserved_cost
