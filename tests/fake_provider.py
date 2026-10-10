
import asyncio


class FakeProvider:
    """A provider that returns scripted results or raises errors."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    async def generate(
        self,
        messages,
        model,
        temperature,
        max_tokens,
    ):
        self.calls += 1
        outcome = self.outcomes.pop(0)

        if isinstance(outcome, Exception):
            raise outcome

        return outcome


class SlowFakeProvider:
    """A provider that takes too long to respond."""

    async def generate(
        self,
        messages,
        model,
        temperature,
        max_tokens,
    ):
        await asyncio.sleep(1)
