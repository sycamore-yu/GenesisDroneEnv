from typing import Any, Protocol


class TaskCore(Protocol):
    """Task semantics only: observations, rewards, terminations. No Scene / dynamics."""

    def reset_task(self, *args: Any, **kwargs: Any) -> None: ...

    def observe(self, state: Any) -> tuple[Any, Any]:
        """Return (policy_obs, critic_obs)."""
        ...

    def evaluate(
        self,
        state_before: Any,
        state_after: Any,
        action: Any,
        is_alive_before: Any,
        **kwargs: Any,
    ) -> Any:
        """Return reward, physics_loss, policy_loss, terminated, truncated, events, metrics."""
        ...
