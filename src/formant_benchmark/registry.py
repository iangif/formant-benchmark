"""Small typed registry used by extensible dataset and tracker interfaces."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Generic, TypeVar

from formant_benchmark.exceptions import DuplicateRegistrationError, UnknownRegistrationError

T = TypeVar("T")


class Registry(Generic[T]):
    """Map stable string names to extension objects without central conditionals."""

    def __init__(self) -> None:
        self._entries: dict[str, T] = {}

    def register(self, name: str, value: T) -> T:
        """Register *value* under *name* and reject accidental replacement."""
        normalized = name.strip()
        if not normalized:
            raise ValueError("Registry names must be non-empty.")
        if normalized in self._entries:
            raise DuplicateRegistrationError(f"'{normalized}' is already registered.")
        self._entries[normalized] = value
        return value

    def get(self, name: str) -> T:
        """Return a registered value or raise a domain-specific lookup error."""
        try:
            return self._entries[name]
        except KeyError as exc:
            raise UnknownRegistrationError(f"Unknown registry entry: '{name}'.") from exc

    def names(self) -> tuple[str, ...]:
        """Return deterministic registry names."""
        return tuple(sorted(self._entries))

    def values(self) -> Iterable[T]:
        """Iterate values in deterministic name order."""
        return (self._entries[name] for name in self.names())
