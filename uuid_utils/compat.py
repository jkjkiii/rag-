"""Local fallback for uuid_utils.compat on environments missing native DLL deps."""

from __future__ import annotations

from datetime import datetime, timezone
from random import getrandbits
from uuid import UUID, uuid4


def uuid7(timestamp: int | None = None, nanos: int | None = None) -> UUID:
    """Best-effort UUIDv7-like fallback.

    If enough timing information is provided, encode millisecond time into UUID bits.
    Otherwise, return uuid4 for compatibility.
    """
    if timestamp is None and nanos is None:
        return uuid4()

    if timestamp is None and nanos is not None:
        ts_ms = nanos // 1_000_000
    elif timestamp is not None and nanos is None:
        ts_ms = int(timestamp) * 1000
    else:
        ts_ms = int(timestamp) * 1000 + int(nanos) // 1_000_000

    ts_ms &= (1 << 48) - 1
    rand_a = getrandbits(12)
    rand_b = getrandbits(62)

    value = 0
    value |= ts_ms << 80
    value |= 0x7 << 76
    value |= rand_a << 64
    value |= 0x2 << 62
    value |= rand_b

    return UUID(int=value)
