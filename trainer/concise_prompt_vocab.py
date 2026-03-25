"""Vocabulary for concise user-goal prompts: pickup phrasings and drink aliases.

Templates use a single ``{name}`` placeholder filled from per-class alias lists.
Stem convention (e.g. ``IMG_7006__lemon_tea__v01``): middle segment is the
canonical drink key in ``DRINK_NAME_ALIASES``.
"""

from __future__ import annotations

import hashlib
import itertools
import random
import re
from typing import Final

# Canonical keys embedded in dataset stems: ``<base>__<key>__vNN``.
KNOWN_DRINK_KEYS: Final[tuple[str, ...]] = ("sprite", "cola", "lemon_tea")

# Multiple natural names per class (lowercase phrases work inside sentences).
DRINK_NAME_ALIASES: dict[str, tuple[str, ...]] = {
    "sprite": (
        "Sprite",
        "Sprite drink",
        "Sprite soda",
        "the Sprite bottle",
        "the green Sprite",
        "lemon-lime soda",
        "a Sprite",
        "that Sprite",
        "Sprite soft drink",
        "the green bottle drink",
        "citrus soda",
        "chilled Sprite",
        "cold Sprite",
        "the Sprite can",
    ),
    "cola": (
        "cola",
        "Cola drink",
        "a cola",
        "the cola",
        "dark cola",
        "cola soda",
        "that cola bottle",
        "the red-label cola",
        "the cola can",
        "caramel cola",
        "iced cola",
        "cold cola",
        "the dark soda",
        "cola soft drink",
    ),
    "lemon_tea": (
        "lemon tea",
        "lemon tea drink",
        "iced lemon tea",
        "bottled lemon tea",
        "lemon iced tea",
        "citrus tea",
        "tea with lemon",
        "the lemon tea bottle",
        "yellow lemon tea",
        "that lemon tea",
        "lemon flavor tea",
        "cold lemon tea",
        "the tea drink",
        "sweet lemon tea",
        "Asian lemon tea",
        "lemon tea beverage",
    ),
}


def _build_pickup_templates() -> tuple[str, ...]:
    """At least 64 unique English pickup-intent lines with ``{name}``."""
    prefixes_i = (
        "I want to",
        "I'd like to",
        "I need to",
        "I'm trying to",
        "I wish to",
        "I'm hoping to",
        "I would like to",
        "I'm aiming to",
    )
    prefixes_help = (
        "Please help me",
        "Could you help me",
        "Help me",
        "I need help to",
    )
    verbs = (
        "pick up",
        "grab",
        "take",
        "get",
        "reach for",
        "fetch",
        "take hold of",
        "secure",
    )
    objects_i = (
        "{name}",
        "the {name}",
        "a {name}",
        "that {name}",
    )
    objects_help = (
        "{name}",
        "the {name}",
        "grab {name}",
    )

    out: list[str] = []
    for p, v, o in itertools.product(prefixes_i, verbs, objects_i):
        out.append(f"{p} {v} {o}.")

    for p in prefixes_help:
        for o in objects_help:
            if o.startswith("grab "):
                out.append(f"{p} {o}.")
            else:
                for v in verbs:
                    out.append(f"{p} {v} {o}.")

    # Dedupe while preserving order.
    unique = tuple(dict.fromkeys(out))
    return unique


PICKUP_GOAL_TEMPLATES: tuple[str, ...] = _build_pickup_templates()

# Flat dict: id -> template (for introspection / tests); >= 64 entries.
PICKUP_GOAL_TEMPLATES_DICT: dict[str, str] = {
    f"t{i:03d}": tpl for i, tpl in enumerate(PICKUP_GOAL_TEMPLATES)
}

if len(PICKUP_GOAL_TEMPLATES) < 64:
    raise RuntimeError("PICKUP_GOAL_TEMPLATES must contain at least 64 unique templates.")

_STEM_TAIL_RE = re.compile(r"^v\d+$", re.IGNORECASE)


def parse_drink_key_from_stem(stem: str) -> str | None:
    """Return drink key from ``...__<key>__vNN`` or None if not matched."""
    parts = stem.split("__")
    if len(parts) < 3:
        return None
    if not _STEM_TAIL_RE.match(parts[-1]):
        return None
    key = parts[-2]
    if key not in DRINK_NAME_ALIASES:
        return None
    return key


def _stem_hash_int(stem: str) -> int:
    return int(hashlib.md5(stem.encode("utf-8")).hexdigest(), 16)


def pick_template_and_name(stem: str, drink_key: str) -> tuple[str, str]:
    """Deterministic template and display name for a dataset stem."""
    h = _stem_hash_int(stem)
    templates = PICKUP_GOAL_TEMPLATES
    names = DRINK_NAME_ALIASES[drink_key]
    t_idx = h % len(templates)
    n_idx = (h // len(templates)) % len(names)
    return templates[t_idx], names[n_idx]


def build_user_goal_line(stem: str) -> str | None:
    """One-line user goal, or None if stem does not encode a known drink."""
    drink_key = parse_drink_key_from_stem(stem)
    if drink_key is None:
        return None
    tpl, name = pick_template_and_name(stem, drink_key)
    return tpl.format(name=name)


def build_random_user_goal_line(drink_key: str, rng: random.Random | None = None) -> str:
    """Randomly sampled one-line user goal for a given drink class.

    Args:
        drink_key: Canonical drink key (e.g. ``"cola"``).
        rng: Optional ``random.Random`` instance for reproducibility.
             Falls back to the module-level ``random`` if *None*.
    """
    if drink_key not in DRINK_NAME_ALIASES:
        raise ValueError(f"Unknown drink key: {drink_key!r}")
    _rng = rng or random
    tpl = _rng.choice(PICKUP_GOAL_TEMPLATES)
    name = _rng.choice(DRINK_NAME_ALIASES[drink_key])
    return tpl.format(name=name)


CONCISE_PROMPT_HEADER: Final[str] = (
    "You guide a blind user to grasp the drink they asked for. "
    "Reply with one short spoken command only (e.g. move left, grab now).\n\n"
    "User goal:\n"
)
