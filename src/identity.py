"""Student identity, personal seed rule and verification code."""
from __future__ import annotations

import re
from dataclasses import dataclass


class IdentityError(ValueError):
    """Raised when the student ID or seed breaks the assignment's seed rule."""


@dataclass(frozen=True)
class Identity:
    student_id: str         # e.g. ug2424671@ines.ac.rw
    reg_digits: str         # e.g. 2424671  (registration number without the slash)
    last_four: str          # e.g. "4671"   (kept as text so leading zeros survive)
    seed: int               # LLLL as an integer
    verification_code: str  # AI-A2-LLLL


def build_identity(student_id: str, seed: int) -> Identity:
    """Derive LLLL from the student ID and reject any seed that is not LLLL."""
    if not isinstance(student_id, str) or "@" not in student_id:
        raise IdentityError(
            "student-id must look like <username>@<domain>, e.g. ug2424671@ines.ac.rw"
        )
    username = student_id.split("@", 1)[0]
    digits = re.sub(r"\D", "", username)
    if len(digits) < 4:
        raise IdentityError(
            f"student-id username '{username}' must contain at least 4 digits."
        )
    last_four = digits[-4:]
    expected = int(last_four)
    if seed != expected:
        raise IdentityError(
            f"Seed {seed} is not allowed. Your seed must be the last four digits "
            f"of your registration number: {expected}."
        )
    return Identity(
        student_id=student_id,
        reg_digits=digits,
        last_four=last_four,
        seed=seed,
        verification_code=f"AI-A2-{last_four}",
    )
