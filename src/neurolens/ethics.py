"""Ethical guardrails.

Visible, named, runtime-enforced — not a footnote. Every iteration calls
`evaluate(scores, intent, prev_scores)` and acts on the returned `EthicsReport`:

  * Dark pattern detector — amygdala > 0.6 or NAcc > 0.7 without matching
    intent flags the design and refuses agent suggestions that would push it
    further.
  * Valence check — refuse to optimize toward high-arousal-negative states
    (high amygdala without compensating PFC/FFA).
  * Yerkes-Dodson ceiling — flag any single region above 0.85.
  * Trend check — if amygdala or NAcc *rose* this iteration without intent
    matching, raise a regression flag.

The flags drive both the agent prompt (so it can self-correct) and the
transparency report (so the designer sees what happened).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .reward import Intent

DARK_PATTERN_AMYG = 0.60
DARK_PATTERN_NACC = 0.70
AROUSAL_CEILING = 0.85

# Intents that legitimize NAcc activation
NACC_ALLOWED_INTENTS: set[Intent] = {"gamification"}
# Intents that legitimize amygdala activation (only emergency UIs)
AMYG_ALLOWED_INTENTS: set[Intent] = set()  # no UI intent we model excuses high amygdala


Severity = Literal["info", "warn", "block"]


@dataclass
class Flag:
    code: str
    severity: Severity
    message: str


@dataclass
class EthicsReport:
    flags: list[Flag] = field(default_factory=list)
    blocked: bool = False

    def add(self, code: str, severity: Severity, message: str) -> None:
        self.flags.append(Flag(code=code, severity=severity, message=message))
        if severity == "block":
            self.blocked = True

    def summary(self) -> str:
        if not self.flags:
            return "no ethical flags raised"
        return "\n".join(f"[{f.severity.upper()}] {f.code}: {f.message}" for f in self.flags)


def evaluate(
    scores: dict[str, float],
    intent: Intent,
    prev_scores: dict[str, float] | None = None,
) -> EthicsReport:
    rep = EthicsReport()

    amyg = scores.get("Amygdala", 0.0)
    nacc = scores.get("NAcc", 0.0)
    pfc = scores.get("PFC", 0.0)
    ffa = scores.get("FFA", 0.0)

    # Dark pattern: amygdala
    if amyg > DARK_PATTERN_AMYG and intent not in AMYG_ALLOWED_INTENTS:
        rep.add(
            "dark_pattern_amygdala",
            "block",
            f"Amygdala activation {amyg:.2f} > {DARK_PATTERN_AMYG} for intent={intent}. "
            "Likely dark pattern (anxiety / urgency). Agent will refuse edits that "
            "raise it further.",
        )

    # Dark pattern: NAcc outside gamification
    if nacc > DARK_PATTERN_NACC and intent not in NACC_ALLOWED_INTENTS:
        rep.add(
            "dark_pattern_nacc",
            "block",
            f"NAcc activation {nacc:.2f} > {DARK_PATTERN_NACC} for intent={intent}. "
            "Likely compulsive-design pattern. Agent will refuse edits that raise it.",
        )

    # Valence check: high arousal + low PFC/FFA = high-arousal-negative
    arousal = max(amyg, scores.get("Insula", 0.0), scores.get("ACC", 0.0))
    positive = (pfc + ffa) / 2.0
    if arousal > 0.7 and positive < 0.3:
        rep.add(
            "valence_negative",
            "warn",
            f"High-arousal-negative state detected (arousal={arousal:.2f}, "
            f"positive={positive:.2f}). Optimizer should not push further on this axis.",
        )

    # Yerkes-Dodson ceiling
    over = [r for r, s in scores.items() if s > AROUSAL_CEILING]
    if over:
        rep.add(
            "yerkes_ceiling",
            "warn",
            f"Region(s) over arousal ceiling {AROUSAL_CEILING}: {', '.join(over)}. "
            "Optimizer should target the engagement sweet spot, not infinite stimulation.",
        )

    # Trend regression
    if prev_scores is not None:
        d_amyg = amyg - prev_scores.get("Amygdala", amyg)
        d_nacc = nacc - prev_scores.get("NAcc", nacc)
        if d_amyg > 0.05 and intent not in AMYG_ALLOWED_INTENTS:
            rep.add(
                "amygdala_regression",
                "warn",
                f"Amygdala rose by {d_amyg:+.2f} this iteration — last edit may have "
                "introduced anxiety-coded design.",
            )
        if d_nacc > 0.05 and intent not in NACC_ALLOWED_INTENTS:
            rep.add(
                "nacc_regression",
                "warn",
                f"NAcc rose by {d_nacc:+.2f} this iteration outside a gamification "
                "intent — possible drift toward compulsive-design pattern.",
            )

    return rep
