"""Transparency report.

After each iteration, write a plain-language summary capturing:

  * which regions changed and by how much
  * which edit was applied and why the agent chose it
  * any ethics flags raised
  * whether the reward improved

Two outputs:

  * `iteration_record(...)` returns a dict (JSON-serializable) — appended to
    a per-run JSONL log
  * `render_run(...)` produces a Markdown summary at the end of a run

Designed so the designer (and a judge) can read what happened without
opening the code.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .agent import EditChoice
from .ethics import EthicsReport
from .reward import Intent, RewardBreakdown


def iteration_record(
    iteration: int,
    intent: Intent,
    scores: dict[str, float],
    prev_scores: dict[str, float] | None,
    reward: RewardBreakdown,
    edit: EditChoice | None,
    ethics: EthicsReport,
) -> dict[str, Any]:
    deltas: dict[str, float] = {}
    if prev_scores is not None:
        deltas = {r: round(scores[r] - prev_scores.get(r, 0.0), 4) for r in scores}
    return {
        "iteration": iteration,
        "intent": intent,
        "scores": {k: round(v, 4) for k, v in scores.items()},
        "deltas": deltas,
        "reward_total": round(reward.total, 4),
        "reward_targets": {k: round(v, 4) for k, v in reward.targets.items()},
        "reward_penalties": {k: round(v, 4) for k, v in reward.penalties.items()},
        "yerkes_violations": list(reward.yerkes_violations),
        "edit": (
            {"name": edit.edit, "params": edit.params, "rationale": edit.rationale}
            if edit is not None
            else None
        ),
        "ethics_flags": [asdict(f) for f in ethics.flags],
        "ethics_blocked": ethics.blocked,
    }


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def render_run(records: list[dict[str, Any]], out_path: Path) -> None:
    if not records:
        out_path.write_text("# NeuroUI run\n\n_No iterations recorded._\n")
        return

    first = records[0]
    last = records[-1]
    intent = first["intent"]
    delta_total = last["reward_total"] - first["reward_total"]

    md: list[str] = [
        "# NeuroUI Optimizer — Run Report",
        "",
        f"**Intent:** {intent}  ",
        f"**Iterations:** {len(records)}  ",
        f"**Reward delta:** {first['reward_total']:+.3f} → {last['reward_total']:+.3f} "
        f"({delta_total:+.3f})  ",
        "",
        "## Region trajectories",
        "",
        "| Region | Start | End | Δ |",
        "|--------|------:|----:|--:|",
    ]
    regions = list(first["scores"].keys())
    for r in regions:
        s0 = first["scores"][r]
        s1 = last["scores"][r]
        md.append(f"| {r} | {s0:.2f} | {s1:.2f} | {s1 - s0:+.2f} |")

    md += ["", "## Iteration log", ""]
    for rec in records:
        edit = rec["edit"]
        md.append(f"### Iteration {rec['iteration']}")
        if edit:
            md.append(f"- **Edit:** `{edit['name']}` {edit['params'] or ''}")
            md.append(f"- **Why:** {edit['rationale']}")
        else:
            md.append("- **Edit:** _none (baseline)_")
        md.append(f"- **Reward:** {rec['reward_total']:+.3f}")
        if rec["yerkes_violations"]:
            md.append(f"- **Yerkes ceiling exceeded:** {', '.join(rec['yerkes_violations'])}")
        if rec["ethics_flags"]:
            md.append("- **Ethics flags:**")
            for f in rec["ethics_flags"]:
                md.append(f"    - `[{f['severity']}] {f['code']}` — {f['message']}")
        md.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(md))
