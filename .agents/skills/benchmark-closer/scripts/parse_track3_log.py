#!/usr/bin/env python3
"""Parse modded-nanogpt Track 3 validation lines from run logs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


VAL_RE = re.compile(
    r"step:(?P<step>\d+)/(?P<total>\d+)\s+val_loss:(?P<loss>[0-9]+(?:\.[0-9]+)?)"
)


def parse_log(path: Path, target_loss: float) -> dict:
    points = []
    for line in path.read_text(errors="replace").splitlines():
        match = VAL_RE.search(line)
        if not match:
            continue
        points.append(
            {
                "step": int(match.group("step")),
                "total": int(match.group("total")),
                "loss": float(match.group("loss")),
            }
        )

    final = points[-1] if points else None
    first_target = next((point for point in points if point["loss"] <= target_loss), None)
    train_steps = final["total"] if final else None

    return {
        "path": str(path),
        "train_steps": train_steps,
        "final_step": final["step"] if final else None,
        "final_loss": final["loss"] if final else None,
        "first_step_at_or_below_target": first_target["step"] if first_target else None,
        "target_loss": target_loss,
        "beats_target": first_target is not None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path, help="Track 3 log files to parse")
    parser.add_argument("--target-loss", type=float, default=3.28)
    parser.add_argument("--indent", type=int, default=2)
    args = parser.parse_args()

    summaries = [parse_log(path, args.target_loss) for path in args.logs]
    output = summaries[0] if len(summaries) == 1 else summaries
    print(json.dumps(output, indent=args.indent, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
