from __future__ import annotations

import argparse
import json
from pathlib import Path

from focused_experiment import summarize_focused

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    paths = [Path(value) if Path(value).is_absolute() else HERE / value
             for value in args.inputs]
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    ignored = {"output", "datasets", "resume", "summarize"}
    reference = {key: value for key, value in payloads[0]["config"].items()
                 if key not in ignored}
    names = []
    records = []
    seen = set()
    for path, payload in zip(paths, payloads):
        config = {key: value for key, value in payload["config"].items()
                  if key not in ignored}
        if config != reference:
            raise ValueError(f"incompatible configuration: {path}")
        for record in payload["records"]:
            if record["instance"] in seen:
                raise ValueError(f"duplicate instance: {record['instance']}")
            seen.add(record["instance"])
            names.append(str(record["instance"]))
            records.append(record)
    merged = {
        "experiment": payloads[0]["experiment"],
        "config": dict(payloads[0]["config"]),
        "metric": payloads[0]["metric"],
        "records": records,
        "wall_seconds": sum(payload.get("wall_seconds", 0.0) for payload in payloads),
        "merged_sources": [str(path.relative_to(HERE)) for path in paths],
    }
    merged["config"]["datasets"] = ",".join(names)
    merged["config"]["resume"] = False
    merged["config"]["summarize"] = False
    out = Path(args.output)
    if not out.is_absolute():
        out = HERE / out
    out.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    summary = summarize_focused(merged)
    summary_out = out.with_name(out.stem + "_summary.json")
    summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "instances": len(records), "ranking": summary["quality_ranking"],
        "versus_random": summary["comparisons"],
        "versus_suspicious": summary["comparisons_vs_suspicious"],
    }, indent=2))


if __name__ == "__main__":
    main()
