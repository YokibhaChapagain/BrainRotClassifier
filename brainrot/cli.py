"""Command line interface.

  brainrot classify video.mp4 [more.mp4 | folder/ | https://youtube.com/...]
  brainrot eval labels.csv          # CSV with columns: source,label  (label: brain_rot / not_brain_rot)
  brainrot serve                    # web UI on http://localhost:8000
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

from .config import Settings

VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v", ".mpeg", ".mpg", ".3gp", ".flv", ".wmv"}
BAR = 20


def _expand(inputs: list[str]) -> list[str]:
    out = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            out += sorted(str(f) for f in p.rglob("*") if f.suffix.lower() in VIDEO_EXTS)
        else:
            out.append(item)
    return out


def _settings(args: argparse.Namespace) -> Settings:
    s = Settings()
    for name in ("model", "mode", "threshold", "max_frames", "max_seconds"):
        val = getattr(args, name, None)
        if val is not None:
            setattr(s, name, val)
    if getattr(args, "fps", None) is not None:
        s.native_fps = args.fps
    return s


def _print_result(r) -> None:
    v = r.verdict
    colour = "\033[91m" if r.is_brain_rot else "\033[92m"
    reset = "\033[0m" if sys.stdout.isatty() else ""
    colour = colour if sys.stdout.isatty() else ""
    print(f"\n{'=' * 70}\n{r.source}")
    print(f"{colour}{r.label.upper().replace('_', ' ')}{reset}  score {r.brain_rot_score}/100 "
          f"(threshold {r.threshold}, confidence {r.confidence:.0%})  [{r.mode}, {r.model}, {r.elapsed_s}s]")
    print(f"\nSummary: {v.content_summary}")
    print("\nCriteria (0-10):")
    for name, score in v.criteria.model_dump().items():
        print(f"  {name:<34} {'#' * (score * BAR // 10):<{BAR}} {score}")
    if v.detected_tropes:
        print("\nTropes: " + "; ".join(v.detected_tropes))
    if v.redeeming_qualities:
        print("Redeeming: " + "; ".join(v.redeeming_qualities))
    if r.signals:
        s = r.signals
        print(f"\nSignals: {s.duration_s:.0f}s, {s.cuts_per_minute} cuts/min, motion {s.mean_motion}, "
              f"{'vertical' if s.is_vertical else 'horizontal'}, volume {s.mean_volume_db} dB")
    print(f"\nReasoning: {v.reasoning}")


def cmd_classify(args: argparse.Namespace) -> int:
    from .classifier import BrainRotClassifier

    clf = BrainRotClassifier(_settings(args))
    sources = _expand(args.inputs)
    if not sources:
        print("No videos found.", file=sys.stderr)
        return 1
    failures = 0
    out = open(args.output, "a") if args.output else None
    for src in sources:
        try:
            r = clf.classify(src, context=args.context or "", download=args.download)
        except Exception as e:
            failures += 1
            logging.error("Failed on %s: %s", src, e)
            continue
        if args.json:
            print(r.model_dump_json(indent=2))
        else:
            _print_result(r)
        if out:
            out.write(r.model_dump_json() + "\n")
    if out:
        out.close()
    return 1 if failures == len(sources) else 0


def cmd_eval(args: argparse.Namespace) -> int:
    from .classifier import BrainRotClassifier

    clf = BrainRotClassifier(_settings(args))
    with open(args.csv, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("source")]
    base = Path(args.csv).parent
    tp = fp = tn = fn = errors = 0
    results = []
    for row in rows:
        src = row["source"].strip()
        if not src.startswith("http") and not Path(src).is_absolute():
            src = str(base / src)
        truth = row["label"].strip().lower() in {"brain_rot", "brainrot", "1", "yes", "true"}
        try:
            r = clf.classify(src, context=row.get("context", "") or "", download=args.download)
        except Exception as e:
            errors += 1
            logging.error("Failed on %s: %s", src, e)
            continue
        pred = r.is_brain_rot
        tp += pred and truth
        fp += pred and not truth
        tn += (not pred) and (not truth)
        fn += (not pred) and truth
        mark = "OK " if pred == truth else "XX "
        print(f"{mark} score={r.brain_rot_score:>3} pred={r.label:<14} truth={'brain_rot' if truth else 'not_brain_rot':<14} {src}")
        results.append({"source": src, "truth": truth, **json.loads(r.model_dump_json())})

    n = tp + fp + tn + fn
    if n:
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        print(f"\n{n} videos ({errors} errors) | accuracy {(tp + tn) / n:.1%} | precision {precision:.1%} "
              f"| recall {recall:.1%} | F1 {f1:.2f}")
        print(f"confusion: TP={tp} FP={fp} TN={tn} FN={fn}")
    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2))
        print(f"Wrote detailed results to {args.output}")
    return 0 if n else 1


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("brainrot.web.server:app", host=args.host, port=args.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="brainrot", description="Classify videos as brain rot (or not) with Gemini.")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--mode", choices=["native", "frames"], help="native = send whole video (default); frames = keyframes + audio")
        p.add_argument("--model", help="Gemini model id (default from GEMINI_MODEL)")
        p.add_argument("--threshold", type=int, help="score >= threshold => brain rot (default 50)")
        p.add_argument("--max-frames", dest="max_frames", type=int, help="keyframes in frames mode (default 16)")
        p.add_argument("--max-seconds", dest="max_seconds", type=float, help="only analyse first N seconds (0 = all; default 180)")
        p.add_argument("--fps", type=float, help="frames/sec Gemini samples in native mode (default 2)")
        p.add_argument("--download", action="store_true", help="download YouTube URLs with yt-dlp instead of passing the URL to Gemini")
        p.add_argument("-o", "--output", help="write results (JSONL for classify, JSON for eval)")

    p = sub.add_parser("classify", help="classify one or more videos / folders / URLs")
    p.add_argument("inputs", nargs="+")
    p.add_argument("--context", help="extra context, e.g. the caption/hashtags of the post")
    p.add_argument("--json", action="store_true", help="print raw JSON")
    common(p)
    p.set_defaults(func=cmd_classify)

    p = sub.add_parser("eval", help="measure accuracy on a labelled CSV (columns: source,label[,context])")
    p.add_argument("csv")
    common(p)
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("serve", help="start the web UI")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        return args.func(args)
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
