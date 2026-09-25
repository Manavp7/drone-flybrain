from __future__ import annotations

import argparse
from dataclasses import asdict
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Industrial inspection simulation. No aircraft/hardware control.")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("evaluate", help="Run paired seeded episodes and save raw results")
    p.add_argument("--scenarios", type=int, default=100)
    p.add_argument("--seed", type=int, default=100000)
    p.add_argument("--variants", nargs="+", choices=["baseline", "bio_proxy"], default=["baseline", "bio_proxy"])
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--output", type=Path, default=Path("results/new_run"))
    p.add_argument("--replays", type=int, default=32)
    p = sub.add_parser("replay", help="Reproduce one scenario with a complete trajectory")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--variant", choices=["baseline", "bio_proxy"], default="baseline")
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("report", help="Build a standalone interactive HTML report")
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--template", type=Path, default=Path("web/index.html"))
    p.add_argument("--output", type=Path, default=Path("Inspection_Simulation_Report.html"))
    p = sub.add_parser("compare", help="Compare two frozen conventional-controller releases")
    p.add_argument("--original", type=Path, required=True)
    p.add_argument("--revised", type=Path, required=True)
    p.add_argument("--original-source", type=Path, required=True)
    p.add_argument("--revised-source", type=Path, default=Path.cwd())
    p.add_argument("--integration-status", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("serve", help="Serve local reports on loopback; read-only static files")
    p.add_argument("--directory", type=Path, default=Path.cwd())
    p.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if args.command == "evaluate":
        from .evaluation import evaluate
        result = evaluate(args.output, args.scenarios, args.seed, args.variants, args.workers, args.replays)
        print(json.dumps({"overall": result["overall"], "paired": result["paired"]}, indent=2))
    elif args.command == "replay":
        from .runner import run_episode
        from .scenarios import generate_scenario
        from .evaluation import write_json
        if args.output.exists():
            raise FileExistsError(f"Refusing to replace {args.output}")
        s = generate_scenario(args.seed)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, {"id": f"{args.seed}-{args.variant}", "scenario": asdict(s),
                                 "result": asdict(run_episode(s, args.variant, record=True))})
    elif args.command == "compare":
        from .comparison import compare_releases
        status = json.loads(args.integration_status.read_text()) if args.integration_status else []
        result = compare_releases(args.original, args.revised, args.original_source,
                                  args.revised_source, args.output, status)
        print(json.dumps({"overall": result["overall"], "paired": result["paired"]}, indent=2))
    elif args.command == "report":
        from .evaluation import build_report
        build_report(args.results, args.template, args.output)
        print(args.output.resolve())
    else:
        directory = args.directory.resolve()
        handler = partial(SimpleHTTPRequestHandler, directory=str(directory))
        server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
        print(f"Local report: http://127.0.0.1:{args.port}/Inspection_Simulation_Report.html", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()


if __name__ == "__main__":
    main()
