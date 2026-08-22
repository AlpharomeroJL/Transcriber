#!/usr/bin/env python3
"""Graph loop engine — the computed scheduling layer for this repository's build campaign.

The graph (graph.json) declares nodes of work with dependencies, owned paths and
acceptance gates. This engine derives everything else: which nodes may run, which
may run concurrently (file-disjointness is checked, not assumed), what each lane's
brief says, and whether the campaign is complete. State (state.json) is the single
source of truth and is written only by the orchestrator via this tool — never by lanes.

Commands:
  check                validate the DAG (cycles, unknown deps, owned-path conflicts)
  status               per-node status table
  frontier             ready nodes (all deps done, status pending) as JSON
  emit NODE            full dispatch brief for one node (markdown, self-contained)
  start NODE [...]     mark node(s) in_progress
  gate NODE            run the node's gate command; record true exit; done on 0
  fail NODE -r REASON  mark failed with a one-line reason
  reset NODE           failed/in_progress -> pending (for re-dispatch)
  log-wave -n NODES -m NOTE   append a wave entry to the ledger
  complete             exit 0 iff every node is done
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GRAPH = Path(__file__).resolve().parent / "graph.json"
STATE = Path(__file__).resolve().parent / "state.json"
ARTIFACTS = Path(__file__).resolve().parent / "artifacts"

STATUSES = ("pending", "in_progress", "done", "failed")


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_graph() -> dict:
    with open(GRAPH, encoding="utf-8") as f:
        g = json.load(f)
    ids = [n["id"] for n in g["nodes"]]
    if len(ids) != len(set(ids)):
        sys.exit("graph.json: duplicate node ids")
    return g


def load_state() -> dict:
    if STATE.exists():
        with open(STATE, encoding="utf-8") as f:
            return json.load(f)
    g = load_graph()
    return {
        "campaign": g["campaign"],
        "updated": utc(),
        "nodes": {n["id"]: {"status": "pending", "evidence": [], "reason": None} for n in g["nodes"]},
        "waves": [],
    }


def save_state(state: dict) -> None:
    state["updated"] = utc()
    STATE.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def sync_state(g: dict, state: dict) -> None:
    for n in g["nodes"]:
        state["nodes"].setdefault(n["id"], {"status": "pending", "evidence": [], "reason": None})


def node_map(g: dict) -> dict[str, dict]:
    return {n["id"]: n for n in g["nodes"]}


def ancestors(nid: str, nodes: dict[str, dict]) -> set[str]:
    seen: set[str] = set()
    stack = list(nodes[nid]["deps"])
    while stack:
        d = stack.pop()
        if d not in seen:
            seen.add(d)
            stack.extend(nodes[d]["deps"])
    return seen


def norm_pattern(p: str) -> str:
    return p.split("*", 1)[0]


def paths_conflict(a: list[str], b: list[str]) -> str | None:
    for pa in a:
        for pb in b:
            na, nb = norm_pattern(pa), norm_pattern(pb)
            if na.startswith(nb) or nb.startswith(na):
                return f"{pa!r} vs {pb!r}"
    return None


def cmd_check(_args: argparse.Namespace) -> int:
    g = load_graph()
    nodes = node_map(g)
    errors: list[str] = []
    for n in g["nodes"]:
        for d in n["deps"]:
            if d not in nodes:
                errors.append(f"{n['id']}: unknown dep {d!r}")
    # cycle detection via DFS colouring
    WHITE, GREY, BLACK = 0, 1, 2
    colour = {i: WHITE for i in nodes}

    def dfs(i: str, trail: list[str]) -> None:
        colour[i] = GREY
        for d in nodes[i]["deps"]:
            if d not in nodes:
                continue
            if colour[d] == GREY:
                errors.append(f"cycle: {' -> '.join(trail + [i, d])}")
            elif colour[d] == WHITE:
                dfs(d, trail + [i])
        colour[i] = BLACK

    for i in nodes:
        if colour[i] == WHITE:
            dfs(i, [])
    # owned-path conflicts between nodes that could run concurrently
    # (neither is an ancestor of the other in the DAG)
    anc = {i: ancestors(i, nodes) for i in nodes}
    ids = sorted(nodes)
    for x in range(len(ids)):
        for y in range(x + 1, len(ids)):
            a, b = ids[x], ids[y]
            if a in anc[b] or b in anc[a]:
                continue
            c = paths_conflict(nodes[a].get("owned", []), nodes[b].get("owned", []))
            if c:
                errors.append(f"owned-path conflict between concurrent nodes {a} and {b}: {c}")
    if errors:
        print("\n".join(errors))
        return 1
    print(f"graph OK: {len(nodes)} nodes, DAG acyclic, concurrent owned paths disjoint")
    return 0


def cmd_status(_args: argparse.Namespace) -> int:
    g, state = load_graph(), load_state()
    sync_state(g, state)
    width = max(len(n["id"]) for n in g["nodes"])
    done = 0
    for n in g["nodes"]:
        s = state["nodes"][n["id"]]
        done += s["status"] == "done"
        ev = s["evidence"][-1] if s["evidence"] else None
        ev_s = f"exit={ev['exit']} {ev['utc']}" if ev else "-"
        reason = f"  [{s['reason']}]" if s.get("reason") else ""
        print(f"{n['id']:<{width}}  {s['status']:<11}  deps={','.join(n['deps']) or '-':<30}  {ev_s}{reason}")
    print(f"-- {done}/{len(g['nodes'])} done; waves logged: {len(state['waves'])}")
    return 0


def ready_nodes(g: dict, state: dict) -> list[dict]:
    out = []
    for n in g["nodes"]:
        s = state["nodes"][n["id"]]
        if s["status"] != "pending":
            continue
        if all(state["nodes"][d]["status"] == "done" for d in n["deps"]):
            out.append(n)
    return out


def cmd_frontier(_args: argparse.Namespace) -> int:
    g, state = load_graph(), load_state()
    sync_state(g, state)
    ready = ready_nodes(g, state)
    print(json.dumps({
        "wave": len(state["waves"]) + 1,
        "nodes": [{"id": n["id"], "title": n["title"]} for n in ready],
    }, indent=2))
    return 0


STANDING = """\
## Standing constraints (restated in every brief; they do not travel implicitly)

- Repo root: /home/user/Transcriber. Toolchain: use `.venv/bin/python`, `.venv/bin/pytest`,
  `.venv/bin/ruff`, `.venv/bin/mypy` (the venv is pre-provisioned; do not pip-install anything).
- WRITE ONLY inside your owned paths listed above, plus your completion artifact
  `.graphloop/artifacts/<node-id>.json`. Never touch `.graphloop/graph.json`,
  `.graphloop/state.json`, or any other node's files. Read anything you like.
- Do not run `git commit`, `git push`, or edit git state; the orchestrator commits at wave boundaries.
- Code ships with its tests in the same lane. The gate below must pass with a TRUE process
  exit (run it verbatim) before you write your artifact. `rc=$?` after a pipe is not a true exit.
- Tests must be offline and deterministic: no network, no model downloads, no wall-clock or
  randomness dependence. Mock/fake external engines via injection, not by patching internals of
  third-party packages at a distance.
- Style: PEP 8 via ruff (run `.venv/bin/ruff format <owned files>` before finishing), full type
  annotations that pass the repo's mypy config, docstrings on public API. Match existing idioms.
- Never conclude a module, tool or suite "does not exist" from one failed call — check the repo
  first. If genuinely blocked, write your artifact with `"gate_exit": null` and a one-line
  `"blocked"` reason instead of fabricating completion.

## Completion artifact (this, not narration, is your deliverable signal)

Write `.graphloop/artifacts/<node-id>.json`:
{"node": "<id>", "summary": "<1-3 lines>", "files": ["<paths written>"],
 "gate_cmd": "<the gate>", "gate_exit": 0}
"""


def brief_for(n: dict) -> str:
    owned = "\n".join(f"- `{p}`" for p in n.get("owned", []))
    return (
        f"# Lane brief — node `{n['id']}`: {n['title']}\n\n"
        f"## Specification\n\n{n['spec']}\n\n"
        f"## Owned paths (hard boundary)\n\n{owned}\n\n"
        f"## Acceptance gate (run verbatim from repo root)\n\n```\n{n['gate']}\n```\n\n"
        f"{STANDING}"
    )


def cmd_emit(args: argparse.Namespace) -> int:
    g = load_graph()
    nodes = node_map(g)
    if args.node not in nodes:
        sys.exit(f"unknown node {args.node!r}")
    print(brief_for(nodes[args.node]))
    return 0


def set_status(ids: list[str], status: str, reason: str | None = None) -> int:
    g, state = load_graph(), load_state()
    sync_state(g, state)
    nodes = node_map(g)
    for i in ids:
        if i not in nodes:
            sys.exit(f"unknown node {i!r}")
        state["nodes"][i]["status"] = status
        state["nodes"][i]["reason"] = reason
    save_state(state)
    print(f"{','.join(ids)} -> {status}")
    return 0


def cmd_gate(args: argparse.Namespace) -> int:
    g, state = load_graph(), load_state()
    sync_state(g, state)
    nodes = node_map(g)
    if args.node not in nodes:
        sys.exit(f"unknown node {args.node!r}")
    gate = nodes[args.node]["gate"]
    print(f"[gate {args.node}] {gate}")
    proc = subprocess.run(gate, shell=True, cwd=ROOT, capture_output=True, text=True)
    tail = (proc.stdout + proc.stderr).strip().splitlines()[-15:]
    print("\n".join(tail))
    state["nodes"][args.node]["evidence"].append({"cmd": gate, "exit": proc.returncode, "utc": utc()})
    state["nodes"][args.node]["status"] = "done" if proc.returncode == 0 else "failed"
    if proc.returncode == 0:
        state["nodes"][args.node]["reason"] = None
    save_state(state)
    print(f"[gate {args.node}] exit={proc.returncode} -> {state['nodes'][args.node]['status']}")
    return proc.returncode


def cmd_log_wave(args: argparse.Namespace) -> int:
    state = load_state()
    state["waves"].append({"n": len(state["waves"]) + 1, "nodes": args.nodes.split(","), "utc": utc(), "note": args.message})
    save_state(state)
    print(f"wave {len(state['waves'])} logged")
    return 0


def cmd_complete(_args: argparse.Namespace) -> int:
    g, state = load_graph(), load_state()
    sync_state(g, state)
    pending = [i for i, s in state["nodes"].items() if s["status"] != "done"]
    if pending:
        print(f"incomplete: {','.join(pending)}")
        return 1
    print(f"campaign {state['campaign']} COMPLETE: {len(state['nodes'])}/{len(state['nodes'])} nodes done")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="graphloop")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    sub.add_parser("status")
    sub.add_parser("frontier")
    p = sub.add_parser("emit"); p.add_argument("node")
    p = sub.add_parser("start"); p.add_argument("nodes", nargs="+")
    p = sub.add_parser("gate"); p.add_argument("node")
    p = sub.add_parser("fail"); p.add_argument("node"); p.add_argument("-r", "--reason", required=True)
    p = sub.add_parser("reset"); p.add_argument("node")
    p = sub.add_parser("log-wave"); p.add_argument("-n", "--nodes", required=True); p.add_argument("-m", "--message", required=True)
    sub.add_parser("complete")
    args = ap.parse_args()
    ARTIFACTS.mkdir(exist_ok=True)
    if args.cmd == "check":
        return cmd_check(args)
    if args.cmd == "status":
        return cmd_status(args)
    if args.cmd == "frontier":
        return cmd_frontier(args)
    if args.cmd == "emit":
        return cmd_emit(args)
    if args.cmd == "start":
        return set_status(args.nodes, "in_progress")
    if args.cmd == "gate":
        return cmd_gate(args)
    if args.cmd == "fail":
        return set_status([args.node], "failed", args.reason)
    if args.cmd == "reset":
        return set_status([args.node], "pending")
    if args.cmd == "log-wave":
        return cmd_log_wave(args)
    if args.cmd == "complete":
        return cmd_complete(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
