# The graph loop

This directory is the **computed scheduling layer** that built this repository. Instead of a
hand-written build plan, the work was declared once as a dependency graph of nodes
(`graph.json`) — each with a specification, a hard owned-paths boundary, and an acceptance
gate — and a small stdlib-only engine (`graphloop.py`) derived everything else:

- **what may run**: `frontier` lists nodes whose dependencies are all done;
- **what may run concurrently**: `check` proves that any two nodes that could run in the same
  wave own disjoint file paths (overlap is only legal along an ancestor edge in the DAG);
- **what each lane is told**: `emit NODE` renders a self-contained brief — spec, owned paths,
  gate, and the standing constraints restated (constraints do not travel implicitly);
- **what "done" means**: `gate NODE` runs the node's acceptance command and records the
  **true process exit** plus a UTC timestamp as evidence in `state.json`;
- **what "finished" means**: `complete` exits 0 only when every node is done. "Go to
  completion" is literally: loop until this command succeeds.

## The loop protocol

```
while ! graphloop.py complete:
    wave = graphloop.py frontier            # computed, not hand-picked
    for node in wave: graphloop.py start node
    dispatch each node as its own agent     # full tool environment, brief from `emit`
    wait for completion artifacts           # .graphloop/artifacts/<node>.json
    for node in wave: graphloop.py gate node   # orchestrator re-runs gates; true exit only
    graphloop.py log-wave                   # ledger entry
    commit                                  # state file + landed work, every wave boundary
    # failed gates -> fix in place or `reset` and re-dispatch with the failure evidence
```

Lanes communicate by artifacts, not narration: a lane's deliverable signal is its artifact
file, and the orchestrator trusts the gate's exit code — re-run by the orchestrator, never
reported by the lane — as the only completion proof.

## Files

| File | Role |
| --- | --- |
| `graph.json` | Node declarations: id, title, deps, owned paths, gate, spec. The only authored input. |
| `state.json` | Single source of truth: per-node status, gate evidence (cmd/exit/UTC), wave ledger. Orchestrator-written only. |
| `graphloop.py` | The engine. Stdlib-only so the scheduler itself has zero dependencies. |
| `artifacts/*.json` | Per-lane completion artifacts (and the review lane's findings). |
| `PLAN.md` | Product decisions and research notes behind the graph. |

Bootstrap note: the graph loop cannot schedule its own construction (lanes would need the
machinery that does not exist yet), so `bootstrap` was executed inline by a single writer and
then recorded as the first done node — the standard bootstrap exception.
