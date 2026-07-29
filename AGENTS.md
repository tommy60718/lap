# LAP Mission Router

For CoVer–Pi0.5–UR5e mission work in this nested repository, read the
canonical workflow at
[`../../docs/yangsen/MISSION_WORKFLOW.md`](../../docs/yangsen/MISSION_WORKFLOW.md)
before acting. The outer `osx_vla/AGENTS.md` is not guaranteed to be discovered
when Codex treats this nested Git repository as the project root.

Keep LAP implementation, environment, test, lint, and commit rules local to
this repository. Do not close an OSX-VLA issue from LAP tests alone; verify the
complete PRD and issue evidence contract.

For CUDA-dependent CoVer mission evidence, follow the sandbox/container
false-negative rule in `/home/yangsen/AGENTS.md`. Sandboxed
`torch.cuda.is_available() == False` is not admissible machine evidence.
Required GPU checks must demonstrate `/dev/nvidiactl`, `nvidia-smi`, and LAP's
own Python environment from the same confirmed host-level execution context.
If that context is unavailable, record GPU verification as pending; do not
diagnose the physical host as broken or attempt driver/device-node remediation.
