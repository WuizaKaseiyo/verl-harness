compute_select skill — pick a compute target by probing host capabilities, optionally overridden by user preference.

## The three targets

| Target        | When applicable                                                                  | Launch mechanism                  |
|---------------|----------------------------------------------------------------------------------|-----------------------------------|
| `local-direct`| Host has CUDA-visible GPUs; no scheduler needed                                  | Bash the training script directly |
| `local-slurm` | Host is a Slurm login node (or has sbatch/srun in PATH)                          | `sbatch` from the agent's shell   |
| `ssh-slurm`   | Host has a configured ssh alias for a remote Slurm cluster's login node          | `ssh <alias> sbatch …`            |

## The probes

Always run all three probes; record results regardless of which target is finally chosen.

### gpu.access probe

```bash
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader 2>/dev/null
```

Pass iff the command returns ≥ 1 GPU row. Record: device count, names (e.g., `NVIDIA A100 80GB PCIe`), driver version, free memory.

For NPU targets, also probe `npu-smi info` and record similarly. (verl auto-detects torch_npu; the recipe doesn't need to differentiate.)

### slurm.access probe

```bash
sinfo --version          # must succeed
squeue -V                # must succeed
squeue --noheader | head # to confirm the daemon is reachable
sinfo --format='%P %a %l %D %N' --noheader | head -20
```

Pass iff `sinfo` and `squeue` are both on PATH and the daemon is reachable. Record: partition list, default partition.

### ssh.exec probe

The user (or the env var `$VERL_HARNESS_REMOTE`) supplies the ssh alias.

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 <alias> sinfo --version
```

Pass iff the ssh handshake succeeds non-interactively (key-based auth) and the remote returns a slurm version string. If it prompts for a password / passphrase, the probe fails (the harness must not interactively unlock auth).

## Selection rule

```
if compute_pref is explicit:
    if the corresponding probe passed → use it
    else → halt with "user requested X but X-capability is unavailable"

else (compute_pref == "auto"):
    # ranking — prefer the shortest / fastest feedback loop
    if local-direct probe passed and the host has ≥ recipe_min_gpu_count GPUs:
        use local-direct
    elif local-slurm probe passed:
        use local-slurm
    elif ssh-slurm probe passed:
        use ssh-slurm
    else:
        halt with "no compute target available"
```

`recipe_min_gpu_count` = `nodes × gpus_per_node` from `training_intent.md` (or the recipe's defaults if intent left them blank). If the host has the GPUs to run the job in a single-node `local-direct` configuration, prefer that — faster feedback than queueing.

## Slurm parameter resolution

For slurm targets, collect:

| sbatch directive   | Source                                                                  |
|--------------------|-------------------------------------------------------------------------|
| `--partition`      | `training_intent.md` `slurm.partition`; else ask the user (HITL)        |
| `--account`        | `training_intent.md` `slurm.account`; else ask                          |
| `--time`           | `slurm.time_limit`; else recipe estimate + 50% safety margin            |
| `--nodes`          | `nodes` from intent (or recipe default)                                 |
| `--ntasks-per-node`| 1 (Ray's head + worker model — one srun per node)                       |
| `--gpus-per-node`  | `gpus_per_node`                                                         |
| `--cpus-per-task`  | partition default × `gpus_per_node` / 8; recommend at least 8 × G       |
| `--mem`            | partition default; recommend ≥ 200 GB for 8-GPU training nodes          |
| `--output`         | `<output_dir>/slurm-%j.out`                                             |
| `--error`          | `<output_dir>/slurm-%j.err`                                             |

If `slurm.partition` is unset and HITL is allowed, ask the user — the partition list from the probe is the menu.

## Output: `workspace/compute/compute_choice.md`

```markdown
# Compute choice

## Target
local-direct                 # one of: local-direct | local-slurm | ssh-slurm

## Probe results
- gpu.access: PASS — 8 × NVIDIA A100 80GB PCIe, driver 535.86.05, 79.0 GiB free per GPU
- slurm.access: FAIL — sinfo not on PATH
- ssh.exec: SKIP — VERL_HARNESS_REMOTE not set

## Why this target
- compute_pref = auto; local-direct probe passed; host has 8 GPUs, recipe asks for 8 × 1-node → local-direct fits in one machine.

## Slurm parameters (omitted on local-direct)
- (not applicable)
```

For a slurm target, fill in the slurm-parameters block instead.

## Things you must not do

- Do not invent partition / account names. If `slurm.partition` is unset and the user is in `--no-hitl`, halt with a clear "no slurm partition specified".
- Do not silently downgrade `compute_pref`. Explicit user preference wins; if the matching capability is missing, halt.
- Do not probe by *running training* — the probes are read-only health checks (sinfo / nvidia-smi / ssh handshake).
