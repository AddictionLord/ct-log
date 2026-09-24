# Training agents: running and tracking ct-log experiments

Operating manual for an agent that launches, monitors and reports ct-log training runs,
often while the user is away and steering through Claude Remote Control. Read it fully
before starting anything. When in doubt, ask the user instead of guessing.

## Ground rules

- Run everything from the repo root (`/home/mary/code/ct-log`) with `python -m ...`.
- **Never print secrets.** `.env` holds `SUPERVISELY_TOKEN` and `MLFLOW_TRACKING_PASSWORD`;
  `~/.bashrc` holds `EULER_TOKEN`. Check presence with `test -n "$VAR"`, never `echo`/`env`/`cat`.
  Do not `cat` raw HTTP responses: Jupyter HTML pages embed the auth token.
- **Everything logged to MLflow is public** (the DagsHub repo is public). No secrets or
  personal data in params, tags or artifacts.
- One training run on the local GPU at a time (4 GB). Check `nvidia-smi` before launching.
- Ask the user before: deleting runs/models, provisioning euler, committing config
  changes, or anything that costs DagsHub storage outside the normal logger.
- Report with numbers (epoch, loss, `mean_iou_fg`, knot IoU, pith px error), not adjectives.

## Where training runs

| Host | GPU | Status |
|---|---|---|
| Local machine (this repo) | GTX 1650, 4 GB | **Default.** conda env `ct-log`, torch 2.7.0+cu128. Data in `/mnt/D/datasets/ct_log`, weights in `/mnt/D/models`. |
| euler.mendelu.cz | RTX 5070 Ti, 16 GB | Provisioned 2026-09-24 under `~/ctlog-eval/` (copied working tree, data, DINOv3 weights). Jobs run via `scripts/euler.py bg` in `~/jobs/<name>/`. Use `num_workers=0`, batch 8 fits. See [euler](#euler-remote-gpu). |

## Launching (local)

Use the supervisor: it sources `.env` (MLflow credentials), resumes after crashes
(configs with `resume: true` write `<checkpoint>.resume.pth` every epoch) and stops once the
run prints its completion marker `Best foreground mean IoU`.

```bash
nvidia-smi --query-compute-apps=pid,used_memory --format=csv   # must be empty
RUN=kwp_v1_<short_description>
nohup bash scripts/supervise_kwp.sh src/configs/train_kwp_v1.yaml "$RUN" "logs/$RUN" \
    > /dev/null 2>&1 &
```

- Other entry points: `scripts/run_kwp_sweep.sh stage1|stage2` (sequential queue),
  `scripts/await_data_and_train.sh [HH:MM] [min_logs]` (waits for generated data, then trains).
- CLI overrides of `src.train_kwp`: `--config --run_name --num_epochs --window --n_layers
  --checkpoint_path --local_log_dir --lr_schedule --train_logs --val_logs --resume`.
- Direct run without the supervisor must load `.env` first, or MLflow gets a 401:
  `set -a; source .env; set +a; python -u -m src.train_kwp --config ... --run_name ...`
- Pick a new `--checkpoint_path` per experiment so runs do not overwrite each other's
  checkpoints in `/mnt/D/models/ct-log/`.

## Monitoring

| What | Where |
|---|---|
| Training output | `logs/<run>/run.log` (`tail -n 20`) |
| Restarts / give-ups | `logs/<run>/supervisor.log` |
| Finished | `grep "Best foreground mean IoU" logs/<run>/run.log` |
| MLflow broken | `grep "MLflow logging disabled\|MLflow model logging skipped" logs/<run>/run.log` — training continues, but nothing reaches DagsHub; tell the user |
| Process alive | `pgrep -af "src.train_kwp"` |
| MLflow run id | Not in the log until the run ends (`View run ... runs/<id>`). While running: `MlflowClient().search_runs([exp_id], filter_string="attributes.run_name = '<run>'")` |

Stop a run — supervisor first, otherwise it restarts the trainer:

```bash
pkill -f "supervise_kwp.sh .* $RUN "
pkill -f "src.train_kwp .*--run_name $RUN"
```

## MLflow on DagsHub

- Tracking URI (set in every `src/configs/*.yaml`): `https://dagshub.com/AddictionLord/ct-log.mlflow`
  UI: <https://dagshub.com/AddictionLord/ct-log/experiments>
- Credentials: `MLFLOW_TRACKING_USERNAME` / `MLFLOW_TRACKING_PASSWORD` in `.env`. The password is
  a DagsHub OAuth token that **expires 2026-10-24**; after that runs log "MLflow logging disabled".
- The user's shell also exports `MLFLOW_TRACKING_URI` for an unrelated work server. Always set
  the URI explicitly (configs do); never rely on the env var.
- One experiment per config (`mlflow_experiment_name`), e.g. `ct-log-kwp-v1`.
- **Never delete a run in the DagsHub UI while its training is still running.** Every later write
  is rejected with `INVALID_PARAMETER_VALUE` ("MLflow metric logging failed" in `run.log`; training
  continues, metrics are lost). Deleted runs are hidden from `search_runs`; pass
  `run_view_type=ViewType.ALL` and check `run.info.lifecycle_stage` when a run seems missing.
- A supervisor restart opens a **new MLflow run with the same name**; metrics of a crashed and
  resumed training are split across those runs. Mention it when reporting.

Query runs (loads credentials without printing them):

```bash
set -a; source .env; set +a
python - <<'EOF'
import mlflow
mlflow.set_tracking_uri("https://dagshub.com/AddictionLord/ct-log.mlflow")
runs = mlflow.search_runs(experiment_names=["ct-log-kwp-v1"], order_by=["start_time DESC"], max_results=10)
cols = [c for c in runs.columns if c.startswith(("tags.mlflow.runName", "status", "metrics.val/"))]
print(runs[["run_id", "start_time", *cols]].to_string())
EOF
```

### Logged models

- `MlflowLogger.log_model` logs the seg head on every new best (5-epoch smoothed val fg IoU) as a
  native MLflow PyTorch model in **pt2** (`torch.export`) format, exported from a CPU copy. It
  appears in the run's *Logged models* and the experiment's *Models* tab.
- Load anywhere, then move:
  ```python
  model = mlflow.pytorch.load_model("models:/<model_id>")   # GraphModule on CPU
  model = model.to("cuda")                                  # do NOT pass device= to load_model
  ```
  Input `[B, num_patches, feature_dim]` (kwp v1: `[B, 400, 4096]`), B is dynamic.
- pt2 files load only with the **same or newer torch** than the exporter (both hosts: 2.7.0).
- ONNX: `torch.onnx.export(model, (example,), dynamo=True, dynamic_shapes=({0: torch.export.Dim("batch", min=1)},))`
  (needs `onnxscript==0.2.7` with torch 2.7 in this env).
- Size: ~217 MB per logged model (158 MB weights + 59 MB input-example JSON). DagsHub free tier
  is 20 GB, and **deleting a model does not free storage** (DagsHub rejects artifact deletes).
  Do not log extra models or artifacts by hand.
- End-to-end check of the whole path: `python -m eval.mlflow_e2e log` then
  `python -m eval.mlflow_e2e verify --run-id <id>` (logs to experiment `ct-log/dinov3`).

## euler (remote GPU)

Only when the user asks. Access is via `scripts/euler.py` (JupyterLab REST API + kernel
websocket; no SSH). It scrubs `EULER_TOKEN` and `MLFLOW_TRACKING_PASSWORD` from its output.

```bash
eval "$(grep '^export EULER_' ~/.bashrc)"      # ~/.bashrc returns early in non-interactive shells
uv run --no-project scripts/euler.py exec "nvidia-smi; df -h ~"
uv run --no-project scripts/euler.py bg <name> "<long command>" --min-free-mb 8000   # ~/jobs/<name>/{log,exit,pid}
uv run --no-project scripts/euler.py exec "tail -n 20 ~/jobs/<name>/log"
uv run --no-project scripts/euler.py put <local> <remote> / get <remote> <local>   # base64 via REST
uv run --no-project scripts/euler.py close      # delete the kernel when done
```

- `put` is reliable up to ~150 MB per file. Split bigger files (`split -b 150M`), upload the
  parts, `cat` them together on euler and compare `md5sum` on both sides. `put` checks the
  uploaded size and exits non-zero on failure. Don't pipe it into `tail`/`grep` when you rely
  on `$?`.
- Run `uv run` with `--no-project`; otherwise uv creates `uv.lock` and `.venv` in the repo.
- **Shared account**: every user is `jovyan` with the same home and token. `~/work`, `~/.ssh`,
  `.gitconfig` belong to others: never read, modify or delete them. Work only in `~/ctlog-eval/`
  and `~/jobs/`.
- Any credential placed there is readable by the other users. The DagsHub token may go there
  only with the user's approval, as a file that the job deletes right after reading it.
- GPU is shared and other users' processes are invisible in the container: judge by
  `memory.used`, and stop and ask if less than half is free.
- Environment: conda base with torch 2.7.0+cu128 preinstalled; do not `pip install` into base
  (shared), use a venv or `pip install --target`. `/dev/shm` is 64 MB, so set DataLoader
  `num_workers=0` (ultralytics: `workers=0, cache="ram"`). No web proxy (no MLflow UI there).
- Provisioning a fresh copy (needs the user's OK): clone the public GitHub repo, copy datasets
  (~660 MB) and DINOv3 weights, and fix `REPO_DIR` in `src/segmentation_head.py`
  (hard-coded `/home/mary/code/dinov3`).
