import argparse
import os
from pathlib import Path
import tempfile
from typing import Dict, Optional

import mlflow
import numpy as np
import onnxruntime as ort
from src.loggers.mlflow_logger import MlflowLogger
from src.segmentation_head import SimpleSegmentationHead
from src.utils.metrics import EpochMetrics
import torch
from torch import nn

TRACKING_URI = "https://dagshub.com/AddictionLord/ct-log.mlflow"
EXPERIMENT = "ct-log/dinov3"
MODEL_NAME = "e2e_seg_head"
HEAD_KWARGS = {"feature_dim": 4096, "num_classes": 4, "input_size": 320}


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end check of MlflowLogger model logging.")
    sub = parser.add_subparsers(dest="mode", required=True)
    log_parser = sub.add_parser("log")
    log_parser.add_argument("--token-file", type=Path, default=None)
    log_parser.add_argument("--steps", type=int, default=20)
    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    if args.mode == "log":
        log_run(args.token_file, args.steps)
    else:
        verify_run(args.run_id)


def log_run(token_file: Optional[Path], steps: int) -> None:
    if token_file is not None:
        os.environ["MLFLOW_TRACKING_USERNAME"] = "AddictionLord"
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token_file.read_text().strip()
        token_file.unlink()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    head = SimpleSegmentationHead(**HEAD_KWARGS).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    logger = MlflowLogger(experiment_name=EXPERIMENT, run_name="e2e-model-logging", tracking_uri=TRACKING_URI)
    logger.start()
    mlflow.set_tags({"purpose": "e2e model logging test, safe to delete", "device": str(device)})
    logger.log_params({"steps": steps, "torch": torch.__version__, **HEAD_KWARGS})

    features = torch.randn(4, head.feature_map_size**2, head.feature_dim, device=device)
    target = torch.randint(0, HEAD_KWARGS["num_classes"], (4, 320, 320), device=device)
    for step in range(steps):
        head.train()
        optimizer.zero_grad()
        loss = loss_fn(head(features), target)
        loss.backward()
        optimizer.step()
        logger.log_metrics(EpochMetrics(epoch=step, split="train", loss=loss.item(), mean_iou=0.0))

    logger.log_model(head, MODEL_NAME, head.example_input())
    params_device = next(head.parameters()).device
    head.eval()
    reference_input = torch.randn(
        3, head.feature_map_size**2, head.feature_dim, generator=torch.Generator().manual_seed(1)
    )
    with torch.no_grad():
        reference_output = head(reference_input.to(device)).cpu().contiguous()
    with tempfile.TemporaryDirectory() as tmp_dir:
        np.save(Path(tmp_dir) / "input.npy", reference_input.numpy())
        np.save(Path(tmp_dir) / "output.npy", reference_output.numpy())
        mlflow.log_artifacts(tmp_dir, "reference")

    run_id = mlflow.active_run().info.run_id
    logger.end()
    print("RUN_ID", run_id)
    print("TRAINING_MODEL_DEVICE_AFTER_LOG", params_device)


def verify_run(run_id: str) -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    run = mlflow.get_run(run_id)
    print("run", run.info.status, "| metric keys", sorted(run.data.metrics), "| params", run.data.params)
    models = mlflow.MlflowClient().search_logged_models(
        experiment_ids=[run.info.experiment_id], filter_string="source_run_id = '%s'" % run_id
    )
    print("logged models", [(m.name, m.model_id, str(m.status)) for m in models])
    model_uri = "models:/%s" % models[0].model_id

    reference_dir = Path(mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="reference"))
    inputs = torch.from_numpy(np.load(reference_dir / "input.npy"))
    expected = torch.from_numpy(np.load(reference_dir / "output.npy"))
    results = {}

    cpu_module = mlflow.pytorch.load_model(model_uri)
    print("loaded", type(cpu_module).__name__, "on", next(cpu_module.parameters()).device)
    with torch.no_grad():
        results["pt2 cpu batch=3"] = _max_err(cpu_module(inputs), expected)
        results["pt2 cpu batch=1"] = _max_err(cpu_module(inputs[:1]), expected[:1])
        if torch.cuda.is_available():
            cuda_module = mlflow.pytorch.load_model(model_uri).to("cuda")
            results["pt2 cuda batch=3"] = _max_err(cuda_module(inputs.cuda()).cpu(), expected)

    try:
        results.update(_onnx_check(cpu_module, inputs, expected))
    except Exception as error:  # noqa: BLE001
        print("onnx export FAILED (%s: %s)" % (type(error).__name__, str(error).splitlines()[0]))
    for key, err in results.items():
        print("%-26s max_abs_err=%.2e %s" % (key, err, "OK" if err < 1e-3 else "MISMATCH"))


def _onnx_check(module: nn.Module, inputs: torch.Tensor, expected: torch.Tensor) -> Dict[str, float]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        onnx_path = Path(tmp_dir) / "model.onnx"
        onnx_program = torch.onnx.export(
            module, (inputs[:2],), dynamo=True, dynamic_shapes=({0: torch.export.Dim("batch", min=1)},)
        )
        onnx_program.save(str(onnx_path))
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        return {
            "onnx cpu batch=3": _max_err(
                torch.from_numpy(session.run(None, {input_name: inputs.numpy()})[0]), expected
            ),
            "onnx cpu batch=1": _max_err(
                torch.from_numpy(session.run(None, {input_name: inputs[:1].numpy()})[0]), expected[:1]
            ),
        }


def _max_err(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return (actual.float() - expected.float()).abs().max().item()


if __name__ == "__main__":
    main()
