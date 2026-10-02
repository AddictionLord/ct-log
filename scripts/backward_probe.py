import argparse
import time
from typing import Callable, List, Tuple

from src.segmentation_head import create_dinov3_segmentor
from src.train_kwp import extract_features
import torch
import torch.utils.checkpoint
import torchvision

N_LAYERS = 4
NUM_CLASSES = 3


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure peak memory and throughput of trainable-backbone variants.")
    parser.add_argument("--weights", type=str, required=True, help="DINOv3 ViT-L/16 backbone weights.")
    parser.add_argument("--res", type=int, nargs="+", default=[320, 784], help="Square input resolutions.")
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8], help="Batch sizes to try.")
    parser.add_argument("--steps", type=int, default=6, help="Steps per setting; the first 2 are warm-up.")
    parser.add_argument(
        "--variants",
        type=str,
        nargs="+",
        default=["dino_frozen", "dino_last4", "dino_full", "dino_full_ckpt", "deeplabv3_r101"],
        help="Variants to measure.",
    )
    args = parser.parse_args()

    device = torch.device("cuda")
    for variant in args.variants:
        for res in args.res:
            for batch in args.batches:
                result = measure(variant, res, batch, args.weights, args.steps, device)
                print("RESULT variant=%s res=%d batch=%d %s" % (variant, res, batch, result), flush=True)
                if result == "OOM":
                    break


def measure(variant: str, res: int, batch: int, weights: str, steps: int, device: torch.device) -> str:
    step_fn, params = build_variant(variant, res, weights, device)
    optimizer = torch.optim.AdamW(params, lr=1e-5)
    images = torch.randn(batch, 3, res, res, device=device)
    masks = torch.randint(0, NUM_CLASSES, (batch, res, res), device=device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    times: List[float] = []
    try:
        for step in range(steps):
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = step_fn(images)
                loss = torch.nn.functional.cross_entropy(logits.float(), masks)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            if step >= 2:
                times.append(time.perf_counter() - start)
    except torch.cuda.OutOfMemoryError:
        return "OOM"
    finally:
        del optimizer, step_fn, params
        torch.cuda.empty_cache()
    step_s = sum(times) / len(times)
    trainable_note = "peak=%.2fGB step=%.3fs samples/s=%.2f" % (
        torch.cuda.max_memory_allocated() / 1e9,
        step_s,
        batch / step_s,
    )
    return trainable_note


def build_variant(
    variant: str, res: int, weights: str, device: torch.device
) -> Tuple[Callable[[torch.Tensor], torch.Tensor], List[torch.nn.Parameter]]:
    if variant == "deeplabv3_r101":
        model = torchvision.models.segmentation.deeplabv3_resnet101(
            weights_backbone=torchvision.models.ResNet101_Weights.DEFAULT, num_classes=NUM_CLASSES
        ).to(device)
        model.train()
        return (lambda images: model(images)["out"]), list(model.parameters())

    backbone, head = create_dinov3_segmentor(
        backbone_weights=weights, num_classes=NUM_CLASSES, input_size=res, n_layers=N_LAYERS
    )
    backbone, head = backbone.to(device), head.to(device)
    params = list(head.parameters())
    if variant == "dino_frozen":
        backbone.eval()

        def frozen_step(images: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                features = extract_features(backbone, images, N_LAYERS)
            return head(features)

        return frozen_step, params

    blocks = list(backbone.blocks)
    trainable_blocks = blocks[-4:] if variant == "dino_last4" else blocks
    for block in trainable_blocks:
        for param in block.parameters():
            param.requires_grad = True
            params.append(param)
    if variant == "dino_full":
        for param in backbone.parameters():
            if not param.requires_grad:
                param.requires_grad = True
                params.append(param)
    if variant == "dino_full_ckpt":
        for param in backbone.parameters():
            if not param.requires_grad:
                param.requires_grad = True
                params.append(param)
        for block in blocks:
            wrap_checkpoint(block)
    backbone.train()
    return (lambda images: head(extract_features(backbone, images, N_LAYERS))), params


def wrap_checkpoint(block: torch.nn.Module) -> None:
    forward = block.forward

    def checkpointed(*args, **kwargs):
        return torch.utils.checkpoint.checkpoint(forward, *args, use_reentrant=False, **kwargs)

    block.forward = checkpointed


if __name__ == "__main__":
    main()
