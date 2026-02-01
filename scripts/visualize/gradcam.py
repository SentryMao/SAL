"""
Simple Grad-CAM visualization example for SSL_SeqModel.

This script demonstrates how to visualize attention maps using Grad-CAM
for a single model and audio file.

Usage:
    # Basic usage with SSL_SeqModel (default):
    python scripts/visualize/gradcam.py \
        --audio_path /path/to/audio.wav \
        --labels_path /path/to/labels.npy \
        --sample_id your_sample_id \
        --ckpt /path/to/xlsr2_300m.pt

    # With trained model checkpoint:
    python scripts/visualize/gradcam.py \
        --audio_path /path/to/audio.wav \
        --labels_path /path/to/labels.npy \
        --sample_id your_sample_id \
        --ckpt /path/to/xlsr2_300m.pt \
        --model_ckpt /path/to/model_checkpoints/last.ckpt \
        --model_type your_model_type(ssl_seq or ssl_seq_8labels_2loss) \
        --ssl_mode your_ssl_mode(s3prl or s3prl_weighted) \
        --device cuda \
        --output_path /path/to/your_output_path.png \
        --show

Arguments:
    Required:
        --audio_path: Path to audio file (.wav format)
        --labels_path: Path to segment labels (.npy file, dictionary format)
        --sample_id: Sample ID to load from labels dictionary
        --ckpt: Path to SSL encoder checkpoint (e.g., xlsr2_300m.pt)

    Optional:
        --model_type: Model type (default: "ssl_seq", choices: ssl_seq, ssl_seq_8labels_2loss)
        --model_ckpt: Path to trained model checkpoint (optional)
        --ssl_encoder: SSL encoder type (auto-detected from ckpt if not specified)
        --ssl_mode: SSL encoder mode (default: "s3prl", use "s3prl_weighted" for 8labels_2loss)
        --seq_model: Sequence model type (default: "cf", choices: lstm, rnn, tf, cf)
        --num_layers: Layers for sequence model (default: 2)
        --num_heads: Heads for transformer/conformer (default: 4)
        --pool: Pooling type (default: "avg", choices: avg, att)
        --pool_head_num: Attention heads for pool_head when pool=att (default: 1)
        --resolution: Segment resolution in seconds (default: 0.16)
        --device: Device to use (default: "cpu", choices: cpu, cuda)
        --output_path: Output path for visualization (default: "./gradcam_output.png")
        --show: Show plot interactively (flag)

Data Format:
    - labels_path should be a .npy file containing a dictionary:
        {sample_id: numpy_array_of_labels, ...}
    - Labels are segment-level (0=fake, 1=real)
    - Audio should be 16kHz sample rate
"""

import argparse
import os
import sys
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

def parse_args():
    parser = argparse.ArgumentParser(
        description="Grad-CAM visualization for SSL_SeqModel")
    parser.add_argument("--audio_path", type=str, required=True,
                        help="Path to audio file (.wav)")
    parser.add_argument("--labels_path", type=str, required=True,
                        help="Path to segment labels (.npy file)")
    parser.add_argument("--sample_id", type=str, required=True,
                        help="Sample ID to load from labels")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Checkpoint path for SSL encoder (e.g., xlsr2_300m.pt)")
    parser.add_argument("--model_ckpt", type=str,
                        help="Optional trained model checkpoint")
    parser.add_argument("--model_type", type=str, default="ssl_seq",
                        choices=["ssl_seq", "ssl_seq_8labels_2loss"],
                        help="Model type: ssl_seq (SSL_SeqModel) or ssl_seq_8labels_2loss (SSL_SeqModel_8Labels_2Loss)")
    parser.add_argument("--ssl_encoder", type=str, default=None,
                        help="SSL encoder type (e.g., 'xlsr', 'large', 'base'). Auto-detected from ckpt if not specified")
    parser.add_argument("--ssl_mode", type=str, default="s3prl",
                        choices=["s3prl", "s3prl_weighted"],
                        help="SSL encoder mode (default: s3prl, use s3prl_weighted for SSL_SeqModel_8Labels_2Loss)")
    parser.add_argument("--seq_model", type=str, default="cf",
                        choices=["lstm", "rnn", "tf", "cf"],
                        help="Sequence model type")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="Layers for sequence model (for tf/cf)")
    parser.add_argument("--num_heads", type=int, default=4,
                        help="Heads for transformer/conformer")
    parser.add_argument("--pool", type=str, default="avg",
                        choices=["avg", "att"],
                        help="Pooling type for pool_head")
    parser.add_argument("--pool_head_num", type=int, default=1,
                        help="Attention heads for pool_head when pool=att")
    parser.add_argument("--resolution", type=float, default=0.16,
                        help="Segment resolution used in model (seconds per segment)")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"], help="Device")
    parser.add_argument("--output_path", type=str, default="./gradcam_output.png",
                        help="Output path for visualization")
    parser.add_argument("--show", action="store_true",
                        help="Show plot interactively")
    args = parser.parse_args()
    return args


def ensure_project_import():
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../.."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


def load_audio_and_labels(audio_path: str, labels_path: str, sample_id: str, 
                          resolution: float) -> Tuple[np.ndarray, np.ndarray]:
    """Load audio file and corresponding segment labels."""
    import soundfile as sf
    x, _ = sf.read(audio_path)
    labels = np.load(labels_path, allow_pickle=True).item()
    y = labels[sample_id]
    target_len = int(16000 * resolution * len(y))
    if len(x) < target_len:
        x = np.pad(x, (0, int(target_len - len(x))), mode="constant")
    else:
        x = x[: int(target_len)]
    return x, y


class SeqIOHook:
    def __init__(self):
        self.input_act: Optional[torch.Tensor] = None
        self.output_act: Optional[torch.Tensor] = None
        self.input_grad: Optional[torch.Tensor] = None
        self.output_grad: Optional[torch.Tensor] = None

    def forward_hook(self, module, inp, out):
        self.input_act = inp[0].detach()
        if isinstance(out, (tuple, list)):
            self.output_act = out[0].detach()
        else:
            self.output_act = out.detach()

    def backward_hook(self, module, grad_in, grad_out):
        self.input_grad = grad_in[0].detach() if grad_in is not None and len(
            grad_in) > 0 else None
        gout = grad_out[0] if grad_out is not None and len(
            grad_out) > 0 else None
        if isinstance(gout, (tuple, list)):
            self.output_grad = gout[0].detach()
        else:
            self.output_grad = gout.detach() if gout is not None else None


def compute_temporal_gradcam(activations: torch.Tensor,
                             gradients: torch.Tensor) -> torch.Tensor:
    weights = gradients.mean(dim=1, keepdim=True)  # [B, 1, D]
    cam = (weights * activations).sum(dim=2)  # [B, T]
    cam = F.relu(cam)
    cam_min = cam.amin(dim=1, keepdim=True)
    cam_max = cam.amax(dim=1, keepdim=True)
    cam_norm = (cam - cam_min) / (cam_max - cam_min + 1e-8)
    return cam_norm


def load_checkpoint_if_exists(model: torch.nn.Module, ckpt_path: Optional[str],
                              device: torch.device) -> None:
    if ckpt_path is None or not os.path.isfile(ckpt_path):
        return
    ckpt_obj = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt_obj, dict) and "state_dict" in ckpt_obj:
        state_dict = ckpt_obj["state_dict"]
    elif isinstance(ckpt_obj, dict):
        state_dict = ckpt_obj
    else:
        raise RuntimeError(f"Unsupported checkpoint format at {ckpt_path}")
    target_keys = set(model.state_dict().keys())

    def strip_prefixes(name: str) -> str:
        for p in ("module.", "model.", "net."):
            if name.startswith(p):
                return name[len(p):]
        return name

    cleaned_and_filtered = {}
    for k, v in state_dict.items():
        nk = strip_prefixes(k)
        if nk in target_keys:
            cleaned_and_filtered[nk] = v
    model.load_state_dict(cleaned_and_filtered, strict=False)


def resample_to_target(arr: np.ndarray, T: int) -> np.ndarray:
    """Resample array to target length T."""
    if len(arr) == T:
        return arr
    src_idx = np.linspace(0, len(arr) - 1, num=len(arr))
    tgt_idx = np.linspace(0, len(arr) - 1, num=T)
    idx = np.clip(np.round(tgt_idx).astype(int), 0, len(arr) - 1)
    return arr[idx]


def main():
    args = parse_args()
    print(f"Using device: {args.device}")
    print(f"Model type: {args.model_type}")

    ensure_project_import()
    from src.models.net.model import SSL_SeqModel, SSL_SeqModel_8Labels_2Loss

    # Load audio and labels
    x_np, y = load_audio_and_labels(
        args.audio_path, args.labels_path, args.sample_id, args.resolution)
    x = torch.from_numpy(x_np).unsqueeze(0).float()

    device = torch.device(args.device)

    # Auto-detect ssl_encoder from ckpt if not specified
    if args.ssl_encoder is None:
        ckpt_lower = args.ckpt.lower()
        if "large" in ckpt_lower or "wavlm" in ckpt_lower:
            args.ssl_encoder = "large"
        elif "xlsr" in ckpt_lower:
            args.ssl_encoder = "xlsr"
        else:
            args.ssl_encoder = "base"
        print(f"Auto-detected ssl_encoder: {args.ssl_encoder}")

    # Build model based on model_type
    if args.model_type == "ssl_seq_8labels_2loss":
        model = SSL_SeqModel_8Labels_2Loss(
            ckpt=args.ckpt,
            seq_model=args.seq_model,
            ssl_encoder=args.ssl_encoder,
            mode=args.ssl_mode,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            pool=args.pool,
            resolution_train=args.resolution,
            resolution_test=args.resolution,
            pool_head_num=args.pool_head_num,
        ).to(device).eval()
    else:  # ssl_seq
        model = SSL_SeqModel(
            ckpt=args.ckpt,
            seq_model=args.seq_model,
            ssl_encoder=args.ssl_encoder,
            mode=args.ssl_mode,
            pool=args.pool,
            resolution_train=args.resolution,
            resolution_test=args.resolution,
        ).to(device).eval()
    
    load_checkpoint_if_exists(model, args.model_ckpt, device)

    # Register hooks for Grad-CAM
    hook = SeqIOHook()
    fh = model.seq_model.register_forward_hook(hook.forward_hook)
    bh = model.seq_model.register_full_backward_hook(hook.backward_hook)

    # Forward pass
    x = x.to(device)
    output = model(x)

    # Handle different model output formats
    if args.model_type == "ssl_seq_8labels_2loss":
        # SSL_SeqModel_8Labels_2Loss returns (out1, out2) where out2 is [B, T, 2]
        out1, out2 = output
        output = out2  # Use the binary classification head
    # SSL_SeqModel returns output directly as [B, T, 2]

    # Choose target class (class 0 = fake, class 1 = real)
    target_class = 0
    probs = F.softmax(output, dim=-1)
    target_time_idx = int(probs[0, :, target_class].argmax(dim=0).item())
    print(f"Target time index: {target_time_idx}, target class: {target_class}")

    # Backward pass to compute gradients
    model.zero_grad(set_to_none=True)
    score = output[0, :, target_class].sum()
    score.backward()

    # Compute Grad-CAM
    assert hook.input_act is not None and hook.input_grad is not None
    assert hook.output_act is not None and hook.output_grad is not None
    
    cam = compute_temporal_gradcam(hook.output_act, hook.output_grad)[
        0].detach().cpu().numpy()

    # Prepare visualization data
    target_T = len(cam)
    y_np = np.array(y, dtype=float)
    if len(y_np) != target_T:
        y_np = np.array([y_np[np.clip(int(round(t)), 0, len(y_np) - 1)] 
                        for t in np.linspace(0, len(y_np) - 1, num=target_T)],
                       dtype=float)
    
    # Get prediction probabilities
    probs_np = probs[0, :, 1].detach().cpu().numpy()  # Probability of class 1 (real)
    
    # Resample probabilities to match CAM length if needed
    if len(probs_np) != target_T:
        probs_np = resample_to_target(probs_np, target_T)
    
    # Time axis in seconds
    times = np.arange(target_T) * args.resolution

    # Create visualization
    import matplotlib
    matplotlib.use("Agg" if not args.show else matplotlib.get_backend())
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib import gridspec

    fig = plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[1.5, 1], hspace=0.1)

    # Top plot: labels and prediction probabilities
    ax_top = fig.add_subplot(gs[0])
    
    # Draw label spans
    start = 0
    spans = []
    for t in range(1, len(y_np) + 1):
        if t == len(y_np) or y_np[t] != y_np[start]:
            lab = int(y_np[start])
            spans.append((start, t, lab))
            start = t
    
    delta_t = (times[1] - times[0]) if len(times) > 1 else args.resolution
    for s, e, lab in spans:
        end_time = (times[e - 1] + delta_t) if e > 0 else (times[e] + delta_t)
        span_color = "#DBE8FC" if lab == 1 else "#FFE6CC"
        ax_top.axvspan(times[s], end_time, facecolor=span_color, alpha=1.0, linewidth=0)
    
    # Plot prediction probabilities
    ax_top.plot(times, probs_np, label="Prediction probability", 
                linewidth=2, color="orange", zorder=3)
    
    # Vertical lines at label boundaries
    for s, e, _ in spans[1:]:
        ax_top.axvline(x=times[s], color="gray", linestyle="--", 
                      linewidth=1.5, alpha=0.7)
    
    ax_top.set_ylabel("Probability", fontsize=12)
    ax_top.set_title("Grad-CAM Visualization", fontsize=14)
    ax_top.set_xticks([])
    ax_top.set_xlim(times[0], times[-1] if len(times) > 1 else times[0])
    ax_top.margins(x=0)
    ax_top.legend(loc='upper right', fontsize=10)
    ax_top.tick_params(axis='y', labelsize=10)
    
    # Add legend for labels
    real_patch = mpatches.Patch(color='#DBE8FC', alpha=1, label='Real (1)')
    fake_patch = mpatches.Patch(color='#FFE6CC', alpha=1, label='Fake (0)')
    ax_top.legend(handles=[real_patch, fake_patch], loc='upper left', 
                 fontsize=10, framealpha=0.9)

    # Bottom plot: Grad-CAM heatmap
    ax_bot = fig.add_subplot(gs[1])
    cam_img = cam[np.newaxis, :]
    im = ax_bot.imshow(
        cam_img,
        extent=[times[0], times[-1] if len(times) > 1 else times[0], 0, 1],
        aspect="auto",
        cmap="OrRd",
        interpolation="nearest",
        origin="lower",
    )
    ax_bot.set_yticks([])
    ax_bot.set_ylabel("CAM", fontsize=12)
    ax_bot.set_xlabel("Time (s)", fontsize=12)
    
    # Set x-axis ticks (only 0.5 multiples: 0, 0.5, 1.0, 1.5, ...)
    max_time = times[-1] if len(times) > 1 else times[0]
    target_ticks = np.arange(0, max_time + 0.5, 0.5)
    tick_times = [times[np.argmin(np.abs(times - t))] for t in target_ticks]
    tick_labels = [f"{t:.0f}" if t % 1 == 0 else f"{t:.1f}" for t in tick_times]
    ax_bot.set_xticklabels(tick_labels, fontsize=10)
    ax_bot.set_xlim(times[0], times[-1])
    
    # Vertical lines at label boundaries
    for s, e, _ in spans[1:]:
        ax_bot.axvline(x=times[s], color="gray", linestyle="--", 
                      linewidth=1.5, alpha=0.7)

    # Save figure
    fig.tight_layout()
    fig.savefig(args.output_path, dpi=150, bbox_inches='tight')
    print(f"Saved Grad-CAM visualization to: {args.output_path}")
    
    if args.show:
        plt.show()
    plt.close(fig)

    # Cleanup hooks
    fh.remove()
    bh.remove()


if __name__ == "__main__":
    main()
