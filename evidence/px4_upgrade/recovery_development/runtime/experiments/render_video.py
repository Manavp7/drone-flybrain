"""Render recorded Flyvis results; never generate replacement neural activity."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run, out = args.run.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    summary = json.loads((run / "summary.json").read_text())
    source = json.loads((run / "source.json").read_text())
    metadata = json.loads((run / "input.json").read_text())
    with np.load(run / "clip.npz", allow_pickle=False) as clip:
        frames, timestamps = clip["frames"], clip["timestamps"]
    with np.load(run / "cell_type_activity.npz", allow_pickle=False) as grouped:
        names, means, times = grouped["cell_types"], grouped["mean"], grouped["time_s"]
    channels = [f"T{t}{direction}" for t in (4, 5) for direction in "abcd"]
    channels = [name for name in channels if name in names]
    ids = [list(names).index(name) for name in channels]
    selected = means[:, ids]
    low, high = min(0., float(selected.min())), max(.1, float(selected.max()))
    capture = cv2.VideoCapture(source["video_path"])
    indices = set(metadata["selected_source_indices"])
    rgb_frames = []
    for i in range(max(indices) + 1):
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError("Original source video no longer decodes")
        if i in indices:
            rgb_frames.append(frame)
    capture.release()
    if len(rgb_frames) != len(frames):
        raise RuntimeError("Source video and recorded selected indices disagree")

    os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.facecolor": "#101923", "axes.facecolor": "#101923",
                         "text.color": "#edf5f6", "axes.labelcolor": "#becad0",
                         "xtick.color": "#becad0", "ytick.color": "#becad0",
                         "axes.edgecolor": "#40505b", "font.size": 10})
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.6), constrained_layout=True)
    middle = len(frames) // 2
    axes[0, 0].imshow(cv2.cvtColor(rgb_frames[middle], cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Public drone video · Experienciausuario · CC0")
    axes[0, 0].axis("off")
    square = frames[middle]
    h, w = square.shape
    side = min(h, w)
    square = square[(h-side)//2:(h+side)//2, (w-side)//2:(w+side)//2]
    axes[0, 1].imshow(square, cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title("Recorded luminance crop before 721-receptor sampling")
    axes[0, 1].axis("off")
    for i, label in enumerate(channels):
        axes[1, 0].plot(times-times[0], selected[:, i], label=label, lw=1.3)
    axes[1, 0].set(title="Recorded T4/T5 cell-type mean activity", xlabel="Clip time (s)", ylabel="Model activity (arbitrary units)")
    axes[1, 0].legend(ncol=4, fontsize=8, frameon=False)
    heat = axes[1, 1].imshow(means.T, aspect="auto", origin="lower", cmap="viridis",
                             extent=[0, times[-1]-times[0], 0, len(names)])
    axes[1, 1].set(title=f"All {len(names)} cell types · mean activity", xlabel="Clip time (s)", ylabel="Cell-type index")
    fig.colorbar(heat, ax=axes[1, 1], label="Arbitrary units")
    fig.suptitle(f"Real Flyvis inference · {summary['activity_shape'][1]:,} neurons · {summary['simulated_clip_s']:.2f} s video", fontsize=18)
    fig.savefig(out / "neural_activity.png", dpi=160)
    plt.close(fig)

    raw = out / "preview_intermediate.mp4"
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), 24., (1120, 630))
    if not writer.isOpened():
        raise RuntimeError("Preview video encoder unavailable")
    stimulus_times = np.asarray(summary["stimulus_timestamps_s"])
    step_ids = np.searchsorted(stimulus_times + 1e-10, timestamps, side="left")
    valid = np.flatnonzero(step_ids < len(stimulus_times))
    t0, t1 = float(timestamps[0]), float(timestamps[valid[-1]])

    def text(canvas, value, position, scale=.6, color=(224, 235, 238)):
        cv2.putText(canvas, value, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

    try:
        for playback_t in np.arange(t0, t1 + 1e-8, 1/24):
            j = max(0, int(np.searchsorted(timestamps, playback_t, side="right")-1))
            step = int(step_ids[j])
            canvas = np.full((630, 1120, 3), (35, 25, 16), dtype=np.uint8)
            text(canvas, "FLYVIS / REAL PRETRAINED NETWORK", (28, 36), .8)
            text(canvas, f"45,669 neurons | CPU | input {playback_t-t0:0.2f} / {t1-t0:0.2f} s", (28, 67), .52)
            original = cv2.resize(rgb_frames[j], (640, 360), interpolation=cv2.INTER_AREA)
            canvas[100:460, 28:668] = original
            text(canvas, "Public drone video / Experienciausuario / CC0", (28, 488), .5)
            text(canvas, "RECORDED T4/T5 MEAN ACTIVITY", (710, 102), .53)
            text(canvas, "Arbitrary model units; not motor commands", (710, 128), .4)
            for c, label in enumerate(channels):
                y = 164 + c*38
                xzero = int(785 + (0-low)/(high-low)*260)
                xvalue = int(785 + (float(selected[step, c])-low)/(high-low)*260)
                text(canvas, label, (710, y+4), .55)
                cv2.line(canvas, (xzero, y-11), (xzero, y+11), (125, 130, 130), 1)
                cv2.rectangle(canvas, (min(xzero, xvalue), y-8), (max(xzero, xvalue)+1, y+8), (172, 220, 97), -1)
                text(canvas, f"{selected[step,c]:.2f}", (1050, y+4), .4)
            text(canvas, "Camera pixels -> 721 receptors -> neural activity", (28, 538), .62)
            text(canvas, "Recorded offline inference. Drone control and navigation accuracy were not tested.", (28, 575), .47)
            progress = (playback_t-t0)/max(t1-t0, .01)
            cv2.rectangle(canvas, (28, 604), (1092, 609), (64, 72, 74), -1)
            cv2.rectangle(canvas, (28, 604), (28+int(1064*progress), 609), (172, 220, 97), -1)
            writer.write(canvas)
    finally:
        writer.release()
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(raw),
                    "-c:v", "libx264", "-crf", "22", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                    str(out/"flyvis_video_preview.mp4")], check=True)
    raw.unlink()
    (out/"preview_metadata.json").write_text(json.dumps({
        "input_run": str(run), "data": "Actual recorded activity; no synthetic replacement",
        "video_rate": 24, "display_timing": "24fps presentation with source-PTS zero-order hold",
        "neural_alignment": "First neural step containing each displayed input; response occurs dt later",
        "cell_type_display": channels, "license": "Source video CC0-1.0", "author": "Experienciausuario",
    }, indent=2)+"\n")
    print(out)


if __name__ == "__main__":
    main()
