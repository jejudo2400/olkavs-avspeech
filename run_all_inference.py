import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fusion_inference.py for every noisy audio clip under a root directory.")
    parser.add_argument(
        "--audio_root",
        type=Path,
        default=Path("sample_data/test_preprocessed_noisy/test"),
        help="Root directory that holds *_snr*.wav files (default: sample_data/test_preprocessed_noisy/test)")
    parser.add_argument(
        "--video_root",
        type=Path,
        default=Path("sample_data/test_preprocessed/test"),
        help="Root directory that holds clean MP4 clips matching the audio tree (default: sample_data/test_preprocessed/test)")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/inference.yaml"),
        help="Path to inference YAML config (default: configs/inference.yaml)")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("zoo/audio_visual.pt"),
        help="Path to AVSR checkpoint (default: zoo/audio_visual.pt)")
    parser.add_argument(
        "--whisper_model",
        type=str,
        default="medium",
        help="Whisper model size to use (default: medium)")
    parser.add_argument(
        "--stt_conf_method",
        type=str,
        choices=["avg", "min", "prod"],
        default="avg",
        help="Confidence aggregation method for Whisper (default: avg)")
    parser.add_argument(
        "--threshold_low",
        type=float,
        default=0.3,
        help="Lower STT confidence threshold (default: 0.3)")
    parser.add_argument(
        "--threshold_high",
        type=float,
        default=0.7,
        help="Upper STT confidence threshold (default: 0.7)")
    parser.add_argument(
        "--pattern",
        type=str,
        default="*_snr*.wav",
        help="Glob pattern (relative to audio_root) to pick noisy wav files (default: *_snr*.wav)")
    parser.add_argument(
        "--log_detail",
        action="store_true",
        help="Pass --log_detail to fusion_inference.py for per-frame/token diagnostics.")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print resolved commands without executing them.")
    return parser.parse_args()


def resolve_video_path(audio_path: Path, audio_root: Path, video_root: Path) -> Path:
    rel_path = audio_path.relative_to(audio_root)
    stem = rel_path.stem
    if "_snr" in stem:
        clean_stem, _ = stem.rsplit("_snr", 1)
    else:
        clean_stem = stem
    if not clean_stem:
        raise ValueError(f"Cannot determine clean clip name for {audio_path}")
    video_rel = rel_path.with_name(f"{clean_stem}.mp4")
    return (video_root / video_rel).resolve()


def run_single(audio_path: Path, video_path: Path, out_json: Path, args: argparse.Namespace) -> int:
    cmd = [
        str(sys.executable),
        "fusion_inference.py",
        "--config", str(args.config),
        "--video_path", str(video_path),
        "--audio_path", str(audio_path),
        "--checkpoint", str(args.checkpoint),
        "--whisper_model", args.whisper_model,
        "--stt_conf_method", args.stt_conf_method,
        "--threshold_low", str(args.threshold_low),
        "--threshold_high", str(args.threshold_high),
        "--out", str(out_json),
    ]
    if args.log_detail:
        cmd.append("--log_detail")

    print(f"[BatchFusion] audio={audio_path}" \
          f" | video={video_path}" \
          f" | out={out_json}")
    if args.dry_run:
        print("[BatchFusion] DRY-RUN ->", " ".join(cmd))
        return 0

    proc = subprocess.run(cmd)
    return proc.returncode


def main():
    args = parse_args()
    audio_root = args.audio_root.resolve()
    video_root = args.video_root.resolve()

    if not audio_root.exists():
        raise FileNotFoundError(f"audio_root not found: {audio_root}")
    if not video_root.exists():
        raise FileNotFoundError(f"video_root not found: {video_root}")

    audio_files = sorted(audio_root.rglob(args.pattern))
    if not audio_files:
        print(f"[BatchFusion] No files matched pattern '{args.pattern}' under {audio_root}")
        return

    failures = []
    for idx, audio_path in enumerate(audio_files, 1):
        try:
            video_path = resolve_video_path(audio_path, audio_root, video_root)
        except Exception as exc:
            print(f"[BatchFusion][{idx}/{len(audio_files)}] Skip {audio_path}: {exc}")
            failures.append(audio_path)
            continue

        out_json = audio_path.with_suffix('.json')
        ret = run_single(audio_path, video_path, out_json, args)
        if ret != 0:
            print(f"[BatchFusion][{idx}/{len(audio_files)}] FAILED with exit code {ret}")
            failures.append(audio_path)
        else:
            print(f"[BatchFusion][{idx}/{len(audio_files)}] Done")

    if failures:
        print(f"[BatchFusion] Completed with {len(failures)} failure(s). See log above.")
    else:
        print("[BatchFusion] All files processed successfully.")


if __name__ == "__main__":
    main()
