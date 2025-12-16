import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
import importlib


def _load_video_clip_class():
    try:
        module = importlib.import_module("moviepy.editor")
        return module.VideoFileClip
    except ModuleNotFoundError as exc:
        raise ImportError(
            "moviepy가 설치되어 있지 않습니다. requirements.txt 를 설치해 주세요."
        ) from exc


def load_sentence_info(meta_path: Path) -> List[Dict[str, Any]]:
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    if isinstance(meta, list):
        meta = meta[0]
    sentences = meta.get("Sentence_info") or []
    if not sentences:
        raise ValueError(f"Sentence_info가 비어있습니다: {meta_path}")
    return sentences


def ensure_output_dir(base_dir: Path, video_name: str) -> Path:
    clip_dir = base_dir / Path(video_name).stem
    clip_dir.mkdir(parents=True, exist_ok=True)
    return clip_dir


def split_video(
    video_path: Path,
    meta_path: Path,
    output_root: Path,
    label_path: Path | None = None,
):
    VideoFileClip = _load_video_clip_class()
    sentences = load_sentence_info(meta_path)
    clip_dir = ensure_output_dir(output_root, video_path.stem)
    label_entries: List[str] = []
    with VideoFileClip(str(video_path)) as ref_clip:
        total_duration = ref_clip.duration or 0
        default_fps = ref_clip.fps or 25

    for idx, sentence in enumerate(sentences):
        start = float(sentence.get("start_time", 0))
        end = float(sentence.get("end_time", 0))
        if end <= start:
            print(f"[WARN] 구간이 올바르지 않아 건너뜁니다 (ID={sentence.get('ID')} start={start} end={end})")
            continue
        if start >= total_duration:
            print(f"[WARN] start가 전체 길이보다 길어 건너뜁니다 (start={start} > total={total_duration})")
            continue
        trimmed_end = min(end, total_duration)

        video_out = clip_dir / f"{idx}.mp4"
        video_segment = VideoFileClip(str(video_path)).subclip(start, trimmed_end)
        silent_clip = video_segment.without_audio()
        silent_clip.write_videofile(
            str(video_out),
            codec="libx264",
            audio=False,
            fps=video_segment.fps or default_fps,
            verbose=False,
            logger=None,
        )
        silent_clip.close()
        video_segment.close()

        text = sentence.get("sentence_text", "").replace("\n", " ").strip()
        label_entries.append(
            "\t".join([
                video_out.as_posix(),
                text,
            ])
        )
        print(f"[{idx+1}/{len(sentences)}] saved -> {video_out.name}")

    if label_path:
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("\n".join(label_entries), encoding="utf-8")
        print(f"Saved label file -> {label_path}")

    print(f"생성된 세그먼트 수: {len(label_entries)} (출력 폴더: {clip_dir})")


def parse_args():
    p = argparse.ArgumentParser(description="원본 영상을 JSON 구간으로 분할")
    p.add_argument("video", type=Path, help="전체 영상 mp4 경로")
    p.add_argument("meta", type=Path, help="Sentence_info가 포함된 JSON 경로")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("sample_data/split_clips"),
        help="분할된 영상/오디오를 저장할 루트 폴더",
    )
    p.add_argument(
        "--label",
        type=Path,
        default=None,
        help="(선택) 결과 메타 정보를 TSV 형식으로 저장할 경로",
    )
    return p.parse_args()


def main():
    args = parse_args()
    split_video(
        video_path=args.video,
        meta_path=args.meta,
        output_root=args.output_dir,
        label_path=args.label,
    )


if __name__ == "__main__":
    main()
