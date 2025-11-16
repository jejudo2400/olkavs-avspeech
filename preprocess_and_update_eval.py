import os
import argparse
import cv2


def crop_video_to_face_region(
    in_video_path: str,
    out_video_path: str,
    crop_size: int = 96,   # README 기준 96 x 96
    target_fps: float = None,
):
    cap = cv2.VideoCapture(in_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {in_video_path}")

    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    if orig_fps <= 0:
        orig_fps = 25.0
    fps = target_fps if target_fps is not None else orig_fps

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise RuntimeError(f"Cannot load cascade: {cascade_path}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    os.makedirs(os.path.dirname(out_video_path), exist_ok=True)
    writer = cv2.VideoWriter(out_video_path, fourcc, fps, (crop_size, crop_size), True)

    last_face_roi = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80),
        )

        if len(faces) > 0:
            faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
            fx, fy, fw, fh = faces[0]

            # 얼굴 박스를 살짝 키워서 턱/이마까지 포함
            cx = fx + fw // 2
            cy = fy + fh // 2
            side = int(max(fw, fh) * 1.2)

            x1 = max(cx - side // 2, 0)
            y1 = max(cy - side // 2, 0)
            x2 = min(cx + side // 2, w)
            y2 = min(cy + side // 2, h)

            face_roi = frame[y1:y2, x1:x2]
            last_face_roi = face_roi
        else:
            # 얼굴을 못 찾으면 직전 프레임 그대로 사용
            if last_face_roi is None:
                face_roi = frame
            else:
                face_roi = last_face_roi

        face_resized = cv2.resize(face_roi, (crop_size, crop_size))
        writer.write(face_resized)

    cap.release()
    writer.release()
    print(f"[Video] {in_video_path} -> {out_video_path}")



def process_eval_file(
    eval_in_path: str,
    eval_out_path: str,
    video_out_dir: str,
    crop_size: int = 112,
):
    """
    eval_in_path (TSV: video_path, audio_path, text, token_ids...)를 읽어서:
      1) 각 video_path에 대해 mouth crop mp4 생성
      2) 1열만 새 비디오 경로로 바꾼 TSV를 eval_out_path로 저장
    """
    os.makedirs(os.path.dirname(eval_out_path), exist_ok=True)
    os.makedirs(video_out_dir, exist_ok=True)

    with open(eval_in_path, "r", encoding="utf-8") as fin, \
         open(eval_out_path, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            cols = line.split("\t")
            if len(cols) < 4:
                print(f"⚠️  Skipping line (not enough columns): {line}")
                continue

            video_path = cols[0]
            audio_path = cols[1]
            text = cols[2]
            token_ids = "\t".join(cols[3:])  # 4열 이후는 그대로 유지

            # 입력 동영상 이름 기준으로 출력 파일명 설정
            base_name = os.path.basename(video_path)        # my1.mp4
            name_no_ext, _ = os.path.splitext(base_name)    # my1
            out_video_path = os.path.join(video_out_dir, f"{name_no_ext}_crop.mp4")

            # 실제 크롭 실행
            crop_video_to_face_region(
                in_video_path=video_path,
                out_video_path=out_video_path,
                crop_size=crop_size,
            )

            # TSV 한 줄 쓰기 (비디오 경로만 바뀜)
            new_line = "\t".join([out_video_path, audio_path, text, token_ids])
            fout.write(new_line + "\n")

    print(f"\n✅ Done. New eval file saved to: {eval_out_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--eval_in",
        type=str,
        required=True,
        help="기존 my_eval.txt 경로 (TSV: video, audio, text, token_ids...)",
    )
    parser.add_argument(
        "--eval_out",
        type=str,
        default="sample_data/my_eval_processed.txt",
        help="전처리 후 새로 저장할 eval 파일 경로",
    )
    parser.add_argument(
        "--video_out_dir",
        type=str,
        default="sample_data/processed_videos",
        help="크롭된 비디오(mp4)를 저장할 폴더",
    )
    parser.add_argument(
        "--crop_size",
        type=int,
        default=112,
        help="입 주변 패치 크기 (crop_size x crop_size)",
    )

    args = parser.parse_args()

    process_eval_file(
        eval_in_path=args.eval_in,
        eval_out_path=args.eval_out,
        video_out_dir=args.video_out_dir,
        crop_size=args.crop_size,
    )


if __name__ == "__main__":
    main()
