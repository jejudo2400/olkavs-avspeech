import os
import argparse
from typing import List, Tuple

import cv2
import mediapipe as mp
import numpy as np


# =========================
# 0. Mediapipe FaceMesh 설정
# =========================

mp_face_mesh = mp.solutions.face_mesh

# 입 주변 랜드마크 인덱스 (outer + inner lips 근처)
MOUTH_LANDMARKS = list(range(61, 89))  # 61~88


# =========================
# 1. 한 프레임에서 립 bbox 계산
# =========================

# 얼굴 전체 중 '입'에 해당하는 주요 랜드마크 인덱스 집합
MOUTH_LANDMARKS = [
    61, 146, 91, 181, 84, 17,
    314, 405, 321, 375, 291, 308, 324, 318, 402, 317,
    14, 87, 178, 88, 95,
    185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312,
    13, 82, 81, 42, 183, 78,
]


def compute_lip_bbox_from_landmarks(
    landmarks,
    img_w: int,
    img_h: int,
    margin: float = 0.1,
) -> Tuple[int, int, int, int]:
    """
    landmarks: mediapipe face_landmarks.landmark
    반환: (top, left, bottom, right) = (y1, x1, y2, x2)
    AIHub Lip_bounding_box처럼 '입만 감싸는 직사각형'을 만든다.
    """
    xs, ys = [], []
    for idx in MOUTH_LANDMARKS:
        lm = landmarks[idx]
        xs.append(int(lm.x * img_w))
        ys.append(int(lm.y * img_h))

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    bw = x_max - x_min
    bh = y_max - y_min

    # 가로/세로에 각각 margin 비율만큼만 여유를 줌 (정사각형으로 만들지 않음)
    margin_x = margin
    margin_y = margin

    x1 = int(max(x_min - bw * margin_x, 0))
    x2 = int(min(x_max + bw * margin_x, img_w - 1))
    y1 = int(max(y_min - bh * margin_y, 0))
    y2 = int(min(y_max + bh * margin_y, img_h - 1))

    return y1, x1, y2, x2  # (top, left, bottom, right)



# =========================
# 1-1. BBox 간단 smoothing (선택)
# =========================

def smooth_bboxes(
    bboxes: List[Tuple[int, int, int, int]],
    k: int = 3,
) -> List[Tuple[int, int, int, int]]:
    """
    간단 이동 평균으로 bbox를 부드럽게 만들기.
    k=3이면 현재 프레임 기준 앞/뒤 3프레임씩 포함해서 평균.
    """
    if not bboxes:
        return bboxes

    smoothed: List[Tuple[int, int, int, int]] = []
    n = len(bboxes)

    for i in range(n):
        y1s, x1s, y2s, x2s = [], [], [], []
        for j in range(max(0, i - k), min(n, i + k + 1)):
            y1, x1, y2, x2 = bboxes[j]
            y1s.append(y1)
            x1s.append(x1)
            y2s.append(y2)
            x2s.append(x2)
        y1 = int(np.mean(y1s))
        x1 = int(np.mean(x1s))
        y2 = int(np.mean(y2s))
        x2 = int(np.mean(x2s))
        smoothed.append((y1, x1, y2, x2))

    return smoothed


# =========================
# 2. 비디오 전체에서 프레임별 립 bbox 추출
# =========================

def extract_lip_bboxes_mediapipe(
    video_path: str,
    margin: float = 0.1,
    crop_size: int = 224,  # 리사이즈 기준 크기 (입 픽셀 통계 계산용)
    smooth_k: int = 3,     # bbox smoothing 윈도우 크기
) -> List[Tuple[int, int, int, int]]:
    """
    비디오 전체에 대해 mediapipe FaceMesh로 프레임별 립 bounding box 추출.
    실패 시에는 직전 bbox를 그대로 사용.
    또한, 리사이즈(crop_size x crop_size) 했을 때
    입 높이가 몇 픽셀인지 통계를 출력.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    raw_bboxes: List[Tuple[int, int, int, int]] = []
    last_bbox = None

    # 통계용
    mouth_px_list = []
    total_frames = 0

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            total_frames += 1
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            bbox = None
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0].landmark

                # 1) lip bbox 계산
                xs, ys = [], []
                for idx in MOUTH_LANDMARKS:
                    lm = face_landmarks[idx]
                    xs.append(int(lm.x * w))
                    ys.append(int(lm.y * h))
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)

                bw = x_max - x_min
                bh = y_max - y_min

                # 2) 정사각형 side + margin
                side = max(bw, bh) * (1.0 + margin)

                # 3) 리사이즈 후 입 높이(px) 추정
                if side > 0:
                    mouth_frac = bh / side  # 패치 안에서 입이 차지하는 비율
                    mouth_px_resized = mouth_frac * crop_size
                    mouth_px_list.append(mouth_px_resized)

                # 4) 실제 bbox (top, left, bottom, right)
                cx = (x_min + x_max) / 2.0
                cy = (y_min + y_max) / 2.0
                x1 = int(max(cx - side / 2.0, 0))
                y1 = int(max(cy - side / 2.0, 0))
                x2 = int(min(cx + side / 2.0, w - 1))
                y2 = int(min(cy + side / 2.0, h - 1))
                bbox = (y1, x1, y2, x2)

            if bbox is None:
                # 이번 프레임에서 실패 → 직전 bbox 사용
                if last_bbox is not None:
                    bbox = last_bbox
                else:
                    # 첫 프레임부터 계속 실패하는 극단적인 경우:
                    # 이미지 중앙 정사각형으로 fallback
                    side = min(h, w) // 3
                    cx, cy = w // 2, h // 2
                    x1 = max(cx - side // 2, 0)
                    y1 = max(cy - side // 2, 0)
                    x2 = min(cx + side // 2, w - 1)
                    y2 = min(cy + side // 2, h - 1)
                    bbox = (y1, x1, y2, x2)

            raw_bboxes.append(bbox)
            last_bbox = bbox

    cap.release()

    # BBox smoothing
    bboxes = smooth_bboxes(raw_bboxes, k=smooth_k) if smooth_k > 0 else raw_bboxes

    print(f"[BBox] {video_path} -> {len(bboxes)} frames (total={total_frames})")

    # 입 픽셀 통계 출력
    if mouth_px_list:
        arr = np.array(mouth_px_list)
        print(f"[Mouth Stats] (resize={crop_size}x{crop_size}) for {video_path}")
        print(f"  frames with landmarks : {len(mouth_px_list)}/{total_frames}")
        print(f"  mouth px mean         : {arr.mean():.2f}")
        print(f"  mouth px median       : {np.median(arr):.2f}")
        print(f"  mouth px min/max      : {arr.min():.2f} / {arr.max():.2f}")
    else:
        print(f"[Mouth Stats] no landmarks detected for {video_path}")

    return bboxes


# =========================
# 3. AIHub / olkavs style crop 함수
# =========================

def crop_video_like_aidataset(
    video_path: str,
    bboxes: List[Tuple[int, int, int, int]],
    save_path: str,
    resize_shape=(224, 224),  # ★ 기본 224x224로 변경
    fps_override: float = None,
):
    """
    olkavs preprocess.py 구조를 따라:
      - 프레임별로 bbox로 자른 후
      - resize_shape로 리사이즈해서
      - 새 mp4로 저장
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    if orig_fps <= 0:
        orig_fps = 25.0
    fps = fps_override or orig_fps

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    out = cv2.VideoWriter(save_path, fourcc, fps, resize_shape)

    frame_idx = 0
    written = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= len(bboxes):
            break

        top, left, bottom, right = bboxes[frame_idx]
        h, w = frame.shape[:2]

        top = int(max(0, min(top, h - 1)))
        bottom = int(max(0, min(bottom, h)))
        left = int(max(0, min(left, w - 1)))
        right = int(max(0, min(right, w)))

        if bottom <= top or right <= left:
            # 이상한 bbox면 전체 중앙 크롭으로 대체
            side = min(h, w) // 3
            cx, cy = w // 2, h // 2
            left = max(cx - side // 2, 0)
            top = max(cy - side // 2, 0)
            right = min(cx + side // 2, w)
            bottom = min(cy + side // 2, h)

        crop = frame[top:bottom, left:right]
        crop = cv2.resize(crop, resize_shape, interpolation=cv2.INTER_LINEAR)

        out.write(crop)
        written += 1
        frame_idx += 1

    cap.release()
    out.release()
    print(f"[Crop] {video_path} -> {save_path} (frames={written}, fps={fps:.2f})")


# =========================
# 4. my_eval.txt를 읽어 전체 처리
# =========================

def process_eval_file_mediapipe_like_aidataset(
    eval_in_path: str,
    eval_out_path: str,
    video_out_dir: str,
    crop_size: int = 224,   # ★ 기본 224
    margin: float = 0.1,
):
    """
    my_eval.txt (video, audio, text, token_ids...) 를 읽어서:
      1) 각 video_path에 대해 mediapipe로 프레임별 lip bbox 추출
      2) olkavs 스타일로 crop + resize해서 새 mp4 생성
      3) 1열을 새 비디오 경로로 바꾼 TSV를 eval_out_path로 저장
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
                print(f"⚠️  skip (not enough cols): {line}")
                continue

            video_path = cols[0]
            audio_path = cols[1]
            text = cols[2]
            token_ids = "\t".join(cols[3:])  # 나머지 전부

            base = os.path.basename(video_path)
            name_no_ext, _ = os.path.splitext(base)
            out_video_path = os.path.join(video_out_dir, f"{name_no_ext}_mp_lip_ai.mp4")

            print(f"\n=== Processing {video_path} ===")
            # 1) mediapipe 로 프레임별 립 bbox 추출 (+ 입 픽셀 통계)
            bboxes = extract_lip_bboxes_mediapipe(
                video_path,
                margin=margin,
                crop_size=crop_size,
                smooth_k=3,
            )
            # 2) olkavs 스타일로 비디오 crop 후 저장 (224x224)
            crop_video_like_aidataset(
                video_path=video_path,
                bboxes=bboxes,
                save_path=out_video_path,
                resize_shape=(crop_size, crop_size),
            )

            # 3) 새 TSV 라인 기록
            new_line = "\t".join([out_video_path, audio_path, text, token_ids])
            fout.write(new_line + "\n")

    print(f"\n✅ New eval file saved: {eval_out_path}")


# =========================
# 5. main
# =========================

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
        default="sample_data/my_eval_mediapipe_ai.txt",
        help="전처리 후 저장할 eval 파일 경로",
    )
    parser.add_argument(
        "--video_out_dir",
        type=str,
        default="sample_data/processed_videos_mp_ai",
        help="크롭된 비디오를 저장할 폴더",
    )
    parser.add_argument(
        "--crop_size",
        type=int,
        default=224,  # ★ 기본값 224
        help="출력 패치 크기 (crop_size x crop_size, 학습 전처리와 맞추려면 224 권장)",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.1,
        help="입 bbox 주변 여유 비율 (0.1 ~ 0.3 사이에서 튜닝 추천)",
    )

    args = parser.parse_args()

    process_eval_file_mediapipe_like_aidataset(
        eval_in_path=args.eval_in,
        eval_out_path=args.eval_out,
        video_out_dir=args.video_out_dir,
        crop_size=args.crop_size,
        margin=args.margin,
    )


if __name__ == "__main__":
    main()
