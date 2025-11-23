import os
import json
import csv
import argparse

def collect_fusion(json_root, out_csv):
    rows = []

    for root, _, files in os.walk(json_root):
        for file in files:
            if not file.endswith(".json"):
                continue

            path = os.path.join(root, file)
            if os.path.basename(path) == "lip_J_1_F_02_C002_A_001.json":
                continue
            with open(path, "r", encoding="utf-8") as f:
                j = json.load(f)

            if isinstance(j, list):
                if not j:
                    continue
                j = j[0]

            audio = j.get("audio", "")
            video = j.get("video", "")
            avsr_text = j.get("avsr_text", "")
            whisper_text = j.get("whisper_text", "")
            fused_text = j.get("fused_text", "")
            selected_text = j.get("selected_text", "")
            avsr_conf = j.get("avsr_ctc_confidence", None)
            stt_conf = j.get("stt_confidence", None)
            gt_text = j.get("gt_text", "")


            weights = j.get("weights", {})
            selection = j.get("selection", {})

            decision = selection.get("decision", "")
            th_low = selection.get("threshold_low", None)
            th_high = selection.get("threshold_high", None)
            stt_conf_method = selection.get("stt_conf_method", "")

            rows.append({
                "json_path": path,
                "audio_path": audio,
                "video_path": video,
                "avsr_text": avsr_text,
                "whisper_text": whisper_text,
                "selected_text": selected_text,
                "fused_text": fused_text,
                "gt_text": gt_text,
                "avsr_conf": avsr_conf,
                "stt_conf": stt_conf,
                "weight_avsr": weights.get("avsr"),
                "weight_whisper": weights.get("whisper"),
                "decision": decision,
                "threshold_low": th_low,
                "threshold_high": th_high,
                "stt_conf_method": stt_conf_method,
            })

    # CSV로 저장
    fieldnames = [
        "json_path", "audio_path", "video_path",
        "avsr_text", "whisper_text", "selected_text", "fused_text",
        "avsr_conf", "stt_conf",
        "weight_avsr", "weight_whisper","gt_text",
        "decision", "threshold_low", "threshold_high", "stt_conf_method"
    ]

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"총 {len(rows)}개의 json을 분석했습니다.")
    print(f"CSV 저장 위치: {out_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_root", type=str, required=True,
                        help="fusion_inference 결과 json 들이 들어있는 상위 폴더")
    parser.add_argument("--out_csv", type=str, required=True,
                        help="요약 CSV를 저장할 경로")
    args = parser.parse_args()

    collect_fusion(args.json_root, args.out_csv)
