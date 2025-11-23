# make_noisy_eval.py
import os

SRC_EVAL = "sample_data/test_eval_mediapipe.txt"  # 기존 깨끗한 eval 파일
OUT_EVAL_TEMPLATE = "sample_data/test_eval_snr{snr}.txt"
SNR_LIST = [0.0]

def make_noisy_eval():
    with open(SRC_EVAL, "r", encoding="utf-8") as f:
        lines = [l.rstrip("\n") for l in f if l.strip()]

    for snr in SNR_LIST:
        out_path = OUT_EVAL_TEMPLATE.format(snr=int(snr))
        with open(out_path, "w", encoding="utf-8") as fo:
            for line in lines:
                parts = line.split("\t")
                # [0]=video_path, [1]=audio_path, 나머지 = transcript, token 등이라고 가정

                audio_path = parts[1]

                # 1) test_preprocessed → test_preprocessed_noisy 로 변경
                audio_path = audio_path.replace(
                    "sample_data/test_preprocessed",
                    "sample_data/test_preprocessed_noisy"
                )

                # 2) 파일명에 _snrX.0 붙이기
                base, ext = os.path.splitext(audio_path)  # ...\8.wav
                audio_path = f"{base}_snr{snr}{ext}"      # ...\8_snr10.0.wav

                parts[1] = audio_path
                fo.write("\t".join(parts) + "\n")

        print(f"Saved eval: {out_path}")

if __name__ == "__main__":
    make_noisy_eval()
