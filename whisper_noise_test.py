import numpy as np
import whisper_timestamped as whisper

audio_path = r"sample_data\test_preprocessed_noisy\test\lip_subset\a\b\c\d\lip_J_1_F_02_C002_A_001\0_snr0.0.wav"

model = whisper.load_model("small")

# whisper timestamped는 한 번에 처리 가능
result = whisper.transcribe(model, audio_path, language="ko")

print("Whisper 결과:", result["text"])

# token-level logprob 조회
logprobs = [t["logprob"] for t in result["segments"][0]["tokens"]]

probs = [np.exp(lp) for lp in logprobs]  # logprob → 확률 값

conf_avg = float(np.mean(probs))
conf_min = float(np.min(probs))

print("평균 confidence:", conf_avg)
print("최소 confidence:", conf_min)
