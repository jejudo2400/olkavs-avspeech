import whisper_noise_test
import sounddevice as sd
import numpy as np
import time

# 설정값
SAMPLE_RATE = 16000  # Whisper가 요구하는 샘플 레이트
DURATION = 5         # 녹음할 시간 (초)
BLOCK_SIZE = int(SAMPLE_RATE * DURATION)

# 1. Whisper 모델 로드
# "base" 모델은 빠르고 성능도 준수합니다.
# 더 높은 정확도를 원하면 "small", "medium", "large" 등을 사용할 수 있으나,
# 그만큼 처리 속도가 느려집니다. 실시간용으로는 "base"나 "small"을 추천합니다.
try:
    model = whisper_noise_test.load_model("base")
    print("Whisper 'base' 모델을 로드했습니다.")
except Exception as e:
    print(f"모델 로드 중 오류 발생: {e}")
    print("모델을 다운로드해야 할 수 있습니다. 인터넷 연결을 확인하세요.")
    exit()

print("\n--- 실시간 음성 인식을 시작합니다. ---")
print(f"매 {DURATION}초마다 음성을 인식하여 출력합니다. (Ctrl+C로 종료)")

try:
    while True:
        print(f"\n{DURATION}초간 말씀하세요...")

        # 2. 실시간 오디오 녹음
        # sd.rec()는 녹음을 시작하고 바로 다음 코드로 넘어갑니다 (논블로킹).
        # sd.wait()를 만나야 녹음이 끝날 때까지 기다립니다.
        myrecording = sd.rec(BLOCK_SIZE, samplerate=SAMPLE_RATE, channels=1, dtype='float32')
        
        # 녹음이 끝날 때까지 대기
        sd.wait()  
        
        # 3. Whisper로 오디오 전사 (Transcription)
        print("음성 처리 중...")
        
        # 녹음된 데이터를 1차원 배열로 만듭니다.
        audio_data = myrecording.flatten()
        
        # Whisper 모델을 사용하여 오디오 데이터를 텍스트로 변환
        # language="ko"로 설정하여 한국어 인식을 명시합니다.
        result = model.transcribe(audio_data, language="ko", fp16=False)
        
        # 4. 결과 출력
        transcribed_text = result["text"]
        
        if transcribed_text.strip():  # 공백만 있는 결과는 무시
            print(f"인식된 텍스트: {transcribed_text}")
        else:
            print("(인식된 내용 없음)")

except KeyboardInterrupt:
    print("\n프로그램을 종료합니다.")

except Exception as e:
    print(f"오류가 발생했습니다: {e}")