import os
import argparse
import glob
import numpy as np
import soundfile as sf

def resample_audio(audio, orig_sr, target_sr):
    """Simple linear resampling to match differing sample rates."""
    if orig_sr == target_sr:
        return audio
    duration = len(audio) / orig_sr
    target_len = max(1, int(round(duration * target_sr)))
    orig_times = np.linspace(0.0, duration, num=len(audio), endpoint=False)
    target_times = np.linspace(0.0, duration, num=target_len, endpoint=False)
    return np.interp(target_times, orig_times, audio)

def load_wav_mono(path, target_sr=None):
    audio, sr = sf.read(path)
    # stereo -> mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    # 필요 시 리샘플링으로 목표 샘플레이트와 맞춘다
    if target_sr is not None and sr != target_sr:
        audio = resample_audio(audio, sr, target_sr)
        sr = target_sr
    return audio, sr

def repeat_or_trim(noise, target_len):
    if len(noise) >= target_len:
        return noise[:target_len]
    # 필요할 만큼 반복해서 붙이기
    n_repeat = int(np.ceil(target_len / len(noise)))
    noise_rep = np.tile(noise, n_repeat)
    return noise_rep[:target_len]

def mix_at_snr(
    clean,
    noise,
    snr_db,
    noise_gain=1.0,
    clean_gain=1.0,
    headroom_db=None,
    snr_offset=0.0,
):
    """
    clean, noise: 1D numpy array (same length)
    snr_db: 원하는 SNR (dB), 예: 5, 10, 20
    noise_gain / clean_gain: optional pre-scaling
    headroom_db: when SNR<0, clamp the computed noise gain to this headroom (dB)
    snr_offset: shift requested SNR (dB)
    """
    clean_scaled = clean * clean_gain
    noise_pre_scaled = noise * noise_gain

    # 신호/노이즈 파워
    p_clean = np.mean(clean_scaled ** 2)
    p_noise = np.mean(noise_pre_scaled ** 2) + 1e-12  # 0 나눔 방지용 작은 값

    # 원하는 SNR: p_clean / (a^2 * p_noise) = 10^(snr_db/10)
    effective_snr = snr_db + snr_offset
    target_ratio = 10 ** (effective_snr / 10.0)
    a = np.sqrt(p_clean / (p_noise * target_ratio))

    if effective_snr < 0 and headroom_db is not None:
        max_gain = 10 ** (headroom_db / 20.0)
        a = min(a, max_gain)

    noise_scaled = noise_pre_scaled * a
    noisy = clean_scaled + noise_scaled

    # clipping 방지: [-1, 1] 범위로 스케일링
    max_abs = np.max(np.abs(noisy)) + 1e-12
    if max_abs > 1.0:
        noisy = noisy / max_abs

    return noisy

def process_folder(
    clean_dir,
    noise_path,
    out_dir,
    snr_list,
    sr=None,
    noise_gain=1.0,
    clean_gain=1.0,
    headroom_db=None,
    snr_offset=0.0,
):
    os.makedirs(out_dir, exist_ok=True)

    # 소음 파일 로드
    noise, noise_sr = load_wav_mono(noise_path, target_sr=sr)

    clean_paths = sorted(glob.glob(os.path.join(clean_dir, "**", "*.wav"), recursive=True))

    print(f"Found {len(clean_paths)} clean wav files in {clean_dir}")

    for clean_path in clean_paths:
        clean, clean_sr = load_wav_mono(clean_path, target_sr=sr or None)

        # 첫 파일의 sr을 기준으로 삼고, 이후에는 검사
        if sr is None:
            sr = clean_sr
            if noise_sr != sr:
                noise = resample_audio(noise, noise_sr, sr)
                noise_sr = sr
        else:
            if clean_sr != sr:
                raise ValueError(f"Sample rate mismatch in {clean_path}: {clean_sr} != {sr}")

        # 길이 맞추기
        noise_segment = repeat_or_trim(noise, len(clean))

        for snr_db in snr_list:
            noisy = mix_at_snr(
                clean,
                noise_segment,
                snr_db,
                noise_gain=noise_gain,
                clean_gain=clean_gain,
                headroom_db=headroom_db,
                snr_offset=snr_offset,
            )

            rel_path = os.path.relpath(clean_path, clean_dir)
            base, _ = os.path.splitext(rel_path)

            out_subdir = os.path.join(out_dir, os.path.dirname(base))
            os.makedirs(out_subdir, exist_ok=True)

            out_name = f"{os.path.basename(base)}_snr{snr_db}.wav"
            out_path = os.path.join(out_subdir, out_name)

            sf.write(out_path, noisy, sr)
            print(f"Saved: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_dir", type=str, required=True,
                        help="깨끗한 음성 wav들이 들어있는 상위 폴더")
    parser.add_argument("--noise_wav", type=str, required=True,
                        help="소음 wav 파일 경로 (예: cafe_noise.wav)")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="노이즈가 섞인 wav를 저장할 폴더")
    parser.add_argument("--snrs", type=float, nargs="+", default=[0, -5, -10],
                        help="추가할 SNR 리스트 (dB)")
    parser.add_argument("--sr", type=int, default=None,
                        help="기대 샘플레이트 (예: 16000). None이면 첫 파일 sr 기준")
    parser.add_argument("--noise_gain", type=float, default=0.85,
                        help="노이즈에 적용할 사전 스케일 (기본 0.85)")
    parser.add_argument("--clean_gain", type=float, default=1.0,
                        help="클린 오디오에 적용할 사전 스케일 (기본 1.0)")
    parser.add_argument("--headroom_db", type=float, default=6.0,
                        help="음수 SNR 시 노이즈 게인 상한 (dB, 기본 6dB)")
    parser.add_argument("--snr_offset", type=float, default=2.0,
                        help="요청 SNR에 더해지는 오프셋 (dB, 기본 +2dB)")

    args = parser.parse_args()

    process_folder(
        clean_dir=args.clean_dir,
        noise_path=args.noise_wav,
        out_dir=args.out_dir,
        snr_list=args.snrs,
        sr=args.sr,
        noise_gain=args.noise_gain,
        clean_gain=args.clean_gain,
        headroom_db=args.headroom_db,
        snr_offset=args.snr_offset,
    )