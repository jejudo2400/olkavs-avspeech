import tkinter as tk
from tkinter import filedialog, messagebox
import threading
from pathlib import Path
from datetime import datetime
import importlib

from fusion_inference import run_fusion, parse_args as fusion_parse_args


def _get_video_file_clip():
    try:
        module = importlib.import_module("moviepy.editor")
        return module.VideoFileClip
    except ModuleNotFoundError as exc:
        raise ImportError(
            "moviepy가 설치되어 있지 않습니다. requirements.txt 업데이트 후 pip install -r requirements.txt 를 실행하세요."
        ) from exc

class FusionUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AVSR + STT Demo")
        self.geometry("700x450")

        tk.Button(self, text="영상 불러오기", command=self.pick_video).pack(pady=10)
        self.path_var = tk.StringVar(value="선택된 파일 없음")
        tk.Label(self, textvariable=self.path_var, wraplength=650).pack()

        duration_frame = tk.Frame(self)
        duration_frame.pack(pady=5)
        tk.Label(duration_frame, text="사용할 구간 길이(초)").pack(side=tk.LEFT)
        self.duration_var = tk.DoubleVar(value=6.0)
        tk.Spinbox(
            duration_frame,
            from_=1.0,
            to=20.0,
            increment=0.5,
            width=5,
            textvariable=self.duration_var
        ).pack(side=tk.LEFT, padx=5)

        self.status_var = tk.StringVar(value="대기 중")
        tk.Label(self, textvariable=self.status_var, fg="blue").pack(pady=5)

        self.output = tk.Text(self, height=15, width=80)
        self.output.pack(padx=10, pady=10)

    def pick_video(self):
        video_path = filedialog.askopenfilename(
            title="영상 선택",
            filetypes=[("MP4 Video", "*.mp4"), ("All files", "*.*")]
        )
        if not video_path:
            return
        self.path_var.set(video_path)
        threading.Thread(target=self.run_inference, args=(video_path,), daemon=True).start()

    def run_inference(self, video_path: str):
        try:
            video = Path(video_path)
            try:
                clip_seconds = max(0.5, float(self.duration_var.get()))
            except (tk.TclError, ValueError):
                clip_seconds = 6.0

            config_path = Path("configs/inference.yaml")
            if not config_path.exists():
                messagebox.showerror("오류", f"config 파일을 찾을 수 없습니다: {config_path}")
                return

            self.status_var.set("클립 잘라내는 중...")
            trimmed_video, trimmed_audio, used_duration = self.prepare_clip(video, clip_seconds)
            self.status_var.set(f"{used_duration:.1f}초 클립 추론 중...")

            out_path = trimmed_video.with_suffix(".json")
            args = self.build_fusion_args(config_path, trimmed_video, trimmed_audio, out_path)

            run_fusion(args)

            # 결과 JSON을 열어 출력(예시)
            import json
            data = json.loads(Path(args.out).read_text(encoding="utf-8"))
            display = (
                f"원본: {video}\n"
                f"클립: {trimmed_video.name} ({used_duration:.2f}s)\n"
                f"AVSR   : {data['avsr_text']}\n"
                f"Whisper: {data['whisper_text']}\n"
                f"Selected: {data['selected_text']}\n"
            )
            self.output.delete("1.0", tk.END)
            self.output.insert(tk.END, display)
            self.status_var.set("완료")
        except Exception as e:
            messagebox.showerror("오류", str(e))
            self.status_var.set("실패")

    def prepare_clip(self, source: Path, desired_seconds: float):
        clip_dir = Path("tmp/ui_clips")
        clip_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{source.stem}_{timestamp}"
        video_out = clip_dir / f"{base_name}.mp4"
        audio_out = clip_dir / f"{base_name}.wav"

        VideoFileClip = _get_video_file_clip()
        with VideoFileClip(str(source)) as clip:
            total_duration = clip.duration or 0
            if total_duration <= 0:
                raise ValueError("영상 길이를 확인할 수 없습니다.")

            usable = min(desired_seconds, total_duration)
            subclip = clip.subclip(0, usable)

            audio_segment = subclip.audio
            if audio_segment is None:
                raise ValueError("선택한 영상에 오디오 트랙이 없습니다.")

            # 오디오는 Whisper 입력을 위해 16kHz PCM으로 저장
            audio_segment.write_audiofile(
                str(audio_out),
                fps=16000,
                nbytes=2,
                codec="pcm_s16le",
                verbose=False,
                logger=None,
            )
            audio_segment.close()

            # 영상은 AVSR 입력용으로만 필요하므로 오디오 제거 후 저장
            video_clip = subclip.without_audio()
            target_fps = clip.fps or 25
            video_clip.write_videofile(
                str(video_out),
                codec="libx264",
                audio=False,
                fps=target_fps,
                verbose=False,
                logger=None,
            )

            video_clip.close()
            subclip.close()

        return video_out, audio_out, usable

    def build_fusion_args(self, config_path: Path, video_path: Path, audio_path: Path, out_path: Path):
        arg_list = [
            "--config", str(config_path),
            "--video_path", str(video_path),
            "--audio_path", str(audio_path),
            "--out", str(out_path),
        ]
        return fusion_parse_args(arg_list)

if __name__ == "__main__":
    app = FusionUI()
    app.mainloop()