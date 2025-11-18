"""
Fusion Inference: Combine AVSR model and Whisper transcription with a weighted heuristic.

Usage (example):
  python fusion_inference.py \
      --config configs/inference.yaml \
      --video_path sample_data/processed_videos/my1_crop.mp4 \
      --audio_path sample_data/my1.wav \
      --avsr_weight 0.6 --whisper_weight 0.4

Prerequisites:
  pip install openai-whisper moviepy pyyaml
  (FFmpeg must be installed and on PATH for Whisper/video decoding.)

Notes:
  - This script loads a single video/audio pair; if your AVSR checkpoint expects raw video frames
    ensure config raw_video=True and pass an mp4. If it expects pre-extracted numpy features set raw_video=False
    and pass a .npy path.
  - Fusion here is a simple alignment heuristic; upgrade later with token-level probabilities if exposed.
"""

import os
import argparse
import yaml
import torch
from pathlib import Path
from difflib import SequenceMatcher

try:
    import whisper
except ImportError:
    whisper = None  # We will warn later.

from vocabulary.utils import KsponSpeechVocabulary, grp2char
from avsr.utils.model_builder import build_model
from avsr.utils.getter import select_search
from dataset.dataset import _parse_video, _parse_audio  # avoid _parse_transcript for empty dummy

import librosa
import numpy as np


def load_checkpoint(model, checkpoint_path, device='cpu'):
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)


def prepare_avsr_inputs(config, vocab, video_path, audio_path):
    # Video
    vids = _parse_video(video_path, is_raw=config['raw_video'])
    if config['raw_video']:
        vids = vids.permute(3,0,1,2)  # T H W C -> C T H W
    # Audio
    signal, _ = librosa.load(audio_path, sr=config['audio_sample_rate'])
    # Raw audio transform (only reshape) if raw specified
    if config['audio_transform_method'] == 'raw':
        audio_transform = lambda x: np.expand_dims(x,1)
    else:
        raise ValueError("Currently fusion script only supports audio_transform_method='raw' for simplicity.")
    seqs = _parse_audio(signal, audio_transform, config['audio_normalize'])
    seqs = seqs.permute(1,0)  # F T -> T F

    # Dummy transcript tensor: just <sos> <eos>
    import torch as _torch
    targets = _torch.tensor([vocab.sos_id, vocab.eos_id], dtype=_torch.long)

    vids = vids.unsqueeze(0)
    seqs = seqs.unsqueeze(0)
    targets = targets.unsqueeze(0)
    vid_lengths = torch.full((1,), vids.size(1), dtype=torch.int32)
    seq_lengths = torch.full((1,), seqs.size(1), dtype=torch.int32)
    target_lengths = torch.full((1,), targets.size(1), dtype=torch.int32)
    return vids, seqs, targets, vid_lengths, seq_lengths, target_lengths


def avsr_infer(config, model, vocab, video_path, audio_path, device="cpu"):
    model.eval()
    vids, seqs, targets, vid_lengths, seq_lengths, target_lengths = prepare_avsr_inputs(
        config, vocab, video_path, audio_path)

    vids = vids.to(device)
    seqs = seqs.to(device)
    # Build search object
    search = select_search(
        model=model,
        vocab_size=len(vocab),
        pad_id=vocab.pad_id,
        sos_id=vocab.sos_id,
        eos_id=vocab.eos_id,
        unk_id=vocab.unk_id,
        method=config['search_method'],
        max_len=config['max_len'],
        ctc_rate=config['ctc_rate'],
        mp_num=1,
    )

    # Depending on mode reduce modalities
    mode = config.get('mode', 'avsr')
    if mode == 'asr':
        video_inputs, video_input_lengths = None, None
    else:
        video_inputs, video_input_lengths = vids, vid_lengths
    if mode == 'vsr':
        audio_inputs, audio_input_lengths = None, None
    else:
        audio_inputs, audio_input_lengths = seqs, seq_lengths

    with torch.no_grad():
        outputs, output_lengths = search(
            video_inputs, video_input_lengths,
            audio_inputs, audio_input_lengths,
            device=device,
            beam_size=config['beam_size'],
            D_end=config['EndDetect_D'],
            M_end=config['EndDetect_M'],
            mp_num=1,
        )

    pred_seq = outputs[0, :output_lengths[0]]
    jaso = vocab.label_to_string(pred_seq.unsqueeze(0))[0]
    text = grp2char(jaso)
    return text, jaso


def whisper_infer(audio_path, model_size="small", language=None):
    if whisper is None:
        raise ImportError("whisper is not installed. pip install openai-whisper")
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path, language=language)
    return result.get('text', '').strip()


def fuse_transcripts(avsr_text, whisper_text, weight_avsr=0.6, weight_whisper=0.4):
    """
    Simple character-level alignment fusion.
    Strategy:
      - If one string is contained in the other -> return longer (assume added detail).
      - Else align; for mismatched blocks choose source based on weight (higher weight wins).
    """
    if not avsr_text:
        return whisper_text
    if not whisper_text:
        return avsr_text
    if avsr_text in whisper_text:
        return whisper_text
    if whisper_text in avsr_text:
        return avsr_text

    sm = SequenceMatcher(a=avsr_text, b=whisper_text)
    fused = []
    for opcode, a0, a1, b0, b1 in sm.get_opcodes():
        if opcode == 'equal':
            fused.append(avsr_text[a0:a1])
        else:
            seg_a = avsr_text[a0:a1]
            seg_b = whisper_text[b0:b1]
            # Heuristic: prefer longer if weights nearly equal
            if weight_avsr == weight_whisper:
                chosen = seg_a if len(seg_a) >= len(seg_b) else seg_b
            else:
                chosen = seg_a if weight_avsr > weight_whisper else seg_b
            fused.append(chosen)
    return ''.join(fused)


def parse_args():
    p = argparse.ArgumentParser(description="AVSR + Whisper Fusion Inference")
    p.add_argument('--config', required=True, help='Path to AVSR inference yaml config')
    p.add_argument('--video_path', required=True, help='Video path (mp4 or npy)')
    p.add_argument('--audio_path', required=True, help='Audio wav path (can be extracted from video separately)')
    p.add_argument('--checkpoint', help='Override model_path in config')
    p.add_argument('--whisper_model', default='small', help='Whisper model size (tiny, base, small, medium, large)')
    p.add_argument('--avsr_weight', type=float, default=0.6)
    p.add_argument('--whisper_weight', type=float, default=0.4)
    p.add_argument('--language', default=None, help='Force language for Whisper')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out', default='fusion_output.json', help='Optional JSON output path')
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config, encoding='utf-8') as f:
        config = yaml.safe_load(f)
    if args.checkpoint:
        config['model_path'] = args.checkpoint
    checkpoint = config['model_path']

    vocab = KsponSpeechVocabulary(unit=config['tokenize_unit'])
    torch.manual_seed(config['random_seed'])

    model = build_model(
        vocab_size=len(vocab), pad_id=vocab.pad_id,
        architecture=config['architecture'], loss_fn=config['loss_fn'],
        front_dim=config['front_dim'],
        encoder_n_layer=config['encoder_n_layer'], encoder_d_model=config['encoder_d_model'],
        encoder_n_head=config['encoder_n_head'], encoder_ff_dim=config['encoder_ff_dim'],
        encoder_dropout_p=config['encoder_dropout_p'],
        decoder_n_layer=config['decoder_n_layer'], decoder_d_model=config['decoder_d_model'],
        decoder_n_head=config['decoder_n_head'], decoder_ff_dim=config['decoder_ff_dim'],
        decoder_dropout_p=config['decoder_dropout_p'],
        pass_visual_frontend=not config['raw_video'], verbose=True
    )
    load_checkpoint(model, checkpoint, device='cpu')  # load to cpu first
    model.to(args.device)

    avsr_text, avsr_jaso = avsr_infer(config, model, vocab, args.video_path, args.audio_path, device=args.device)

    if whisper is None:
        print("[WARN] Whisper not installed; skipping whisper transcription. pip install openai-whisper")
        whisper_text = ''
    else:
        whisper_text = whisper_infer(args.audio_path, model_size=args.whisper_model, language=args.language)

    fused = fuse_transcripts(avsr_text, whisper_text, args.avsr_weight, args.whisper_weight)

    print("\n=== Fusion Inference Result ===")
    print(f"AVSR   : {avsr_text}")
    print(f"Whisper: {whisper_text}")
    print(f"Fused  : {fused}")

    try:
        import json
        with open(args.out, 'w', encoding='utf-8') as fw:
            json.dump({
                'video': args.video_path,
                'audio': args.audio_path,
                'avsr_text': avsr_text,
                'whisper_text': whisper_text,
                'fused_text': fused,
                'avsr_jaso': avsr_jaso,
                'weights': {'avsr': args.avsr_weight, 'whisper': args.whisper_weight},
            }, fw, ensure_ascii=False, indent=2)
        print(f"Saved fused output JSON -> {args.out}")
    except Exception as e:
        print(f"[WARN] Failed to save JSON: {e}")


if __name__ == '__main__':
    main()
