import os
import argparse
import yaml
import torch
import torch.nn.functional as F
import math
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
import json

def load_gt_text_from_video_path(video_path: str) -> str:
    """Return Sentence_info text aligned with the clip index in the clean AIHub JSON."""
    try:
        vp = Path(video_path)
        clip_stem = vp.stem
        clip_idx = None
        try:
            clip_idx = int(clip_stem)
        except ValueError:
            pass

        sent_dir = vp.parent  # .../lip_J_xxx
        raw_dir = Path(sent_dir.as_posix())
        replacements = [
            ("/sample_data/test_preprocessed_noisy/test", "/sample_data/test"),
            ("/sample_data/test_preprocessed/test", "/sample_data/test"),
        ]
        raw_str = raw_dir.as_posix()
        for old, new in replacements:
            if old in raw_str:
                raw_str = raw_str.replace(old, new)
                break
        raw_dir = Path(raw_str)
        gt_json = raw_dir.parent / f"{raw_dir.name}.json"
        if not gt_json.exists():
            alt_path = raw_dir / f"{raw_dir.name}.json"
            if alt_path.exists():
                gt_json = alt_path
        if not gt_json.exists():
            print(f"[WARN] GT json missing: {gt_json}")
            return ""

        with open(gt_json, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if isinstance(meta, list) and meta:
            meta = meta[0]
        sentences = meta.get("Sentence_info") or []
        if not sentences:
            return ""

        if clip_idx is not None:
            target_id = clip_idx + 1  # Sentence_info IDs start at 1
            for sentence in sentences:
                try:
                    if int(sentence.get("ID", -1)) == target_id:
                        return sentence.get("sentence_text", "").strip()
                except (TypeError, ValueError):
                    continue
            if 0 <= clip_idx < len(sentences):
                return sentences[clip_idx].get("sentence_text", "").strip()

        return sentences[0].get("sentence_text", "").strip()
    except Exception as e:
        print(f"[WARN] GT text load fail: {e}")
        return ""


def load_checkpoint(model, checkpoint_path, device='cpu'):
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)


def prepare_avsr_inputs(config, vocab, video_path, audio_path, max_frames: int = 150, resize: int = 112):
    # Video
    vids = _parse_video(video_path, is_raw=config['raw_video'])  # T,H,W,C or T,C if not raw
    if config['raw_video']:
        # Limit frames to control memory
        T, H, W, C = vids.shape
        if T > max_frames:
            idx = torch.linspace(0, T - 1, steps=max_frames).round().long()
            vids = vids.index_select(0, idx)
            T = max_frames

        # Convert to N,C,T,H,W for 3D resizing
        vids = vids.permute(3, 0, 1, 2).unsqueeze(0)  # 1,C,T,H,W
        # Resize spatial dims to (resize, resize) to avoid OOM
        _, Cc, Tt, Hh, Ww = vids.shape
        if Hh != resize or Ww != resize:
            vids = F.interpolate(vids, size=(Tt, resize, resize), mode='trilinear', align_corners=False)
        vids = vids.squeeze(0)  # C,T,H,W
    # Audio
    signal, _ = librosa.load(audio_path, sr=config['audio_sample_rate'])
    # Raw audio transform (only reshape) if raw specified
    if config['audio_transform_method'] == 'raw':
        audio_transform = lambda x: np.expand_dims(x, 1)
    else:
        raise ValueError("Currently fusion script only supports audio_transform_method='raw' for simplicity.")
    seqs = _parse_audio(signal, audio_transform, config['audio_normalize'])
    seqs = seqs.permute(1, 0)  # F T -> T F

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


def avsr_infer(config, model, vocab, video_path, audio_path, device="cpu", max_frames: int = 150, resize: int = 112):
    model.eval()
    vids, seqs, targets, vid_lengths, seq_lengths, target_lengths = prepare_avsr_inputs(
        config, vocab, video_path, audio_path, max_frames=max_frames, resize=resize)

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


def avsr_ctc_confidence(
    config,
    model,
    vocab,
    video_path,
    audio_path,
    device="cpu",
    max_frames: int = 150,
    resize: int = 112,
):
    """
    Compute CTC confidence and detailed frame statistics.
    Returns (confidence_float, frame_max_list, stats_dict)
    stats_dict: {mean, min, max, std, num_frames}
    """
    model.eval()
    with torch.no_grad():
        vids, seqs, targets, vid_lengths, seq_lengths, target_lengths = prepare_avsr_inputs(
            config, vocab, video_path, audio_path, max_frames=max_frames, resize=resize)
        vids = vids.to(device)
        seqs = seqs.to(device)
        targets = targets.to(device)
        outputs = model(
            vids, vid_lengths,
            seqs, seq_lengths,
            targets, target_lengths
        )
        if isinstance(outputs, tuple):
            ctc_logp = outputs[1]
        else:
            ctc_logp = outputs
        probs = torch.exp(ctc_logp)
        frame_max = probs.max(dim=-1).values.squeeze(0)  # [T]
        conf = frame_max.mean().item()
        stats = {
            'mean': conf,
            'min': frame_max.min().item(),
            'max': frame_max.max().item(),
            'std': frame_max.std(unbiased=False).item(),
            'num_frames': frame_max.numel(),
        }
        return conf, frame_max.tolist(), stats


def whisper_infer_with_conf(audio_path, model_size="small", language=None, method: str = 'avg'):
    if whisper is None:
        raise ImportError("whisper is not installed. pip install openai-whisper")
    model = whisper.load_model(model_size)
    # word_timestamps=True to increase chance of token-level probabilities
    result = model.transcribe(audio_path, language=language, word_timestamps=True)
    text = result.get('text', '').strip()
    segments = result.get('segments', []) or []

    probs = []
    seg_level_probs = []
    for seg in segments:
        token_probs = []
        # Prefer 'words' with probability if present (some whisper forks)
        words = seg.get('words') if isinstance(seg, dict) else None
        if isinstance(words, list):
            for w in words:
                if isinstance(w, dict) and 'probability' in w:
                    p = float(w['probability'])
                    p = float(min(max(p, 1e-8), 1.0))
                    token_probs.append(p)
        # Fallback to tokens: may be list of ints or dicts
        toks = seg.get('tokens') if isinstance(seg, dict) else None
        if isinstance(toks, list) and not token_probs:
            for tok in toks:
                if isinstance(tok, dict):
                    p = tok.get('probability')
                    if p is None:
                        lp = tok.get('logprob')
                        if lp is not None:
                            p = math.exp(lp)
                    if p is not None:
                        p = float(min(max(p, 1e-8), 1.0))
                        token_probs.append(p)
                # if int token id, no prob available -> skip
        if token_probs:
            probs.extend(token_probs)
        # Keep segment-level avg_logprob as fallback aggregator
        if isinstance(seg, dict) and (seg.get('avg_logprob') is not None):
            p_seg = float(math.exp(seg['avg_logprob']))
            p_seg = float(min(max(p_seg, 1e-8), 1.0))
            seg_level_probs.append(p_seg)

    used_fallback = False
    if probs:
        if method == 'min':
            conf = float(min(probs))
        elif method == 'prod':
            conf = float(math.exp(sum(math.log(p) for p in probs) / len(probs)))
        else:
            conf = float(sum(probs) / len(probs))
    else:
        if seg_level_probs:
            if method == 'min':
                conf = float(min(seg_level_probs))
            elif method == 'prod':
                conf = float(math.exp(sum(math.log(p) for p in seg_level_probs) / len(seg_level_probs)))
            else:
                conf = float(sum(seg_level_probs) / len(seg_level_probs))
            used_fallback = True
        else:
            avg_lp = result.get('avg_logprob', None)
            conf = float(math.exp(avg_lp)) if avg_lp is not None else 0.0
            used_fallback = True

    stt_stats = {
        'method': method,
        'num_tokens': len(probs),
        'num_segments': len(segments),
        'used_fallback': used_fallback,
        'token_mean': float(sum(probs) / len(probs)) if probs else None,
        'token_min': float(min(probs)) if probs else None,
        'token_max': float(max(probs)) if probs else None,
        'geometric_mean': float(math.exp(sum(math.log(p) for p in probs) / len(probs))) if probs else None,
    }
    return text, conf, probs, stt_stats


def fuse_transcripts(avsr_text, whisper_text, weight_avsr=0.2, weight_whisper=0.8):
    """
    Simple character-level alignment fusion.
    Strategy:
      - If one string is contained in the other -> return longer (assume added detail).
      - Else align; for mismatched blocks choose source based on weight (higher weight wins).
    """
    debug_info = []
    debug_info.append(f"weights(avsr={weight_avsr}, whisper={weight_whisper})")
    if not avsr_text:
        return whisper_text
    if not whisper_text:
        return avsr_text
    if avsr_text in whisper_text:
        debug_info.append("avsr in whisper -> choose whisper")
        print("[FusionDebug] " + " | ".join(debug_info))
        return whisper_text
    if whisper_text in avsr_text:
        debug_info.append("whisper in avsr -> choose avsr")
        print("[FusionDebug] " + " | ".join(debug_info))
        return avsr_text

    sm = SequenceMatcher(a=avsr_text, b=whisper_text)
    fused = []
    for opcode, a0, a1, b0, b1 in sm.get_opcodes():
        if opcode == 'equal':
            fused.append(avsr_text[a0:a1])
            debug_info.append(f"equal[{a0}:{a1}]='{avsr_text[a0:a1]}'")
        else:
            seg_a = avsr_text[a0:a1]
            seg_b = whisper_text[b0:b1]
            # Heuristic: prefer longer if weights nearly equal
            if weight_avsr == weight_whisper:
                chosen = seg_a if len(seg_a) >= len(seg_b) else seg_b
                chosen_src = 'avsr' if chosen == seg_a else 'whisper'
            else:
                chosen = seg_a if weight_avsr > weight_whisper else seg_b
                chosen_src = 'avsr' if chosen == seg_a else 'whisper'
            debug_info.append(f"{opcode} a:'{seg_a}' b:'{seg_b}' -> {chosen_src}:'{chosen}'")
            fused.append(chosen)
    result = ''.join(fused)
    debug_info.append(f"result='{result}'")
    print("[FusionDebug] " + " | ".join(debug_info))
    return result


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
    p.add_argument('--max_frames', type=int, default=150, help='Max frames to sample from video to prevent OOM')
    p.add_argument('--resize', type=int, default=112, help='Resize shorter spatial side to this (square)')
    p.add_argument('--stt_conf_method', choices=['avg','min','prod'], default='avg', help='Aggregate Whisper token probs')
    p.add_argument('--threshold_low', type=float, default=0.3, help='STT low threshold')
    p.add_argument('--threshold_high', type=float, default=0.7, help='AVSR high threshold')
    p.add_argument('--log_detail', action='store_true', help='Include per-frame and per-token probability arrays in JSON output')
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

    avsr_text, avsr_jaso = avsr_infer(
        config, model, vocab, args.video_path, args.audio_path,
        device=args.device, max_frames=args.max_frames, resize=args.resize)

    try:
        avsr_conf, avsr_frame_max_list, avsr_stats = avsr_ctc_confidence(
            config, model, vocab, args.video_path, args.audio_path,
            device=args.device, max_frames=args.max_frames, resize=args.resize)
    except Exception as e:
        print(f"[WARN] Failed to compute AVSR CTC confidence: {e}")
        avsr_conf, avsr_frame_max_list, avsr_stats = None, [], {}

    if whisper is None:
        print("[WARN] Whisper not installed; skipping whisper transcription. pip install openai-whisper")
        whisper_text, stt_conf, stt_probs, stt_stats = '', None, [], {}
    else:
        whisper_text, stt_conf, stt_probs, stt_stats = whisper_infer_with_conf(
            args.audio_path, model_size=args.whisper_model, language=args.language, method=args.stt_conf_method)

    fused = fuse_transcripts(avsr_text, whisper_text, args.avsr_weight, args.whisper_weight)

    # Selection by confidence thresholds and comparison
    stt_c = stt_conf if stt_conf is not None else 0.0
    avsr_c = avsr_conf if avsr_conf is not None else 0.0
    if stt_c > args.threshold_high:          # T_good
        selected = whisper_text
        decision = 'STT_conf_good'
    elif stt_c < args.threshold_low:         # T_bad
        selected = avsr_text
        decision = 'AVSR_conf_bad_stt'
    else:
        selected = whisper_text              # 중간 구간도 STT
        decision = 'STT_mid_range'

    print("\n=== Fusion Inference Result ===")
    print(f"AVSR   : {avsr_text}")
    if avsr_conf is not None:
        print(f"AVSR_CTC_conf: {avsr_conf:.4f}")
    print(f"Whisper: {whisper_text}")
    if stt_conf is not None:
        print(f"STT_conf({args.stt_conf_method}): {stt_c:.4f}")
    print(f"Fused  : {fused}")
    print(f"Decision Path: {decision}")
    print(f"Selected({decision}): {selected}")

    try:
        gt_text = load_gt_text_from_video_path(args.video_path)

        with open(args.out, 'w', encoding='utf-8') as fw:
            debug_section = {
                'avsr_stats': avsr_stats,
                'stt_stats': stt_stats,
                'thresholds': {
                    'low': args.threshold_low,
                    'high': args.threshold_high,
                },
                'conditions': {
                    'cond_low_high_met': (stt_conf is not None and stt_conf < args.threshold_low and avsr_conf is not None and avsr_conf > args.threshold_high),
                    'stt_gt_avsr': (stt_conf is not None and avsr_conf is not None and stt_conf > avsr_conf),
                    'fallback_used': decision == 'AVSR_by_fallback',
                },
                'decision_path': decision,
            }
            if args.log_detail:
                debug_section['avsr_frame_max'] = avsr_frame_max_list
                debug_section['stt_token_probs'] = stt_probs

            json.dump({
                'video': args.video_path,
                'audio': args.audio_path,
                'gt_text': gt_text,  # ← 추가
                'avsr_text': avsr_text,
                'avsr_ctc_confidence': avsr_conf,
                'whisper_text': whisper_text,
                'stt_confidence': stt_conf,
                'fused_text': fused,
                'selected_text': selected,
                'avsr_jaso': avsr_jaso,
                'weights': {'avsr': args.avsr_weight, 'whisper': args.whisper_weight},
                'selection': {
                    'decision': decision,
                    'threshold_low': args.threshold_low,
                    'threshold_high': args.threshold_high,
                    'stt_conf_method': args.stt_conf_method,
                },
                'debug': debug_section,
            }, fw, ensure_ascii=False, indent=2)


        print(f"Saved fused output JSON -> {args.out}")
    except Exception as e:
        print(f"[WARN] Failed to save JSON: {e}")


if __name__ == '__main__':
    main()
