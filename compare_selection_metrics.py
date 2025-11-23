import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple


def normalize(text: str) -> str:
    return text.strip()


def levenshtein(a: List[str], b: List[str]) -> int:
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    curr = [0] * (lb + 1)
    for i, token_a in enumerate(a, 1):
        curr[0] = i
        for j, token_b in enumerate(b, 1):
            cost = 0 if token_a == token_b else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev
    return prev[lb]


def cer(ref: str, hyp: str) -> float:
    ref_norm = normalize(ref)
    hyp_norm = normalize(hyp)
    if not ref_norm:
        return 0.0 if not hyp_norm else 1.0
    return levenshtein(list(ref_norm), list(hyp_norm)) / len(ref_norm)


def wer(ref: str, hyp: str) -> float:
    ref_tokens = normalize(ref).split()
    hyp_tokens = normalize(hyp).split()
    if not ref_tokens:
        return 0.0 if not hyp_tokens else 1.0
    return levenshtein(ref_tokens, hyp_tokens) / len(ref_tokens)


def safe_float(value: str) -> Tuple[bool, float]:
    if value in (None, ''):
        return False, 0.0
    try:
        return True, float(value)
    except ValueError:
        return False, 0.0


def pick_confidence_based(row: Dict[str, str]) -> str:
    has_avsr, avsr_c = safe_float(row.get('avsr_conf'))
    has_stt, stt_c = safe_float(row.get('stt_conf'))
    avsr_text = row.get('avsr_text', '')
    whisper_text = row.get('whisper_text', '')

    if has_avsr and has_stt:
        return whisper_text if stt_c >= avsr_c else avsr_text
    if has_stt:
        return whisper_text
    if has_avsr:
        return avsr_text
    return row.get('selected_text', '') or whisper_text or avsr_text


def build_rows(csv_path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with csv_path.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt = row.get('gt_text', '')
            if not gt:
                continue
            after_text = row.get('selected_text', '')
            before_text = pick_confidence_based(row)
            avsr_text = row.get('avsr_text', '')
            stt_text = row.get('whisper_text', '')
            rows.append({
                'json_path': row.get('json_path', ''),
                'avsr_text': avsr_text,
                'stt_text': stt_text,
                'selected_text': after_text,
                'gt_text': gt,
                'cer_before': f"{cer(gt, before_text):.6f}" if before_text else '',
                'wer_before': f"{wer(gt, before_text):.6f}" if before_text else '',
                'cer_after': f"{cer(gt, after_text):.6f}" if after_text else '',
                'wer_after': f"{wer(gt, after_text):.6f}" if after_text else '',
            })
    return rows


def summarize(rows: List[Dict[str, str]]) -> Dict[str, float]:
    def collect(key: str) -> float:
        vals = [float(r[key]) for r in rows if r.get(key)]
        return sum(vals) / len(vals) if vals else 0.0

    return {
        'samples': len(rows),
        'avg_cer_before': collect('cer_before'),
        'avg_wer_before': collect('wer_before'),
        'avg_cer_after': collect('cer_after'),
        'avg_wer_after': collect('wer_after'),
    }


def write_csv(rows: List[Dict[str, str]], summary: Dict[str, float], out_path: Path) -> None:
    if not rows:
        print('[WARN] No rows to write.')
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with out_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    summary_path = out_path.with_name(out_path.stem + '_summary.csv')
    with summary_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    print(f'Saved detail CSV -> {out_path}')
    print(f'Saved summary CSV -> {summary_path}')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Compare CER/WER of selected text before/after fusion decision logic.')
    parser.add_argument('--summary_csv', type=Path, required=True,
                        help='fusion_summary.csv path')
    parser.add_argument('--out_csv', type=Path, default=Path('selection_cer_wer.csv'),
                        help='Output CSV path (workspace-relative).')
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.summary_csv.exists():
        raise FileNotFoundError(f'Summary CSV not found: {args.summary_csv}')
    rows = build_rows(args.summary_csv)
    summary = summarize(rows)
    write_csv(rows, summary, args.out_csv)


if __name__ == '__main__':
    main()
