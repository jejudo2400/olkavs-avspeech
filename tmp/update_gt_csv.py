import csv
from pathlib import Path

CSV_PATH = Path(r"C:\Users\2302\Documents\GitHub\olkavs-avspeech\fusion_summary.csv")
GT_MAP = {
    0: "회사 취직 교수님께서 과제로 내주신 에세이에 엄청 애쓴 내가 너무 한심해.",
    1: "올해 경기가 안 좋아 아 안 좋아서 그런지 내 월 월급 인상은 없었어.",
    2: "나이가 많다고 이번 프로젝트를 맡지 못했어. 정말 화가 나.",
    3: "회사 이번에 중년의 X 신입이 들어왔어. 회사에서 나이를 안 보고 뽑는 건 알았으나 중년의 신입이라고 하니 약간은 불편했지.",
    4: "내 직업이 십 년 안에 기계로 대체될 거래.",
    5: "얼마 전 회사 어 얼마 전 면접 본 회사에서 합격했다고 오늘 연락이 왔어. 너무 신나.",
    6: "어제 회사에서 정말 창피한 일을 경험했어.",
    7: "여름에는 땀을 많이 흘려서 아침에 샤워하지 않나? 어떤 직원은 아침부터 땀 냄새가 구역질 날 정도로 진동을 해.",
    8: "회사 일을 열심히 해봐도 이게 다 무슨 의미가 있는 건지 하는 회의감이 많이 들어.",
    9: "진짜 공부 못하던 친구가 우리 회사에 합격했대. 너무 배 아파.",
    10: "오늘 회사에서 보너스가 나와서 소고기를 사 먹을 꺼야.",
    11: "이번에는 최종 합격했다고 생각했는데 또 떨어졌어.",
    12: "요새 코로나로 일자리가 많이 부족한 것 같아. 친구가 취업 했는데 정말 화가 나.",
    13: "오늘 김 부장이 직원들 앞에서 내게 모욕감을 준 것이 뇌리에서 이 잊히지 않아 잊히지 않아서 화가 나.",
    14: "우리나라에서 성 수소자들의 취업이 쉽지는 않은 것 같애.",
    15: "직장에서 무급 휴가를 가려고 해서 가라고 해서 충격받았어.",
    16: "회사 사정이 안 좋아졌대. 나 지방으로 발령이 났어. 갑자기 생긴 일이라 당황스러워.",
    17: "이번에 새로 맡게 된 프로젝트 하면서 너무 힘들어. 예전에는 거뜬히 해 냈는데 이제는 체력이 부족해서 훨씬 힘드네.",
    18: "직장 상사가 나한테 자꾸 내 업무가 아닌 잡일을 시켜서 짜증 나.",
    19: "나는 회사에 입사하면 내 삶이 달라질 줄 알았는데 그냥 더 피곤하기만 해.",
    20: "고객들이 클라우드 환경으로 전환을 해.",
}

def update_csv():
    with CSV_PATH.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    for row in rows:
        stem = Path(row.get("video_path", "")).stem
        clip = stem.split("_", 1)[0]
        if clip.isdigit():
            idx = int(clip)
            if idx in GT_MAP:
                row["gt_text"] = GT_MAP[idx]

    with CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

if __name__ == "__main__":
    update_csv()
