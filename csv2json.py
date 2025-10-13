import pandas as pd
import json
import os

# -------------------------------
# 설정
# -------------------------------
CSV_PATH = "clotho_captions_development.csv"
AUDIO_DIR = "clotho_audio_development"
JSON_PATH = "sample_clotho.json"

# CSV 읽기
df = pd.read_csv(CSV_PATH)

# 테스트용 5개(head부터 5개)
df_sample = df.head(5)

# -------------------------------
# JSON 변환
# -------------------------------
dataset = []
for _, row in df_sample.iterrows():
    audio_path = os.path.join(AUDIO_DIR, row["file_name"])
    dataset.append({
        "id": row["file_name"],
        "audio_path": audio_path,
        "captions": [
            row["caption_1"],
            row["caption_2"],
            row["caption_3"],
            row["caption_4"],
            row["caption_5"]
        ]
    })

# JSON 저장
with open(JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=2, ensure_ascii=False)

print(f"created sample_clotho.JSON at: {JSON_PATH}")
