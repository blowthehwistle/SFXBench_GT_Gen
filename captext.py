import json
import pandas as pd
from tqdm import tqdm
import random

# -------------------------------
# 설정
# -------------------------------
JSON_PATH = "sample_clotho.json"
OUTPUT_EXCEL = "captions_mock.xlsx"

# JSON load
with open(JSON_PATH, "r", encoding="utf-8") as f:
    dataset = json.load(f)


# test caption 생성 함수
def mock_caption(audio_path):
    templates = [
        "The sound appears to be {desc}.",
        "You can hear {desc} clearly.",
        "This audio includes {desc}.",
        "It sounds like {desc} happening.",
    ]
    desc_options = [
        "birds chirping", "rain falling", "people talking",
        "waves crashing", "a dog barking", "a car passing by"
    ]
    desc = random.choice(desc_options)
    return random.choice(templates).format(desc=desc)

# Caption 생성
results = []
for item in tqdm(dataset, desc="Generating mock captions"):
    audio_path = item["audio_path"]
    caption = mock_caption(audio_path)
    results.append({
        "id": item["id"],
        "audio_path": audio_path,
        "caption": caption
    })

# Excel 저장
df = pd.DataFrame(results)
df.to_excel(OUTPUT_EXCEL, index=False)
print(f"createed caption Excel at: {OUTPUT_EXCEL}")
