import replicate
import pandas as pd
from tqdm import tqdm

# 환경 변수: Replicate API Token
import os
os.environ["REPLICATE_API_TOKEN"] = "YOUR_API_TOKEN"


# 샘플 데이터 (JSON 형태)
dataset = [
    {"id": "sample1", "audio_url": "https://sample.link/audio1.wav"},
    {"id": "sample2", "audio_url": "https://sample.link/audio2.wav"},
    {"id": "sample3", "audio_url": "https://sample.link/audio3.wav"},
]

# Replicate api run
def generate_caption(audio_url):
    """
    audio_url: 공개 접근 가능한 URL
    return: 모델이 생성한 caption
    """
    # AF3 api
    output = replicate.run(
        "zsxkib/audio-flamingo-3:latest",
        input={"audio_url": audio_url}
    )
    return output

# Main Pipeline 
results = []
for item in tqdm(dataset, desc="Generating captions"):
    audio_url = item["audio_url"]
    try:
        caption = generate_caption(audio_url)
    except Exception as e:
        caption = f"[Error] {e}"
    results.append({
        "id": item["id"],
        "audio_url": audio_url,
        "caption": caption
    })

# Excel로 저장
df = pd.DataFrame(results)
df.to_excel("captions_from_url.xlsx", index=False)
print("Caption created: captions_from_url.xlsx")
