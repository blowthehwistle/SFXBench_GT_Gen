import os
import time
import json
import math
import random
from typing import List, Dict, Any, Optional

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import replicate

# -----------------------------
# 0) 환경 변수 및 기본 설정
# -----------------------------
# .env 파일에 REPLICATE_API_TOKEN=xxxx 저장해두면 자동 로드됩니다.
load_dotenv()
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
if not REPLICATE_API_TOKEN:
    raise RuntimeError("No valid REPLICATE_API_TOKEN")

# 모델 버전(해시) 고정
AF3_MODEL = "zsxkib/audio-flamingo-3:419bdd5ed04ba4e4609e66cc5082f6564e9d2c0836f9a286abe74bc20a357b84"

# API 속도 제한 보호용(필요시 조정)
MAX_RETRIES = 5
BASE_BACKOFF = 1.5  # 지수 백오프 배수
SLEEP_BETWEEN_CALLS = 0.2  # 각 호출 사이 딜레이(초). 과금/레이트리밋 상황에 맞게 조정.

# -----------------------------
# 1) 프롬프트 템플릿
# -----------------------------
# 벤치마크용 구조화 출력을 유도하는 SFX 라벨링 프롬프트 예시
SFX_PROMPT = """You are an expert SFX annotator.
Analyze the audio and output a JSON array of events.
Each event must include: "class", "start", "end", and "description".
- "class": short SFX label (e.g., "door_slam", "footstep", "metal_clang")
- "start" / "end": seconds (float). If unknown, estimate.
- "description": concise textual description.

Output JSON only, with no extra text.
"""

SYSTEM_PROMPT = ""  # 필요 없으면 비워두기


# -----------------------------
# 2) 단건 실행 함수 (재시도 포함)
# -----------------------------
def af3_caption_one(audio_url: str,
                    prompt: str = SFX_PROMPT,
                    system_prompt: str = SYSTEM_PROMPT,
                    temperature: float = 0.0,
                    max_length: int = 0) -> Dict[str, Any]:
    """
    단일 오디오 URL에 대해 AF3 캡션/라벨 생성 후 dict로 반환.
    실패 시 지수 백오프로 재시도.
    """
    last_err: Optional[Exception] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            output = replicate.run(
                AF3_MODEL,
                input={
                    "audio": audio_url,
                    "prompt": prompt,
                    "max_length": max_length,
                    "temperature": temperature,
                    "system_prompt": system_prompt,
                    "enable_thinking": True,
                },
            )
            # Replicate 출력 형태가 문자열(텍스트)일 수 있으므로 JSON 파싱 시도
            parsed = None
            if isinstance(output, str):
                txt = output.strip()
                # JSON만 출력하도록 프롬프트를 강제했지만, 혹시 앞뒤 공백/개행 방지
                try:
                    parsed = json.loads(txt)
                except Exception:
                    # JSON이 아니면 그대로 텍스트로 저장
                    parsed = txt
            else:
                parsed = output
            return {"success": True, "output": parsed, "raw": output, "error": None}

        except Exception as e:
            last_err = e
            # 지수 백오프
            sleep_s = (BASE_BACKOFF ** (attempt - 1)) + random.uniform(0, 0.3)
            time.sleep(sleep_s)

    return {"success": False, "output": None, "raw": None, "error": str(last_err)}


# -----------------------------
# 3) 배치 실행 함수
# -----------------------------
def run_batch(input_csv: str,
              audio_col: str = "audio_url",
              id_col: Optional[str] = "id",
              out_xlsx: str = "af3_captions.xlsx",
              out_jsonl: Optional[str] = "af3_captions.jsonl",
              limit: Optional[int] = None) -> pd.DataFrame:
    """
    CSV에서 오디오 URL 목록을 읽어 AF3 캡션/라벨을 생성하고, 엑셀/JSONL로 저장.
    CSV 예시 컬럼: id, audio_url
    """
    df = pd.read_csv(input_csv)
    if limit:
        df = df.head(limit).copy()

    # 결과 컬럼 준비
    df["af3_success"] = False
    df["af3_output_json"] = None
    df["af3_raw_text"] = None
    df["af3_error"] = None

    rows = df.to_dict(orient="records")

    for i, row in enumerate(tqdm(rows, desc="AF3 captioning", ncols=100)):
        audio_url = row.get(audio_col)
        if not isinstance(audio_url, str) or not audio_url.strip():
            row["af3_success"] = False
            row["af3_error"] = "empty_audio_url"
            continue

        # API 호출(재시도 포함)
        res = af3_caption_one(audio_url)

        row["af3_success"] = bool(res["success"])
        if res["success"]:
            # 출력이 JSON(dict/list)이면 문자열로 직렬화하여 저장(엑셀 호환)
            if isinstance(res["output"], (dict, list)):
                row["af3_output_json"] = json.dumps(res["output"], ensure_ascii=False)
                row["af3_raw_text"] = None
            else:
                # 텍스트 그대로
                row["af3_output_json"] = None
                row["af3_raw_text"] = str(res["output"])
        else:
            row["af3_error"] = res["error"]

        # API 보호용 소량 딜레이
        time.sleep(SLEEP_BETWEEN_CALLS)

    # dict 리스트를 DataFrame으로 복원
    out_df = pd.DataFrame(rows)

    # 3-1) Excel 저장 (열 너비/한글 유니코드 OK)
    #  - af3_output_json 또는 af3_raw_text 중 하나에 결과가 들어감
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        out_df.to_excel(writer, index=False, sheet_name="af3_results")

    # 3-2) JSONL(optional) 저장 — 구조화 후후처리 파이프라인에서 쓰기 좋음
    if out_jsonl:
        with open(out_jsonl, "w", encoding="utf-8") as f:
            for _, r in out_df.iterrows():
                obj = {
                    "id": r[id_col] if id_col and id_col in r else None,
                    "audio_url": r[audio_col],
                    "success": bool(r["af3_success"]),
                    "output_json": r["af3_output_json"],
                    "raw_text": r["af3_raw_text"],
                    "error": r["af3_error"],
                }
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    return out_df


# -----------------------------
# 4) 진입점
# -----------------------------
if __name__ == "__main__":
    """
    예시 CSV 포맷 (sfx_dataset.csv)
    --------------------------------
    id,audio_url
    0001,https://.../clip1.mp3
    0002,https://.../clip2.mp3
    ...
    """
    INPUT_CSV = "sfx_dataset.csv"     # 당신의 데이터셋 CSV 경로
    OUT_XLSX = "af3_captions.xlsx"
    OUT_JSONL = "af3_captions.jsonl"

    # 500개만 돌리고 싶으면 limit=500
    df_result = run_batch(
        input_csv=INPUT_CSV,
        audio_col="audio_url",
        id_col="id",
        out_xlsx=OUT_XLSX,
        out_jsonl=OUT_JSONL,
        limit=500
    )

    print(f"Done. Saved to: {OUT_XLSX} / {OUT_JSONL}")