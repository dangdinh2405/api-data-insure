from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from pathlib import Path
import os
from typing import Any, List, Optional
from unidecode import unidecode
import re

app = Flask(__name__)
CORS(app)

DATA_PATH = Path(os.getenv("CSV_PATH", "D:\HaiDang\data\products.csv"))

CATEGORY_SYNONYMS = {
    "accident": [
        "tai nan", "tai-nan", "tai nạn", "tai nạn cá nhân", "tai nạn lao động",
        "personal accident", "accident", "pa", "tai nạn 24/7"
    ],
    "health": [
        "suc khoe", "sức khỏe", "suc-khoe", "health", "medical", "healthcare",
        "chăm sóc sức khỏe", "viện phí", "benh vien", "hospitalization"
    ],
    "life": [
        "nhân thọ", "nhan tho", "nhan-tho", "life", "life insurance",
        "term life", "whole life", "bảo vệ tài chính", "bảo vệ gia đình"
    ],
    "critical illness": [
        "bệnh hiểm nghèo", "critical illness", "ci", "ung thư", "cancer",
        "tim mạch", "đột quỵ", "tai biến", "man tinh", "mãn tính", "bệnh nặng", "bệnh nghiêm trọng"
    ],
    "hospital": [
        "bệnh viện", "benh vien", "hospital", "hospitalization", "nằm viện",
        "viện phí", "medical cost", "inpatient", "outpatient", "ngoại trú", "nội trú", "phẫu thuật", "surgery"
    ],
    "children": [
        "trẻ em", "trẻ con", "trẻ nhỏ", "child", "children", "kid", "kids",
        "bảo hiểm trẻ em", "student insurance", "baby", "mầm non", "học sinh"
    ],
    "travel": [
        "du lịch", "du-lich", "travel", "travel insurance", "trip", "holiday",
        "tour", "overseas", "bảo hiểm du lịch", "bảo hiểm quốc tế"
    ],
    "dental": [
        "nha khoa", "dental", "răng", "răng miệng", "chăm sóc răng",
        "chỉnh nha", "niềng răng", "nha sĩ", "trám răng", "làm răng", "nha khoa thẩm mỹ"
    ],
    "vision": [
        "thị lực", "mắt", "kính", "mắt kính", "vision", "eye", "eye care",
        "optical", "optometry", "khám mắt", "bảo hiểm mắt"
    ]
}


CANONICAL = {}
for canon, syns in CATEGORY_SYNONYMS.items():
    for s in syns:
        CANONICAL[unidecode(s).lower()] = canon

df_cache: Optional[pd.DataFrame] = None

def normalize(s: Any) -> str:
    return unidecode(str(s)).lower().strip() if s is not None else ""

def _read_csv_any(path: Path) -> pd.DataFrame:
    tried = []
    for enc in ("utf-8", "utf-8-sig", "cp1258", "cp1252"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            tried.append(f"{enc}: {e}")
    raise RuntimeError(f"Không đọc được CSV tại {path}. Tried: {' | '.join(tried)}")

def load_csv() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy CSV: {DATA_PATH}")
    df = _read_csv_any(DATA_PATH)

    if "Category" not in df.columns:
        raise ValueError("CSV phải có cột 'Category' (phân biệt hoa thường).")

    df["_category_n"] = df["Category"].apply(normalize)
    return df

def ensure_loaded():
    global df_cache
    if df_cache is None:
        app.logger.info(f"Loading CSV from {DATA_PATH} ...")
        df_cache = load_csv()

@app.get("/healthz")
def healthz():
    try:
        ensure_loaded()
        rows = int(len(df_cache)) if df_cache is not None else 0
        return jsonify({"status": "ok", "rows": rows, "path": str(DATA_PATH)})
    except Exception as e:
        return jsonify({"status": "error", "detail": str(e), "path": str(DATA_PATH)}), 500

@app.post("/reload")
def reload_csv():
    global df_cache
    try:
        df_cache = load_csv()
        return jsonify({"status": "ok", "rows": len(df_cache), "path": str(DATA_PATH)})
    except Exception as e:
        return jsonify({"status": "error", "detail": str(e), "path": str(DATA_PATH)}), 400

@app.post("/recommend")
def recommend():
    try:
        ensure_loaded()
        if df_cache is None:
            return jsonify({"status": "error", "detail": "CSV chưa được nạp."}), 500

        body = request.get_json(silent=True) or {}
        categories = body.get("categories") or []

        limit_raw = body.get("limit", None)
        limit = None
        if isinstance(limit_raw, str):
            if limit_raw.strip().lower() != "all":
                try:
                    v = int(limit_raw)
                    limit = None if v <= 0 else v
                except:
                    limit = None                   
        elif isinstance(limit_raw, (int, float)):
            limit = None if int(limit_raw) <= 0 else int(limit_raw)
        else:
            limit = None

        wanted = set()
        for c in categories:
            wanted.add(CANONICAL.get(normalize(c), normalize(c)))

        mask = pd.Series(True, index=df_cache.index)
        if wanted:
            import re
            syns = []
            for canon in wanted:
                syns.extend(CATEGORY_SYNONYMS.get(canon, [canon]))
            syns_norm = [re.escape(normalize(s)) for s in syns if s]
            if syns_norm:
                cat_re = re.compile("(" + "|".join(syns_norm) + ")")
                mask = mask & df_cache["_category_n"].str.contains(cat_re, na=False, regex=False)

        filtered = df_cache[mask]
        results = filtered if limit is None else filtered.head(limit)

        cols = [c for c in df_cache.columns if c != "_category_n"]
        items = results[cols].to_dict(orient="records")
        return jsonify({"status": "ok", "count": len(items), "items": items})
    except Exception as e:
        return jsonify({"status": "error", "detail": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8001")), debug=True)
