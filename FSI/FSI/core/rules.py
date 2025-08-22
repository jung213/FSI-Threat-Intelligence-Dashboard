import yaml
import numpy as np

def load_rules(path="rules.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def apply_rules(df, rules):
    # 예시 규칙: 금액>5,000,000 → +0.2 점수 가산, 첫 해외국가 → +0.3, 새 디바이스 → +0.1
    score_boost = np.zeros(len(df))
    for i, row in df.reset_index(drop=True).iterrows():
        if "amount_gt" in rules and row["amount"] > rules["amount_gt"]:
            score_boost[i] += rules.get("amount_boost", 0.2)
        if rules.get("first_foreign_country") and row["country"] not in ("KR",):
            # 간단 처리: 계정의 첫 foreign 여부는 실제론 히스토리 필요
            score_boost[i] += rules.get("foreign_boost", 0.3)
        if rules.get("device_change_boost") and row.get("device_change",0)==1:
            score_boost[i] += rules.get("device_boost", 0.1)
    return score_boost
