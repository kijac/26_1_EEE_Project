# alert.py
import json, os, time
from datetime import datetime
from collections import deque

class AlertConfig:
    COOLDOWN_SEC = 10.0
    LOG_DIR      = "./logs"
    LOG_FILE     = "violations.jsonl"
    BUZZER_PIN   = 18

class ViolationAlert:
    def __init__(self):
        os.makedirs(AlertConfig.LOG_DIR, exist_ok=True)
        self._log_path  = os.path.join(AlertConfig.LOG_DIR, AlertConfig.LOG_FILE)
        self._cooldowns = {}
        self._total     = 0
        self._buzzer    = Buzzer(AlertConfig.BUZZER_PIN)

    def process(self, person_idx, helmet, vest):
        now = time.time()
        if now - self._cooldowns.get(person_idx, 0) < AlertConfig.COOLDOWN_SEC:
            return   # 쿨다운 중 → 무시

        self._cooldowns[person_idx] = now
        self._total += 1

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[VIOLATION] {ts} | Person#{person_idx} | H:{helmet} V:{vest}")

        # 로컬 로그만
        with open(self._log_path, 'a') as f:
            f.write(json.dumps({
                "timestamp":  ts,
                "person_idx": person_idx,
                "helmet":     helmet,
                "vest":       vest,
            }) + '\n')

        # 부저
        self._buzzer.beep(0.5)

    def stats(self):
        return {"total": self._total, "log": self._log_path}