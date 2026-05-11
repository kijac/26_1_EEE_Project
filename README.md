"""
dashboard.py
관리 PC에서 실행하는 Flask 수신 서버 + 대시보드

실행:
    pip install flask
    python dashboard.py
    브라우저: http://localhost:5000
"""

from flask import Flask, request, jsonify, render_template_string
import json
import os
from datetime import datetime
from collections import deque

app = Flask(__name__)

LOG_FILE = "./logs/violations.jsonl"
os.makedirs("./logs", exist_ok=True)

# 최근 50건 메모리에 유지 (대시보드 실시간 표시용)
recent_events = deque(maxlen=50)


# ============================================================
# 대시보드 HTML 템플릿
# ============================================================
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta http-equiv="refresh" content="5">
<title>PPE 모니터링 대시보드</title>
<style>
  body { font-family: sans-serif; margin: 2rem; background: #f8f8f8; }
  h1   { font-size: 1.4rem; font-weight: 500; margin-bottom: 1.5rem; }
  .stats { display: flex; gap: 1rem; margin-bottom: 1.5rem; }
  .stat  { background: white; border: 1px solid #e0e0e0; border-radius: 8px;
           padding: 1rem 1.5rem; min-width: 120px; }
  .stat .num  { font-size: 2rem; font-weight: 500; }
  .stat .label { font-size: 0.8rem; color: #888; margin-top: 4px; }
  table { width: 100%; border-collapse: collapse; background: white;
          border-radius: 8px; overflow: hidden; border: 1px solid #e0e0e0; }
  th { background: #f0f0f0; padding: 10px 14px; text-align: left;
       font-size: 0.85rem; font-weight: 500; }
  td { padding: 10px 14px; font-size: 0.85rem; border-top: 1px solid #f0f0f0; }
  .badge { padding: 2px 10px; border-radius: 12px; font-size: 0.78rem; }
  .worn   { background: #e6f4ea; color: #1e7e34; }
  .carried { background: #fff3cd; color: #856404; }
  .nd      { background: #fdecea; color: #b71c1c; }
  .unknown { background: #f0f0f0; color: #555; }
  .ts { color: #888; }
</style>
</head>
<body>
<h1>PPE 모니터링 대시보드</h1>

<div class="stats">
  <div class="stat">
    <div class="num">{{ total }}</div>
    <div class="label">총 위반 건수</div>
  </div>
  <div class="stat">
    <div class="num">{{ today }}</div>
    <div class="label">오늘 위반</div>
  </div>
  <div class="stat">
    <div class="num" style="font-size:1rem;padding-top:0.5rem">{{ last_time }}</div>
    <div class="label">마지막 위반</div>
  </div>
</div>

<table>
  <thead>
    <tr>
      <th>시각</th>
      <th>카메라</th>
      <th>Person</th>
      <th>Helmet</th>
      <th>Vest</th>
    </tr>
  </thead>
  <tbody>
    {% for e in events %}
    <tr>
      <td class="ts">{{ e.timestamp }}</td>
      <td>{{ e.camera_id }}</td>
      <td>#{{ e.person_idx }}</td>
      <td>
        <span class="badge {{ 'worn' if e.helmet=='worn' else ('carried' if e.helmet=='carried' else ('unknown' if e.helmet=='unknown' else 'nd')) }}">
          {{ e.helmet }}
        </span>
      </td>
      <td>
        <span class="badge {{ 'worn' if e.vest=='worn' else ('unknown' if e.vest=='unknown' else 'nd') }}">
          {{ e.vest }}
        </span>
      </td>
    </tr>
    {% endfor %}
  </tbody>
</table>
<p style="margin-top:1rem;font-size:0.8rem;color:#aaa">5초마다 자동 새로고침</p>
</body>
</html>
"""


# ============================================================
# 라즈베리파이에서 POST로 위반 이벤트 수신
# ============================================================
@app.route("/log", methods=["POST"])
def receive_log():
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "no data"}), 400

    # 메모리에 추가
    recent_events.appendleft(data)

    # 파일에 기록
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')

    print(f"[LOG] {data.get('timestamp')} | "
          f"Person #{data.get('person_idx')} | "
          f"H:{data.get('helmet')} V:{data.get('vest')}")
    return jsonify({"status": "ok"}), 200


# ============================================================
# 대시보드 페이지
# ============================================================
@app.route("/")
def dashboard():
    events = list(recent_events)

    today_str = datetime.now().strftime("%Y-%m-%d")
    today_count = sum(1 for e in events if e.get("timestamp", "").startswith(today_str))
    last_time = events[0].get("timestamp", "-") if events else "-"

    # 전체 로그 수 (파일 기준)
    total = 0
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as f:
            total = sum(1 for _ in f)

    return render_template_string(
        DASHBOARD_HTML,
        events=events,
        total=total,
        today=today_count,
        last_time=last_time,
    )


@app.route("/events", methods=["GET"])
def get_events():
    """JSON API (선택사항)"""
    return jsonify(list(recent_events))


if __name__ == "__main__":
    print("대시보드 서버 시작: http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
