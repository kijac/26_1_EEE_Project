"""
main.py
라즈베리파이 5 + Hailo HAT 실행 파일

실행:
    python main.py

의존성:
    pip install ultralytics requests opencv-python-headless
    (Hailo 변환 후에는 ultralytics 대신 HailoRT 사용)
"""

import cv2
import time
import threading
import signal
import sys
from ultralytics import YOLO

from detector_v2 import PPEEnsemble, Config
from visualizer import draw_results


# ============================================================
# 설정
# ============================================================
DET_MODEL_PATH  = "./models/best.pt"        # YOLOv8n detection
POSE_MODEL_PATH = "./models/yolov8n-pose.pt"
CAMERA_ID       = 0                          # Pi Camera: 0
DISPLAY         = True                       # 모니터 연결 시 True
BUZZER_PIN      = 18                         # GPIO 핀 (부저)


# ============================================================
# 부저 (GPIO, RPi에서만 동작)
# ============================================================
class Buzzer:
    def __init__(self, pin: int):
        self.pin = pin
        self._available = False
        try:
            import RPi.GPIO as GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(pin, GPIO.OUT)
            self._gpio = GPIO
            self._available = True
            print(f"[Buzzer] GPIO {pin} 초기화 완료")
        except ImportError:
            print("[Buzzer] RPi.GPIO 없음 → 부저 비활성화 (Colab/PC 환경)")

    def beep(self, duration: float = 0.5):
        if not self._available:
            print(f"[Buzzer] BEEP! ({duration}s)")
            return
        t = threading.Thread(target=self._beep_thread, args=(duration,), daemon=True)
        t.start()

    def _beep_thread(self, duration):
        self._gpio.output(self.pin, self._gpio.HIGH)
        time.sleep(duration)
        self._gpio.output(self.pin, self._gpio.LOW)

    def cleanup(self):
        if self._available:
            self._gpio.cleanup()


# ============================================================
# 메인 루프
# ============================================================
def main():
    print("=== PPE Detection System 시작 ===")

    # 모델 로드
    print("모델 로딩 중...")
    det_model  = YOLO(DET_MODEL_PATH)
    pose_model = YOLO(POSE_MODEL_PATH)
    ensemble   = PPEEnsemble(det_model, pose_model)  # alert/buzzer 내장

    # 카메라
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("[ERROR] 카메라를 열 수 없습니다.")
        sys.exit(1)
    print(f"카메라 시작 (ID: {CAMERA_ID})")

    # Graceful shutdown
    running = True
    def handle_signal(sig, frame):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT,  handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    frame_count = 0
    fps_acc     = 0.0
    fps_display = 0.0

    print("실행 중... (Ctrl+C로 종료)")

    while running:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] 프레임 읽기 실패")
            time.sleep(0.05)
            continue

        frame_count += 1
        t0 = time.time()

        # 앙상블 추론
        results = ensemble.predict(frame)

        # 위반 처리 (alert/buzzer/log는 ensemble.predict() 내부에서 자동 처리)

        # FPS 계산
        dt = time.time() - t0
        fps_acc = fps_acc * 0.9 + (1.0/dt if dt>0 else 0) * 0.1
        if frame_count % 30 == 0:
            fps_display = fps_acc

        # 시각화 + 출력
        if DISPLAY:
            vis, vio_count = draw_results(frame, results)
            cv2.putText(vis, f"FPS:{fps_display:.1f} | Vio:{vio_count}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("PPE Detection", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # 종료 처리
    print("\n=== 시스템 종료 ===")
    stats = ensemble.alert.stats()
    print(f"총 위반 감지: {stats['total']}건")
    print(f"로그 저장: {stats['log']}")

    cap.release()
    if DISPLAY:
        cv2.destroyAllWindows()
    ensemble.cleanup()


if __name__ == "__main__":
    main()
