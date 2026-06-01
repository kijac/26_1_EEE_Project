"""
main.py
라즈베리파이 5 + Hailo HAT 실행 파일 (실시간 카메라 스트리밍 버전)

실행:
    python3 main.py   ← libcamerify 없이 직접 실행

의존성:
    sudo apt install -y python3-picamera2
"""

import cv2
import time
import threading
import signal
import sys
import numpy as np
from ultralytics import YOLO

from detector_v2 import PPEEnsemble, Config
from visualizer import draw_results


# ============================================================
# 설정
# ============================================================
DET_MODEL_PATH  = "./models/ppe_det_final.hef"       # Hailo HEF
POSE_MODEL_PATH = "./models/yolov8n-pose.pt"
DISPLAY         = True
BUZZER_PIN      = 18
CAP_WIDTH       = 3280
CAP_HEIGHT      = 2464


# ============================================================
# Hailo HEF 추론 래퍼
# YOLO()와 동일한 인터페이스 → detector_v2.py 수정 불필요
# ============================================================
class HailoBox:
    def __init__(self, x1, y1, x2, y2, conf, cls_id):
        self.xyxy = [np.array([x1, y1, x2, y2])]
        self.conf = [np.array(conf)]
        self.cls  = [np.array(cls_id)]


class HailoResult:
    def __init__(self, detections, names, orig_shape, imgsz, conf_thresh):
        self.names = names
        h_orig, w_orig = orig_shape[:2]
        scale_x = w_orig / imgsz
        scale_y = h_orig / imgsz

        boxes = []
        class_names = list(names.values())  # ['helmet', 'person', 'vest']
        for cls_id, dets in enumerate(detections):
            if len(dets) == 0:
                continue
            for det in dets:
                if len(det) < 5:
                    continue
                score = float(det[4])
                if score < conf_thresh:
                    continue
                # Hailo NMS 출력: [y1, x1, y2, x2, score] (0~1 정규화)
                y1 = float(det[0]) * h_orig
                x1 = float(det[1]) * w_orig
                y2 = float(det[2]) * h_orig
                x2 = float(det[3]) * w_orig
                boxes.append(HailoBox(x1, y1, x2, y2, score, cls_id))

        self.boxes = boxes if boxes else None


class HailoDetector:
    def __init__(self, hef_path: str):
        from hailo_platform import (HEF, VDevice, HailoStreamInterface,
                                     InferVStreams, ConfigureParams,
                                     InputVStreamParams, OutputVStreamParams,
                                     FormatType)
        self._FormatType = FormatType
        self._InferVStreams = InferVStreams

        self.hef = HEF(hef_path)
        self.target = VDevice()
        configure_params = ConfigureParams.create_from_hef(
            hef=self.hef, interface=HailoStreamInterface.PCIe
        )
        self.network_groups = self.target.configure(self.hef, configure_params)
        self.network_group  = self.network_groups[0]
        self.network_group_params = self.network_group.create_params()

        self.input_vstreams_params = InputVStreamParams.make(
            self.network_group, format_type=FormatType.FLOAT32
        )
        self.output_vstreams_params = OutputVStreamParams.make(
            self.network_group, format_type=FormatType.FLOAT32
        )
        self.input_name = self.hef.get_input_vstream_infos()[0].name

        # helmet=0, person=1, vest=2
        self.names = {0: 'helmet', 1: 'person', 2: 'vest'}
        print(f"[HailoDetector] HEF 로드 완료: {hef_path}")

    def predict(self, frame, imgsz=640, conf=0.3, iou=0.5, verbose=False):
        # 전처리: BGR→RGB, 리사이즈
        # alls에서 /255 정규화를 처리하므로 여기선 0~255 그대로 입력
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (imgsz, imgsz))
        img = img.astype(np.float32)          # /255 하지 않음
        img = np.expand_dims(img, axis=0)     # (1, 640, 640, 3)

        input_data = {self.input_name: img}

        with self._InferVStreams(
            self.network_group,
            self.input_vstreams_params,
            self.output_vstreams_params
        ) as pipeline:
            with self.network_group.activate(self.network_group_params):
                results = pipeline.infer(input_data)

        key = list(results.keys())[0]
        detections = results[key][0]   # list of 3 arrays (per class)

        return [HailoResult(detections, self.names, frame.shape, imgsz, conf)]


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
# Picamera2 래퍼 (cap.read() 인터페이스와 동일하게 사용 가능)
# ============================================================
class PiCamera2Capture:
    """
    Picamera2를 cv2.VideoCapture처럼 사용할 수 있는 래퍼.
    OpenCV의 cap.read() → self.read() 로 대체.
    """
    def __init__(self, width: int, height: int):
        from picamera2 import Picamera2
        self._cam = Picamera2()
        config = self._cam.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"}
        )
        self._cam.configure(config)
        self._cam.start()
        time.sleep(0.5)  # 센서 안정화 대기
        print(f"[Picamera2] 시작 완료: {width}x{height} RGB888")

    def read(self):
        """cap.read() 와 동일한 인터페이스: (ret, frame_bgr) 반환"""
        try:
            frame = self._cam.capture_array()
            return True, frame
        except Exception as e:
            print(f"[Picamera2] 프레임 캡처 오류: {e}")
            return False, None

    def release(self):
        self._cam.stop()
        print("[Picamera2] 카메라 종료")


# ============================================================
# 카메라 초기화: Picamera2 우선, 실패 시 cv2.VideoCapture 폴백
# ============================================================
def open_camera(width: int, height: int):
    """
    1순위: Picamera2 (Pi Camera IMX219 등 CSI 카메라)
    2순위: cv2.VideoCapture (USB 웹캠 등)
    반환: cap 객체 (두 경우 모두 .read() / .release() 인터페이스 동일)
    """
    try:
        cap = PiCamera2Capture(width, height)
        print("카메라 연결 성공! (Picamera2)")
        return cap
    except Exception as e:
        print(f"[WARNING] Picamera2 실패: {e}")
        print("→ cv2.VideoCapture(0) 으로 폴백 시도...")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        print("[ERROR] cv2.VideoCapture도 실패했습니다. 카메라 연결을 확인하세요.")
        sys.exit(1)

    print("카메라 연결 성공! (cv2.VideoCapture)")
    return cap


# ============================================================
# 메인 루프
# ============================================================
def main():
    print("=== PPE Detection System 시작 ===")

    # 모델 로드
    print("모델 로딩 중...")
    det_model  = HailoDetector(DET_MODEL_PATH)   # HEF
    pose_model = YOLO(POSE_MODEL_PATH)            # Pose는 YOLO 유지
    ensemble   = PPEEnsemble(det_model, pose_model)

    # 카메라 열기
    print("카메라 여는 중...")
    cap = open_camera(CAP_WIDTH, CAP_HEIGHT)

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
    fail_count  = 0
    MAX_FAIL    = 10

    print("실행 중... (Ctrl+C로 종료)")

    if DISPLAY:
        cv2.namedWindow("PPE Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("PPE Detection", 960, 540)

    while running:
        ret, frame = cap.read()

        if not ret or frame is None:
            fail_count += 1
            print(f"[WARN] 프레임 읽기 실패 ({fail_count}/{MAX_FAIL})")
            if fail_count >= MAX_FAIL:
                print("[ERROR] 연속 프레임 실패 → 종료합니다.")
                break
            time.sleep(0.05)
            continue

        fail_count = 0
        frame_count += 1
        t0 = time.time()

        # 앙상블 추론
        results = ensemble.predict(frame)

        # FPS 계산
        dt = time.time() - t0
        fps_acc = fps_acc * 0.9 + (1.0 / dt if dt > 0 else 0) * 0.1
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
