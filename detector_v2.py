"""
detector_v2.py
개선된 앙상블 + 알림/로그/부저

v1 대비 추가/변경:
- 클래스별 conf 분리 (helmet 0.45 / person 0.40 / vest 0.30)
- helmet 크기 필터 (person 면적의 25% 초과 제거 — Detection bbox 비교, Pose 무관)
- Pose IoU 매칭 0.5 → 0.35 완화
- 알림 쿨다운 10초
- 로컬 .jsonl 로그
- RPi GPIO 부저 (PC 환경에서는 콘솔 출력으로 대체)

유지된 것:
- helmet 착용/휴대 판단 로직 동일
- vest 판단 로직 동일
- safe = helmet=worn AND vest=worn
- 버퍼: 최근 5프레임 중 3프레임
- Pose 3프레임마다 실행
"""

import cv2
import json
import os
import time
import threading
import numpy as np
from dataclasses import dataclass
from typing import Optional
from collections import deque
from datetime import datetime


# ============================================================
# 설정
# ============================================================
class Config:
    # 클래스별 conf 분리
    DET_CONF = {
        'helmet': 0.45,
        'person': 0.40,
        'vest':   0.30,
    }
    DET_IOU = 0.50

    POSE_CONF      = 0.40
    POSE_IOU_MATCH = 0.35        # 0.5 → 0.35 완화
    POSE_EVERY_N   = 3
    KP_CONF_THRESH = 0.30

    HELMET_DY_RATIO       = 0.15
    HELMET_DX_RATIO       = 0.40
    HELMET_WORN_TOP       = 0.30
    HELMET_MAX_AREA_RATIO = 0.25  # person 면적 대비 helmet bbox 면적 상한

    VEST_DY_RATIO = 0.35
    VEST_WORN_MIN = 0.15
    VEST_WORN_MAX = 0.70

    UNKNOWN_KP_THRESH  = 0.20
    VIOLATION_BUFFER_N = 5
    VIOLATION_BUFFER_M = 3

    # 알림
    ALERT_COOLDOWN_SEC = 10.0
    LOG_DIR  = "./logs"
    LOG_FILE = "violations.jsonl"
    BUZZER_PIN = 18


# COCO keypoint 인덱스
HEAD_KP     = [0, 1, 2, 3, 4]
SHOULDER_KP = [5, 6]


# ============================================================
# 데이터 구조
# ============================================================
@dataclass
class PersonResult:
    bbox:            list
    helmet:          str
    vest:            str
    safe:            bool
    violation:       bool
    head_center:     Optional[tuple]
    shoulder_center: Optional[tuple]
    keypoints:       Optional[np.ndarray]
    kp_conf:         Optional[np.ndarray]
    pose_valid:      bool


# ============================================================
# 유틸
# ============================================================
def calc_iou(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0


def kp_center(kps, kc, indices):
    pts = [kps[i] for i in indices
           if kc[i] > Config.KP_CONF_THRESH and kps[i][0] > 0]
    if not pts:
        return None
    pts = np.array(pts)
    return (float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])))


def is_pose_valid(kps, kc):
    confs = [kc[i] for i in HEAD_KP + SHOULDER_KP]
    return float(np.mean(confs)) >= Config.UNKNOWN_KP_THRESH


def filter_helmet_by_size(helmet_bboxes, person_bbox):
    """
    [v2 추가] helmet bbox 크기 필터.
    person bbox 면적의 25% 초과하는 helmet 제거.
    Detection bbox 비교 — Pose/keypoint 무관.
    """
    pa = (person_bbox[2]-person_bbox[0]) * (person_bbox[3]-person_bbox[1])
    if pa <= 0:
        return helmet_bboxes
    filtered = []
    for b in helmet_bboxes:
        ha = (b[2]-b[0]) * (b[3]-b[1])
        if ha / pa <= Config.HELMET_MAX_AREA_RATIO:
            filtered.append(b)
    return filtered

def filter_helmet_by_keypoint(helmet_bboxes, head_center):
    if head_center is None:
        return helmet_bboxes   # Pose 없는 프레임 → 전부 통과
    hx, hy = head_center
    filtered = []
    for b in helmet_bboxes:
        x1, y1, x2, y2 = b
        w, h = x2-x1, y2-y1
        if (x1 - w*0.1 <= hx <= x2 + w*0.1 and
                y1 - h*0.2 <= hy <= y2 + h*0.2):
            filtered.append(b)
    return filtered

def get_items_inside_person(person_bbox, item_bboxes):
    px1, py1, px2, py2 = person_bbox
    pw, ph = px2-px1, py2-py1
    inside = []
    for b in item_bboxes:
        cx = (b[0]+b[2])/2; cy = (b[1]+b[3])/2
        if (px1-pw*0.1 <= cx <= px2+pw*0.1 and
                py1-ph*0.1 <= cy <= py2+ph*0.1):
            inside.append(b)
    return inside


# ============================================================
# 착용 판단 (v1과 동일 로직)
# ============================================================
def judge_helmet(person_bbox, helmet_bboxes, head_center, pose_valid):
    px1, py1, px2, py2 = person_bbox
    ph, pw = py2-py1, px2-px1

    # [v2] 크기 필터 먼저 적용
    filtered = filter_helmet_by_size(helmet_bboxes, person_bbox)
    inside   = get_items_inside_person(person_bbox, filtered)

    if not inside:
        return 'not_detected'

    if head_center is not None:
        hx, hy = head_center
        best, best_dist = None, float('inf')
        for b in inside:
            bcx = (b[0]+b[2])/2; bcy = (b[1]+b[3])/2
            dist = abs(hy-bcy) + abs(hx-bcx)
            if dist < best_dist:
                best_dist = dist; best = b
        bcx = (best[0]+best[2])/2; bcy = (best[1]+best[3])/2
        if hy-bcy > -ph*Config.HELMET_DY_RATIO and abs(hx-bcx) < pw*Config.HELMET_DX_RATIO:
            return 'worn'
    else:
        for b in inside:
            if (b[1]+b[3])/2 < py1 + ph*Config.HELMET_WORN_TOP:
                return 'worn'

    return 'carried'


def judge_vest(person_bbox, vest_bboxes, shoulder_center, pose_valid):
    px1, py1, px2, py2 = person_bbox
    ph = py2-py1

    inside = get_items_inside_person(person_bbox, vest_bboxes)

    if not inside:
        return 'not_detected'

    if shoulder_center is not None:
        sy = shoulder_center[1]
        for b in inside:
            if abs((b[1]+b[3])/2 - sy) < ph*Config.VEST_DY_RATIO:
                return 'worn'
    else:
        for b in inside:
            rel = ((b[1]+b[3])/2 - py1) / ph if ph > 0 else 0
            if Config.VEST_WORN_MIN <= rel <= Config.VEST_WORN_MAX:
                return 'worn'

    return 'not_detected'


# ============================================================
# 버퍼 (v1과 동일 — index 기반)
# ============================================================
class ViolationBuffer:
    def __init__(self):
        self._bufs = {}

    def update(self, idx, is_vio):
        if idx not in self._bufs:
            self._bufs[idx] = deque(maxlen=Config.VIOLATION_BUFFER_N)
        self._bufs[idx].append(is_vio)
        return sum(self._bufs[idx]) >= Config.VIOLATION_BUFFER_M

    def cleanup(self, active):
        for k in [k for k in self._bufs if k not in active]:
            del self._bufs[k]


# ============================================================
# [v2 추가] 부저
# ============================================================
class Buzzer:
    def __init__(self, pin):
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
            print("[Buzzer] RPi.GPIO 없음 → 콘솔 출력으로 대체")

    def beep(self, duration=0.5):
        t = threading.Thread(target=self._beep, args=(duration,), daemon=True)
        t.start()

    def _beep(self, duration):
        if self._available:
            self._gpio.output(self.pin, self._gpio.HIGH)
            time.sleep(duration)
            self._gpio.output(self.pin, self._gpio.LOW)
        else:
            print(f"[BEEP] {duration}s")

    def cleanup(self):
        if self._available:
            self._gpio.cleanup()


# ============================================================
# [v2 추가] 알림 + 로컬 로그
# ============================================================
class ViolationAlert:
    def __init__(self):
        os.makedirs(Config.LOG_DIR, exist_ok=True)
        self._log_path  = os.path.join(Config.LOG_DIR, Config.LOG_FILE)
        self._cooldowns = {}
        self._total     = 0
        self._buzzer    = Buzzer(Config.BUZZER_PIN)

    def process(self, person_idx, helmet, vest):
        """
        violation 확정 시 호출.
        쿨다운 중이면 무시 (중복 알림 방지).
        """
        now = time.time()
        if now - self._cooldowns.get(person_idx, 0) < Config.ALERT_COOLDOWN_SEC:
            return

        self._cooldowns[person_idx] = now
        self._total += 1

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[VIOLATION] {ts} | Person#{person_idx} | H:{helmet} V:{vest}")

        # 로컬 .jsonl 기록
        with open(self._log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({
                "timestamp":  ts,
                "person_idx": person_idx,
                "helmet":     helmet,
                "vest":       vest,
            }, ensure_ascii=False) + '\n')

        # 부저
        self._buzzer.beep(0.5)

    def stats(self):
        return {"total": self._total, "log": self._log_path}

    def cleanup(self):
        self._buzzer.cleanup()


# ============================================================
# 메인 앙상블
# ============================================================
class PPEEnsemble:
    def __init__(self, det_model, pose_model):
        self.det_model   = det_model
        self.pose_model  = pose_model
        self.frame_count = 0
        self.vio_buffer  = ViolationBuffer()
        self.alert       = ViolationAlert()   # [v2] 알림 내장

        self.CLASS_MAP = {}
        for idx, name in det_model.names.items():
            nl = name.lower()
            if 'person' in nl:                       self.CLASS_MAP[idx] = 'person'
            elif 'helmet' in nl or 'hardhat' in nl:  self.CLASS_MAP[idx] = 'helmet'
            elif 'vest' in nl:                       self.CLASS_MAP[idx] = 'vest'
        print(f"[v2] CLASS_MAP: {self.CLASS_MAP}")

    def predict(self, frame):
        self.frame_count += 1
        run_pose = (self.frame_count % Config.POSE_EVERY_N == 0)

        # [v2] Detection — 가장 낮은 conf로 전부 받고, 이후 클래스별 필터
        det = self.det_model.predict(
            frame, imgsz=640,
            conf=min(Config.DET_CONF.values()),
            iou=Config.DET_IOU, verbose=False
        )[0]

        persons, helmets, vests = [], [], []
        if det.boxes is not None:
            for box in det.boxes:
                cid   = int(box.cls[0])
                if cid not in self.CLASS_MAP: continue
                label = self.CLASS_MAP[cid]
                conf  = float(box.conf[0])
                # [v2] 클래스별 conf 임계값 적용
                if conf < Config.DET_CONF[label]: continue
                b = box.xyxy[0].cpu().numpy().tolist()
                if label == 'person':   persons.append(b)
                elif label == 'helmet': helmets.append(b)
                elif label == 'vest':   vests.append(b)

        # Pose
        pose_list = []
        if run_pose:
            pr = self.pose_model.predict(
                frame, imgsz=640, conf=Config.POSE_CONF, verbose=False
            )[0]
            if pr.keypoints is not None and pr.boxes is not None:
                for i in range(len(pr.boxes)):
                    pb  = pr.boxes[i].xyxy[0].cpu().numpy().tolist()
                    kp  = pr.keypoints[i].data[0].cpu().numpy()
                    pose_list.append((pb, kp[:, :2], kp[:, 2]))

        # 판단
        results = []
        active  = []
        for p_idx, p_bbox in enumerate(persons):
            active.append(p_idx)
            hc, sc, kps, kc_arr, pv = None, None, None, None, False

            if pose_list:
                best_iou, best_i = 0, -1
                for ki, (pb, _, _) in enumerate(pose_list):
                    v = calc_iou(p_bbox, pb)
                    if v > best_iou: best_iou, best_i = v, ki
                # [v2] IoU 임계값 0.35 (완화)
                if best_iou >= Config.POSE_IOU_MATCH and best_i >= 0:
                    _, kps, kc_arr = pose_list[best_i]
                    hc = kp_center(kps, kc_arr, HEAD_KP)
                    sc = kp_center(kps, kc_arr, SHOULDER_KP)
                    pv = is_pose_valid(kps, kc_arr)

            h_st = judge_helmet(p_bbox, helmets, hc, pv)
            v_st = judge_vest(p_bbox, vests, sc, pv)
            is_safe = (h_st == 'worn' and v_st == 'worn')
            confirmed_vio = self.vio_buffer.update(p_idx, not is_safe)

            # [v2] violation 확정 시 알림 처리
            if confirmed_vio:
                self.alert.process(p_idx, h_st, v_st)

            results.append(PersonResult(
                bbox=p_bbox, helmet=h_st, vest=v_st,
                safe=is_safe, violation=confirmed_vio,
                head_center=hc, shoulder_center=sc,
                keypoints=kps, kp_conf=kc_arr, pose_valid=pv,
            ))

        self.vio_buffer.cleanup(active)
        return results

    def cleanup(self):
        self.alert.cleanup()
