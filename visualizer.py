"""
visualizer.py
앙상블 결과 시각화 (Colab/RPi 공통)
"""

import cv2
import numpy as np
from detector_v2 import PersonResult, HEAD_KP, SHOULDER_KP, Config

# ============================================================
# 색상 정의 (BGR)
# ============================================================
C = {
    'safe':         (0,   200,   0),
    'warn':         (0,   180, 255),
    'danger':       (0,     0, 220),
    'unknown':      (160, 160, 160),
    'kp_head':      (255,   0, 255),
    'kp_shoulder':  (255, 255,   0),
    'head_cross':   (0,   255, 255),
}

STATUS_COLOR = {
    'worn':         C['safe'],
    'carried':      C['warn'],
    'not_detected': C['danger'],
    'unknown':      C['unknown'],
}


def draw_results(frame: np.ndarray,
                 results: list,
                 show_kp: bool = True,
                 show_legend: bool = True) -> tuple:
    """
    Returns: (annotated_frame, violation_count)
    """
    out = frame.copy()
    vio_count = 0

    for r in results:
        x1, y1, x2, y2 = [int(v) for v in r.bbox]

        # person bbox 색상
        if r.violation:
            box_color = C['danger']
            vio_count += 1
        elif r.helmet == 'unknown' or r.vest == 'unknown':
            box_color = C['unknown']
        elif r.safe:
            box_color = C['safe']
        else:
            box_color = C['warn']

        cv2.rectangle(out, (x1, y1), (x2, y2), box_color, 2)

        # 상태 라벨
        cv2.rectangle(out, (x1, y1-52), (x1+230, y1), (30, 30, 30), -1)

        h_color = STATUS_COLOR.get(r.helmet, C['unknown'])
        v_color = STATUS_COLOR.get(r.vest,   C['unknown'])
        p_tag   = 'P' if r.pose_valid else '-'
        vio_tag = ' [VIO]' if r.violation else (' [SAFE]' if r.safe else '')

        cv2.putText(out, f"H:{r.helmet}",
                    (x1+4, y1-32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, h_color, 1)
        cv2.putText(out, f"V:{r.vest}",
                    (x1+4, y1-12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, v_color, 1)
        cv2.putText(out, f"[{p_tag}]{vio_tag}",
                    (x1+155, y1-22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # Keypoint
        if show_kp and r.keypoints is not None and r.kp_conf is not None:
            for i in HEAD_KP:
                if r.kp_conf[i] > Config.KP_CONF_THRESH:
                    cv2.circle(out,
                               (int(r.keypoints[i][0]), int(r.keypoints[i][1])),
                               4, C['kp_head'], -1)
            for i in SHOULDER_KP:
                if r.kp_conf[i] > Config.KP_CONF_THRESH:
                    cv2.circle(out,
                               (int(r.keypoints[i][0]), int(r.keypoints[i][1])),
                               4, C['kp_shoulder'], -1)
            if r.head_center:
                cv2.drawMarker(out,
                               (int(r.head_center[0]), int(r.head_center[1])),
                               C['head_cross'], cv2.MARKER_CROSS, 12, 2)

    # 범례
    if show_legend:
        h = out.shape[0]
        cv2.rectangle(out, (0, h-80), (260, h), (30, 30, 30), -1)
        items = [
            ('worn',         C['safe'],    'worn'),
            ('carried',      C['warn'],    'carried'),
            ('not_detected', C['danger'],  'not detected'),
            ('unknown',      C['unknown'], 'unknown (back/bent)'),
        ]
        for i, (_, color, label) in enumerate(items):
            cv2.putText(out, label, (10, h-60+i*16),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

    return out, vio_count
