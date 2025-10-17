# sort.py
# Simple Online and Realtime Tracking (SORT)
# https://github.com/abewley/sort

from __future__ import print_function
import numpy as np
from filterpy.kalman import KalmanFilter
# นำเข้า scipy.optimize.linear_sum_assignment ไว้ด้านบนเลย
from scipy.optimize import linear_sum_assignment # ต้องติดตั้ง scipy

# ฟังก์ชัน IOU (Intersection over Union)
def iou(bb_test, bb_gt):
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    # คำนวณ IOU
    o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1])
              + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh)
    return(o)

# คลาส KalmanBoxTracker (ไม่ได้แก้ไข)
class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox):
        # x, y, s, r, x', y', s'
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1,0,0,0,1,0,0],
                              [0,1,0,0,0,1,0],
                              [0,0,1,0,0,0,1],
                              [0,0,0,1,0,0,0],
                              [0,0,0,0,1,0,0],
                              [0,0,0,0,0,1,0],
                              [0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],
                              [0,1,0,0,0,0,0],
                              [0,0,1,0,0,0,0],
                              [0,0,0,1,0,0,0]])
        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01

        self.kf.x[:4] = self.convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self.convert_bbox_to_z(bbox))

    def predict(self):
        if (self.kf.x[6]+self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        return self.convert_x_to_bbox(self.kf.x)

    @staticmethod
    def convert_bbox_to_z(bbox):
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w/2.
        y = bbox[1] + h/2.
        s = w * h
        r = w / float(h)
        return np.array([x, y, s, r]).reshape((4, 1))

    @staticmethod
    def convert_x_to_bbox(x, score=None):
        w = np.sqrt(x[2] * x[3])
        # x[3] คือ ratio (r) = w/h, x[2] คือ area (s) = w*h
        # w = sqrt(s*r)
        # h = s / w  -> แต่ถ้าดูจาก x[2] = w*h, x[3] = w/h จะได้ w=sqrt(s*r), h=sqrt(s/r)
        # ดังนั้นต้องแก้สูตรให้สอดคล้องกับตัวแปรที่ใช้ใน SORT ดั้งเดิม (h = x[2] / w)
        h = x[2] / w # s/w = h
        
        x1 = x[0] - w/2.
        y1 = x[1] - h/2.
        x2 = x[0] + w/2.
        y2 = x[1] + h/2.
        if score is None:
            return np.array([x1, y1, x2, y2]).reshape((1, 4))
        else:
            return np.array([x1, y1, x2, y2, score]).reshape((1, 5))

# คลาส Sort (ไม่ได้แก้ไขในส่วนหลัก)
class Sort:
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 5))):
        self.frame_count += 1
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []

        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))

        for t in reversed(to_del):
            self.trackers.pop(t)

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)

        for t, trk in enumerate(self.trackers):
            if t not in unmatched_trks:
                # แก้ไข: ตรวจสอบก่อนว่า matched มีข้อมูลหรือไม่
                if matched.size > 0:
                    d = matched[np.where(matched[:, 1] == t)[0], 0]
                    # ตรวจสอบว่ามีข้อมูลการจับคู่อยู่จริงหรือไม่
                    if d.size > 0:
                         trk.update(dets[int(d[0]), :])
                    

        new_trackers = []
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            new_trackers.append(trk)

        self.trackers += new_trackers
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))

# ฟังก์ชัน Linear Assignment (ไม่ได้แก้ไข)
def linear_assignment(cost_matrix):
    # import scipy # ไม่ต้อง import ซ้ำ
    # from scipy.optimize import linear_sum_assignment # ไม่ต้อง import ซ้ำ
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))

# ฟังก์ชัน Associate Detections to Trackers (แก้ไขส่วนที่เกิดปัญหา)
def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0), dtype=int)

    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det, trk)
    matched_indices = linear_assignment(-iou_matrix)

    unmatched_detections = []
    # แก้ไข: เพิ่มการตรวจสอบว่า matched_indices มีอย่างน้อย 2 มิติ (มีคอลัมน์ [:, 0])
    # โดยทั่วไป matched_indices จะเป็น (N, 2) ถ้ามีการจับคู่
    if matched_indices.ndim > 1 and matched_indices.shape[1] > 0:
        matched_dets = matched_indices[:, 0]
        matched_trks = matched_indices[:, 1]
    else:
        # ถ้าไม่มีการจับคู่หรือเป็น array ว่าง (Numpy จะคืน array ว่างมิติเดียว)
        matched_dets = np.empty((0,), dtype=int)
        matched_trks = np.empty((0,), dtype=int)

    for d, det in enumerate(detections):
        # ใช้ matched_dets ที่เตรียมไว้แล้ว
        if d not in matched_dets:
            unmatched_detections.append(d)
            
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        # ใช้ matched_trks ที่เตรียมไว้แล้ว
        if t not in matched_trks:
            unmatched_trackers.append(t)

    matches = []
    for m in matched_indices:
        # ตรวจสอบ IoU
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
            
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)
        
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)