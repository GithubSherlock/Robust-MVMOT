import numpy as np
from scipy.optimize import linear_sum_assignment
import math

from torch.ao.nn.quantized.functional import threshold

def voc_ap(rec, prec):
    """
    --- Official matlab code VOC2012---
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
            mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    rec.insert(0, 0.0) # insert 0.0 at begining of list
    rec.append(1.0) # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0) # insert 0.0 at begining of list
    prec.append(0.0) # insert 0.0 at end of list
    mpre = prec[:]
    """
     This part makes the precision monotonically decreasing
        (goes from the end to the beginning)
        matlab: for i=numel(mpre)-1:-1:1
                    mpre(i)=max(mpre(i),mpre(i+1));
    """
    # matlab indexes start in 1 but python in 0, so I have to do:
    #     range(start=(len(mpre) - 2), end=0, step=-1)
    # also the python function range excludes the end, resulting in:
    #     range(start=(len(mpre) - 2), end=-1, step=-1)
    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
    """
     This part creates a list of indexes where the recall changes
        matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
    """
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i-1]:
            i_list.append(i) # if it was matlab would be i + 1
    """
     The Average Precision (AP) is the area under the curve
        (numerical integration)
        matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i]-mrec[i-1])*mpre[i])
    return ap, mrec, mpre

def getDistance(x1, y1, x2, y2):
    return math.sqrt(pow((x1 - x2), 2) + pow((y1 - y2), 2))


def calculate_precision_recall(gt, det, threshold):
    matched_detections = []
    num_gts = len(gt)
    num_dets = len(det)

    if num_dets == 0:
        return 0, 0  # precision, recall

    gt_detected = np.zeros(num_gts, dtype=bool)
    det_used = np.zeros(num_dets, dtype=bool)

    for i in range(num_gts):
        for j in range(num_dets):
            if getDistance(gt[i][2], gt[i][3], det[j][2], det[j][3]) < threshold:
                if not gt_detected[i] and not det_used[j]:
                    gt_detected[i] = True
                    det_used[j] = True
                    matched_detections.append(det[j])
                    break

    tp = sum(gt_detected)
    fp = num_dets - tp
    fn = num_gts - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    return precision, recall, tp, fp, fn


def calculate_map(gt, det, iou_thresholds):
    average_precisions = []

    for threshold in iou_thresholds:
        precisions = []
        recalls = []
        tp, fp, fn = [],[],[]
        for t in range(1, int(max(gt[:, 0])) + 2):
            gt_frame = gt[np.where(gt[:, 0] == t - 1)]
            det_frame = det[np.where(det[:, 0] == t - 1)]

            precision, recall,  tp_, fp_, fn_ = calculate_precision_recall(gt_frame, det_frame, threshold)
            precisions.append(precision)
            recalls.append(recall)

            tp.append(tp_)
            fp.append(fp_)
            fn.append(fn_)

        # Sort by recall
        sorted_indices = np.argsort(recalls)
        precisions = np.array(precisions)[sorted_indices]
        recalls = np.array(recalls)[sorted_indices]
        # print(precisions)
        # print(recalls)
        # import matplotlib.pyplot as plt
        # plt.scatter(recalls,precisions)
        # plt.xlim([0, 1.2])
        # plt.ylim([0, 1.2])
        # plt.show()
        # Compute AP
        # average_precision = np.sum((recalls[1:] - recalls[:-1]) * precisions[1:])
        average_precisions, mrec, mpre = voc_ap(list(recalls),list(precisions))

    return np.mean(average_precisions)


def CLEAR_MOD_HUN_mAP(gt, det, iou_thresholds, threshold=50 / 2.5):
    """
    @param gt: the ground truth result matrix
    @param det: the detection result matrix
    @param distance_threshold: the distance threshold
    @return: MODA, MODP, recall, precision, MAP
    """

    F = int(max(gt[:, 0])) + 1
    N = int(max(det[:, 1])) + 1
    Fgt = int(max(gt[:, 0])) + 1
    Ngt = int(max(gt[:, 1])) + 1

    M = np.zeros((F, Ngt))

    c = np.zeros((1, F))
    fp = np.zeros((1, F))
    m = np.zeros((1, F))
    g = np.zeros((1, F))

    d = np.zeros((F, Ngt))
    distances = np.inf * np.ones((F, Ngt))

    for t in range(1, F + 1):
        GTsInFrames = np.where(gt[:, 0] == t - 1)
        DetsInFrames = np.where(det[:, 0] == t - 1)
        GTsInFrame = GTsInFrames[0]
        DetsInFrame = DetsInFrames[0]
        GTsInFrame = np.reshape(GTsInFrame, (1, GTsInFrame.shape[0]))
        DetsInFrame = np.reshape(DetsInFrame, (1, DetsInFrame.shape[0]))

        Ngtt = GTsInFrame.shape[1]
        Nt = DetsInFrame.shape[1]
        g[0, t - 1] = Ngtt

        if GTsInFrame is not None and DetsInFrame is not None:
            dist = np.inf * np.ones((Ngtt, Nt))
            for o in range(1, Ngtt + 1):
                GT = gt[GTsInFrame[0][o - 1]][2:4]
                for e in range(1, Nt + 1):
                    E = det[DetsInFrame[0][e - 1]][2:4]
                    dist[o - 1, e - 1] = getDistance(GT[0], GT[1], E[0], E[1])
            tmpai = dist
            tmpai = np.array(tmpai)

            tmpai[tmpai > threshold] = 1e6
            if not tmpai.all() == 1e6:
                HUN_res = np.array(linear_sum_assignment(tmpai)).T
                HUN_res = HUN_res[tmpai[HUN_res[:, 0], HUN_res[:, 1]] < threshold]
                u, v = HUN_res[HUN_res[:, 1].argsort()].T
                for mmm in range(1, len(u) + 1):
                    M[t - 1, u[mmm - 1]] = v[mmm - 1] + 1
        curdetected, = np.where(M[t - 1, :])

        c[0][t - 1] = curdetected.shape[0]
        for ct in curdetected:
            eid = M[t - 1, ct] - 1
            gtX = gt[GTsInFrame[0][ct], 2]

            gtY = gt[GTsInFrame[0][ct], 3]

            stX = det[DetsInFrame[0][int(eid)], 2]
            stY = det[DetsInFrame[0][int(eid)], 3]

            distances[t - 1, ct] = getDistance(gtX, gtY, stX, stY)
        fp[0][t - 1] = Nt - c[0][t - 1]
        m[0][t - 1] = g[0][t - 1] - c[0][t - 1]

    MODP = sum(1 - distances[distances < threshold] / threshold) / np.sum(c) * 100 if sum(
        1 - distances[distances < threshold] / threshold) / np.sum(c) * 100 > 0 else 0
    MODA = (1 - ((np.sum(m) + np.sum(fp)) / np.sum(g))) * 100 if (1 - (
            (np.sum(m) + np.sum(fp)) / np.sum(g))) * 100 > 0 else 0
    recall = np.sum(c) / np.sum(g) * 100 if np.sum(c) / np.sum(g) * 100 > 0 else 0
    precision = np.sum(c) / (np.sum(fp) + np.sum(c)) * 100 if np.sum(c) / (np.sum(fp) + np.sum(c)) * 100 > 0 else 0

    # Calculate MAP
    map_value = calculate_map(gt, det,[threshold]) #, iou_thresholds)

    return recall, precision, MODA, MODP, map_value


if __name__ == "__main__":
    # Example usage
    gt = np.array([[0, 1, 5, 5], [1, 2, 15, 15]])
    det = np.array([[0, 0.9, 5, 5], [1, 0.8, 15, 15]])

    iou_thresholds = [0.5]
    recall, precision, MODA, MODP, mAP = CLEAR_MOD_HUN_mAP(gt, det, iou_thresholds)

    print(f"Recall: {recall}")
    print(f"Precision: {precision}")
    print(f"MODA: {MODA}")
    print(f"MODP: {MODP}")
    print(f"MAP: {mAP}")
