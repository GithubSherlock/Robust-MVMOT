import motmetrics as mm
import numpy as np


def mot_metrics_pedestrian(tSource, gtSource):
    gt = np.loadtxt(gtSource, delimiter=',')
    t = np.loadtxt(tSource, delimiter=',')
    acc = mm.MOTAccumulator()
    for frame in np.unique(gt[:, 0]).astype(int):
        gt_dets = gt[gt[:, 0] == frame][:, (1, 7, 8)]
        t_dets = t[t[:, 0] == frame][:, (1, 7, 8)]

        A = gt_dets[:, 1:3] * 0.025
        B = t_dets[:, 1:3] * 0.025

        C = mm.distances.norm2squared_matrix(A, B, max_d2=1)  # format: gt, t
        C = np.sqrt(C)

        acc.update(gt_dets[:, 0].astype('int').tolist(),
                   t_dets[:, 0].astype('int').tolist(),
                   C,
                   frameid=frame)

    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics)
    return summary


def mot_metrics(tSource, gtSource, scale=0.025):
    gt = np.loadtxt(gtSource, delimiter=',')
    dt = np.loadtxt(tSource, delimiter=',')

    accs = []
    for seq in np.unique(gt[:, 0]).astype(int):
        acc = mm.MOTAccumulator()
        for frame in np.unique(gt[:, 1]).astype(int):
            gt_dets = gt[np.logical_and(gt[:, 0] == seq, gt[:, 1] == frame)][:, (2, 8, 9)]
            dt_dets = dt[np.logical_and(dt[:, 0] == seq, dt[:, 1] == frame)][:, (2, 8, 9)]

            # format: gt, t
            C = mm.distances.norm2squared_matrix(gt_dets[:, 1:3] * scale, dt_dets[:, 1:3] * scale, max_d2=1)
            C = np.sqrt(C)

            acc.update(gt_dets[:, 0].astype('int').tolist(),
                       dt_dets[:, 0].astype('int').tolist(),
                       C,
                       frameid=frame)
        accs.append(acc)

    mh = mm.metrics.create()
    summary = mh.compute_many(accs, metrics=mm.metrics.motchallenge_metrics, generate_overall=True)
    print("\n")
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)
    return summary


if __name__ == "__main__":
    import os

    general_path = '/media/rasho/Data 1/Arbeit/saved_models/EarlyBird_tests/new_old_tests/Assessing_wild_2_splitSegnet_camDropout_res18_Z4_15'
    # best results
    general_path = '/media/rasho/Data 1/Arbeit/saved_models/EarlyBird_tests/new_old_tests/baseline_test/Assessing_wild_1345_mvdet_1_res18'
    tSource = os.path.join(general_path, 'mota_pred.txt')
    gtSource = os.path.join(general_path, 'mota_gt.txt')
    suma = mot_metrics_pedestrian(tSource, gtSource)
    # print(suma)
    for key, value in suma.iloc[0].to_dict().items():
        print(f'track/{key}', value)
