import json
import math
from copy import deepcopy

def load_metrics_as_json(filepath):
    f= open(filepath, "r")
    metrics=[]
    for lid,line in enumerate(f):
        if lid == 0:
            continue
        metrics.append(json.loads(line))
    f.close()
    return metrics

def bucketize_metrics(bucket_size, metrics_lst):
    num_buckets = len(metrics_lst) // bucket_size
    remainder = len(metrics_lst) % bucket_size
    bucketized_metrics=[]
    metrics_types = [k for k in metrics_lst[0].keys() if "loss" in k]
    for i in range(0, num_buckets, bucket_size):
        bucket = dict(bucket_range=[metrics_lst[i]['iter'], metrics_lst[i+bucket_size-1]['iter']])
        for metric in metrics_types:
            sum_metric=0
            for j in range(i, i+bucket_size):
                sum_metric += metrics_lst[j][metric]
            bucket[metric] = sum_metric / bucket_size
        bucketized_metrics.append(bucket)
    last_bucket = dict(bucket_range=[metrics_lst[-remainder]['iter'], metrics_lst[-1]['iter']])
    for metric in metrics_types:
        sum_metric=0
        for k in range(-remainder, 0):
            sum_metric += metrics_lst[k][metric]
        last_bucket[metric] = sum_metric / bucket_size
    bucketized_metrics.append(last_bucket)
    return bucketized_metrics

def compare_metrics(metrics_lst1, metrics_lst2, metrics_lst1_name, metrics_lst2_name, metrics_types):
    assert isinstance(metrics_types, (list, tuple))
    assert all([m in metrics_lst1[0] and m in metrics_lst2[0] for m in metrics_types])
    last_bucket = min(len(metrics_lst1), len(metrics_lst2))
    metric_res = dict(better_total_loss_iters=[], total_loss=0, diff_total_loss_iters=[], min_total_loss=math.inf,
                      total_focus_metrics_loss_iters=[])
    for m in metrics_types:
        metric_res[f"better_{m}_iters"] = []
        metric_res[f"total_{m}_loss"] = 0
        metric_res[f"diff_{m}_iters"] = []
        metric_res[f"min_{m}_loss"] = math.inf

    res={
        "compared_buckets": last_bucket,
        metrics_lst1_name: metric_res,
        metrics_lst2_name: deepcopy(metric_res)
    }



    for i in range(last_bucket):
        m1 = metrics_lst1[i]
        m2 = metrics_lst2[i]

        m1_total_loss = m1['loss']
        m2_total_loss = m2['loss']
        # better_total_loss_iters
        if m1_total_loss < m2_total_loss: # m1 is better
            res[metrics_lst1_name]["better_total_loss_iters"].append(i)
        elif m2_total_loss < m1_total_loss:
            res[metrics_lst2_name]["better_total_loss_iters"].append(i)
        # total loss
        res[metrics_lst1_name]['total_loss'] += m1_total_loss
        res[metrics_lst2_name]['total_loss'] += m2_total_loss
        # diff_total_loss_iters
        res[metrics_lst1_name]['diff_total_loss_iters'].append(m1_total_loss - m2_total_loss)
        res[metrics_lst2_name]['diff_total_loss_iters'].append(m2_total_loss - m1_total_loss)

        # min loss
        if m1_total_loss < res[metrics_lst1_name]['min_total_loss']:
            res[metrics_lst1_name]['min_total_loss'] = m1_total_loss
        if m2_total_loss < res[metrics_lst2_name]['min_total_loss']:
            res[metrics_lst2_name]['min_total_loss'] = m2_total_loss

        m1_total_focus_metrics=0
        m2_total_focus_metrics=0

        for m in metrics_types:
            m1_metric=m1[m]
            m2_metric=m2[m]
            m1_total_focus_metrics += m1_metric
            m2_total_focus_metrics += m2_metric
            # better_{m}_iters
            if m1_metric < m2_metric: # m1 is better
                res[metrics_lst1_name][f"better_{m}_iters"].append(i)
            elif m2_metric < m1_metric:
                res[metrics_lst2_name][f"better_{m}_iters"].append(i)
            # total_{m}_loss
            res[metrics_lst1_name][f"total_{m}_loss"] += m1_metric
            res[metrics_lst2_name][f"total_{m}_loss"] += m2_metric
            # diff_{m}_iters
            res[metrics_lst1_name][f"diff_{m}_iters"].append(m1_metric - m2_metric)
            res[metrics_lst2_name][f"diff_{m}_iters"].append(m2_metric - m1_metric)
            # min loss
            if m1_metric < res[metrics_lst1_name][f'min_{m}_loss']:
                res[metrics_lst1_name][f'min_{m}_loss'] = m1_metric
            if m2_metric < res[metrics_lst2_name][f'min_{m}_loss']:
                res[metrics_lst2_name][f'min_{m}_loss'] = m2_metric

        res[metrics_lst1_name]["total_focus_metrics_loss_iters"].append(m1_total_focus_metrics)
        res[metrics_lst2_name]["total_focus_metrics_loss_iters"].append(m2_total_focus_metrics)

    return res

def get_summary(metrics_lst1, metrics_lst2, metrics_lst1_name, metrics_lst2_name, metrics_types, last_n=100):
    res = compare_metrics(metrics_lst1, metrics_lst2, metrics_lst1_name, metrics_lst2_name, metrics_types)
    total_iters = res['compared_buckets']
    last_n_range = (total_iters-last_n,total_iters)
    m1_total_loss_better = len(res[metrics_lst1_name]["better_total_loss_iters"])
    m2_total_loss_better = len(res[metrics_lst2_name]["better_total_loss_iters"])

    msg = "========== total loss ==========\n"
    msg += f"{metrics_lst1_name}\t is better in {m1_total_loss_better}/{total_iters} buckets\n"
    msg += f"{metrics_lst2_name}\t is better in {m2_total_loss_better}/{total_iters} buckets\n"
    msg += f"{metrics_lst1_name}\t had a minimum loss of {res[metrics_lst1_name]['min_total_loss']}\n"
    msg += f"{metrics_lst2_name}\t had a minimum loss of {res[metrics_lst2_name]['min_total_loss']}\n"

    msg += f"========== total {' '.join(metrics_types)} loss ==========\n"
    msg += f"{metrics_lst1_name}\t had a total focus loss of {sum(res[metrics_lst1_name]['total_focus_metrics_loss_iters'])}\n"
    msg += f"{metrics_lst2_name}\t had a total focus loss of {sum(res[metrics_lst2_name]['total_focus_metrics_loss_iters'])}\n"
    msg += f"{metrics_lst1_name}\t had an average focus loss of {sum(res[metrics_lst1_name]['total_focus_metrics_loss_iters'])/total_iters}\n"
    msg += f"{metrics_lst2_name}\t had an average focus loss of {sum(res[metrics_lst2_name]['total_focus_metrics_loss_iters'])/total_iters}\n"
    msg += f"{metrics_lst1_name}\t had an average focus loss (in the last {last_n*50} iters) of {sum(res[metrics_lst1_name]['total_focus_metrics_loss_iters'][last_n_range[0]:last_n_range[1]])/last_n}\n"
    msg += f"{metrics_lst2_name}\t had an average focus loss (in the last {last_n*50} iters) of {sum(res[metrics_lst2_name]['total_focus_metrics_loss_iters'][last_n_range[0]:last_n_range[1]])/last_n}\n"
    

    for m in metrics_types:
        m1_metric_loss_better = len(res[metrics_lst1_name][f"better_{m}_iters"])
        m2_metric_loss_better = len(res[metrics_lst2_name][f"better_{m}_iters"])
        msg += f"========== {m} loss ==========\n"
        msg += f"{metrics_lst1_name}\t is better in {m1_metric_loss_better}/{total_iters} buckets\n"
        msg += f"{metrics_lst2_name}\t is better in {m2_metric_loss_better}/{total_iters} buckets\n"
        msg += f"{metrics_lst1_name}\t had a minimum {m} loss of {res[metrics_lst1_name][f'min_{m}_loss']}\n"
        msg += f"{metrics_lst2_name}\t had a minimum {m} loss of {res[metrics_lst2_name][f'min_{m}_loss']}\n"
        msg += f"{metrics_lst1_name}\t had an average {m} loss of {res[metrics_lst1_name][f'total_{m}_loss']/total_iters}\n"
        msg += f"{metrics_lst2_name}\t had an average {m} loss of {res[metrics_lst2_name][f'total_{m}_loss']/total_iters}\n"
        msg += f"{metrics_lst1_name}\t had an average {m} loss (in the last {last_n*50} iters) of {sum([e[m] for e in metrics_lst1[last_n_range[0]:last_n_range[1]]])/last_n}\n"
        msg += f"{metrics_lst2_name}\t had an average {m} loss (in the last {last_n*50} iters) of {sum([e[m] for e in metrics_lst2[last_n_range[0]:last_n_range[1]]])/last_n}\n"

    print(msg)
