import argparse
import os
import pandas as pd
from cardioai.metrics import get_metrics_collection
from cardioai.fairness import calculate_fairness
import torch
import numpy as np

parser = argparse.ArgumentParser(description="Compile kfold training results")
parser.add_argument("experiment_dir", type=str, help="Experiment directory")

args = parser.parse_args()
experiment_dir = args.experiment_dir

predictions_all = []
metrics_all = []
metrics_all_per_group = []
fairness_metrics_all = []
test_metrics = get_metrics_collection()
folds = {}

for directory in os.listdir(experiment_dir):
    if directory.startswith("fold_") or directory.startswith("seed_"):
        fold_dir = os.path.join(experiment_dir, directory)
        fold_data = pd.read_csv(os.path.join(fold_dir, "test_predictions.csv"))
        predictions_all.append(fold_data)
        pred = torch.tensor(fold_data["prediction"].tolist())
        label = torch.tensor(fold_data["label"].tolist())
        fold_metrics = test_metrics(pred, label)
        fold_fairness_metrics_per_feature = calculate_fairness(predictions_filepath=os.path.join(fold_dir, "test_predictions.csv"))
        fold_per_group_metrics = {}
        fold_fairness_metrics = {}
        for protected_feature, metrics in fold_fairness_metrics_per_feature.items():
            for metric, values in metrics["per_group_metrics"].by_group.items():
                for index, value in values.items():
                    fold_per_group_metrics[protected_feature + ' ' + metric + ' ' + str(index)] = value
            fairness_metrics = metrics["fairness_metrics"]
            for name, metric in fairness_metrics.items():
                fold_fairness_metrics[protected_feature + ' ' + name] = metric
        # fold_metrics.update(fold_fairness_metrics)
        num_samples = len(fold_data)
        metrics_all.append({"metrics": fold_metrics, "num_samples": num_samples})
        metrics_all_per_group.append({"metrics": fold_per_group_metrics, "num_samples": num_samples})
        fairness_metrics_all.append({"metrics": fold_fairness_metrics, "num_samples": num_samples})
        folds[directory[-1]] = fold_data["f.eid"].tolist()

predictions_all = pd.concat(predictions_all).sort_values(by="f.eid")
predictions_all.to_csv(os.path.join(experiment_dir, "test_predictions_all.csv"))
pred = torch.tensor(predictions_all["prediction"].tolist())
label = torch.tensor(predictions_all["label"].tolist())


all_ids = []
print(f"Total: {len(all_ids)} samples")


writer = open(f"{experiment_dir}/test_metrics.txt", "w")

for index, metrics_all in enumerate([metrics_all, fairness_metrics_all, metrics_all_per_group]):
    if index == 1:
        writer.write(f"\n\n\nFairness metrics:\n\n")
    if index == 2:
        writer.write(f"\n\n\nPer group metrics:\n\n")
    writer.write(f"Multi run test results:\n\n")
    for metric in metrics_all[0]["metrics"]:
        #Check if metric value is numeric and mean calculation is possible
        if isinstance(metrics_all[0]["metrics"][metric], str):
            continue
        metric_name = metric
        metric_values = []
        for fold in metrics_all:
            if isinstance(fold["metrics"][metric_name], torch.Tensor):
                metric_values.append(fold["metrics"][metric_name].item())
            else:
                metric_values.append(fold["metrics"][metric_name])
        mean = sum(metric_values) / len(metric_values)
        mean = np.average(
            metric_values, weights=[fold["num_samples"] for fold in metrics_all]
        )
        mean = np.average(metric_values)
        std = np.std(metric_values)
        writer.write(f"{metric_name}: {mean:.3f}  ({std:.3f})\n")

    writer.write(
        f"\nPer run results: \n\nsamples: {[fold['num_samples'] for fold in metrics_all]}\n"
    )

    for metric in metrics_all[0]["metrics"]:
        metric_name = metric
        metric_values = []
        for fold in metrics_all:
            if isinstance(fold["metrics"][metric_name], torch.Tensor):
                metric_values.append(round(fold["metrics"][metric_name].item(), 3))
            else:
                if isinstance(fold["metrics"][metric_name], float):
                    metric_values.append(round(fold["metrics"][metric_name], 3))
                else:
                    metric_values.append(fold["metrics"][metric_name])

        writer.write(f"{metric_name}: {metric_values}\n")

    if index == 0:
        writer.write(f"\n\nResults for compiled predictions:\n\n")
        compiled_metrics = test_metrics(pred, label)
        for metric in compiled_metrics:
            writer.write(f"{metric}: {compiled_metrics[metric]:.3f}\n")

    if index == 1:
        writer.write(f"\n\nFairness metrics for compiled predictions:\n")
        compiled_fairness_metrics = calculate_fairness(predictions_filepath=os.path.join(experiment_dir, "test_predictions_all.csv"))
        for protected_feature, metrics in compiled_fairness_metrics.items():
            metric_frame = metrics["per_group_metrics"]
            fairness_metrics = metrics["fairness_metrics"]
            writer.write("\n\n\n" + protected_feature)
            writer.write("\n")
            writer.write(metric_frame.by_group.to_string(float_format="{:.3f}".format))
            writer.write("\n\n")
            for name, metric in fairness_metrics.items():
                if isinstance(metric, float):
                    writer.write(f"{name}: {metric:.3f}")
                else:
                    writer.write(f"{name}: {metric}")
                writer.write("\n")

writer.close()
