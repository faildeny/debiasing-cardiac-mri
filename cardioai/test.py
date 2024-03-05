import yaml
import os
import pandas as pd
from monai.data import CacheDataset, DataLoader
import torch
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm

from pytorch_grad_cam import (
    GradCAM,
    HiResCAM,
    ScoreCAM,
    GradCAMPlusPlus,
    AblationCAM,
    XGradCAM,
    EigenCAM,
    FullGrad,
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.utils import save_image
from PIL import Image


import cardioai.dataset as dataset
from cardioai.models import get_model
from cardioai.metrics import get_metrics_collection
from cardioai.transforms import (
    get_dataset_torchio,
    get_transforms_torchio,
    prepare_batch,
)
from cardioai.fairness import calculate_fairness

CONFIG_FILE = "/config.yaml"
GPU_ID = 0


def test(experiment_dir, compute_activations=False):
    print(f"Testing experiment: {experiment_dir}")
    device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")

    with open(experiment_dir + CONFIG_FILE) as file:
        config = yaml.safe_load(file)

    BATCH_SIZE = config["params"]["batch_size"]
    INPUT_SIZE = config["params"]["input_size"]
    MASK_IMAGES = config["params"]["mask_images"]
    VIDEO = config["params"]["video"]
    TIME_STEP = config["params"]["time_step"]
    GRAY2RGB = config["params"]["gray2rgb"]
    TORCHIO_BACKEND = config["use_torchio"]

    ICD_CODE = config["dataset"]["icd_code"]
    testd_list = dataset.get_data_lists(config, only_test=True)
    # testd_list = dataset.get_kfold_split(config, only_test=True)

    print(f"Testing on {len(testd_list)} samples")

    test_balanced_weights = [sample["sample_weight"] for sample in testd_list]

    test_weighted_sampler = WeightedRandomSampler(
        weights=test_balanced_weights, num_samples=len(testd_list), replacement=True
    )
    num_workers = config["num_workers"]
    validation_transforms = get_transforms_torchio(config, test=True)
    vald_ds = get_dataset_torchio(testd_list, validation_transforms)
    test_loader = torch.utils.data.DataLoader(
        vald_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers
    )
    balanced_test_loader = torch.utils.data.DataLoader(
        vald_ds,
        sampler=test_weighted_sampler,
        batch_size=BATCH_SIZE,
        num_workers=num_workers,
    )

    model = get_model(config)
    # model.load_state_dict(torch.load(f"{experiment_dir}/best_model.pth"))
    for filename in os.listdir(experiment_dir):
        if filename.endswith(".pth"):
            model_path = os.path.join(experiment_dir, filename)
            break
    model.load_state_dict(torch.load(model_path))
    if compute_activations:
        activations_path = f"{experiment_dir}/activations"
        if not os.path.exists(activations_path):
            os.mkdir(activations_path)
        device = torch.device("cpu")
        for param in model.parameters():
            param.requires_grad = True
        target_layers = [model.layer4[-1]]
        # cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
        cam = HiResCAM(model=model, target_layers=target_layers, use_cuda=False)

    model.to(device)
    model.train()
    model.eval()

    test_metrics = get_metrics_collection(device)
    balanced_test_metrics = get_metrics_collection(device)

    test_step = 0
    predictions = []
    for test_batch in tqdm(test_loader):
        test_step += 1
        if TORCHIO_BACKEND:
            test_batch = prepare_batch(test_batch, config)
        test_images, targets, ids = (
            test_batch["image"].to(device),
            test_batch["label"].to(device),
            test_batch["id"],
        )
        with torch.no_grad():
            # print(test_images.shape)
            preds = model(test_images)
            targets = torch.argmax(targets, dim=1)
            preds = torch.argmax(preds, dim=1)

            test_metrics.update(preds, targets)

            predictions.extend(
                [
                    {
                        "f.eid": id,
                        "label": bool(target.item()),
                        "prediction": bool(pred.item()),
                    }
                    for id, target, pred in zip(ids, targets, preds)
                ]
            )
        if compute_activations:
            grayscale_cam = cam(input_tensor=test_images)
            for image, heatmap, id, target, pred in zip(
                test_images, grayscale_cam, ids, targets, preds
            ):
                image = image.permute(1, 2, 0).clamp(min=0.0, max=1.0).numpy()
                visualization = show_cam_on_image(image, heatmap, use_rgb=True)
                image = Image.fromarray(visualization)
                target = bool(target.item())
                pred = bool(pred.item())
                filename = f"{target}_as_{pred}_{id}.jpg"
                image.save(f"{activations_path}/{filename}")

    for test_batch in tqdm(balanced_test_loader):
        if TORCHIO_BACKEND:
            test_batch = prepare_batch(test_batch, config)
        test_images, targets, ids = (
            test_batch["image"].to(device),
            test_batch["label"].to(device),
            test_batch["id"],
        )
        with torch.no_grad():
            preds = model(test_images)
            targets = torch.argmax(targets, dim=1)
            preds = torch.argmax(preds, dim=1)
            balanced_test_metrics.update(preds, targets)

    test_metrics_values = test_metrics.compute()
    balanced_test_metrics_values = balanced_test_metrics.compute()
    
    results_string = '\nTest results:\n'
    for test_metric_name, test_metric_value in test_metrics_values.items():
        results_string += f"{test_metric_name}: {test_metric_value:.4f}\n"

    balanced_results_string = '\nBalanced test results:\n'
    for balanced_test_metric_name, balanced_test_metric_value in balanced_test_metrics_values.items():
        balanced_results_string += f"balanced_{balanced_test_metric_name}: {balanced_test_metric_value:.4f}\n"
    
    print(results_string)
    print(balanced_results_string)

    checkpoint_metric = test_metrics_values["balanced_accuracy"].item()

    if os.path.exists(f"{experiment_dir}/best_model.pth"):
        os.rename(
            f"{experiment_dir}/best_model.pth",
            f"{experiment_dir}/model_{ICD_CODE}_{checkpoint_metric:.4f}.pth",
        )

    sorted_predictions = sorted(predictions, key=lambda x: x["f.eid"])
    pd.DataFrame.from_dict(sorted_predictions).to_csv(
        f"{experiment_dir}/test_predictions.csv", index=False
    )

    metrics_per_feature = calculate_fairness(
            predictions_filepath=f"{experiment_dir}/test_predictions.csv",
        )
    
    with open(f"{experiment_dir}/test_metrics.txt", "w") as writer:
        writer.write("Results summary file\n")
        writer.write(results_string)
        writer.write(balanced_results_string)
        writer.write("\n\n\nFairness metrics:\n")
        for protected_feature, metrics in metrics_per_feature.items():
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

    return test_metrics_values, balanced_test_metrics_values


if __name__ == "__main__":
    test(
       "logs/sample_experiment",
        compute_activations=False,
    )
