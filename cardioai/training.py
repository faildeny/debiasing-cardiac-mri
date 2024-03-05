import argparse
import torch
import time
import os
from torch.utils.data import WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
import yaml
from tqdm import tqdm

import cardioai.dataset as dataset
from cardioai.models import get_model
from cardioai.transforms import (
    get_dataset_torchio,
    get_transforms_torchio,
    prepare_batch,
)
from cardioai.visualise import visualise
from cardioai.metrics import get_metrics_collection
from cardioai.test import test

parser = argparse.ArgumentParser(description="Run training on CMR data.")

parser.add_argument(
    "--experiment_dir",
    type=str,
    default="./logs",
    help="Path to save experiment output",
)
parser.add_argument("--gpu_id", type=int, default="0", help="GPU index to use")
parser.add_argument("--fold_id", type=int, default="0", help="Fold index to use")

args = parser.parse_args()
GPU_ID = args.gpu_id
FOLD = args.fold_id
EXPERIMENT_DIR = args.experiment_dir

config_file = "config.yaml"
if EXPERIMENT_DIR != "./logs" and os.path.exists(f"{EXPERIMENT_DIR}/{config_file}"):
    config_file = f"{EXPERIMENT_DIR}/{config_file}"

with open(config_file) as file:
    config = yaml.safe_load(file)
config["params"]["fold"] = FOLD
with open(f"{EXPERIMENT_DIR}/config.yaml", "w") as yaml_file:
    yaml.dump(config, yaml_file)

MODEL = config["params"]["model"]
MAX_EPOCHS = config["params"]["max_epochs"]
BATCH_SIZE = config["params"]["batch_size"]
LEARN_RATE = config["params"]["learning_rate"]
WEIGHT_DECAY = config["params"]["weight_decay"]
INPUT_SIZE = config["params"]["input_size"]
FINE_TUNING = config["params"]["fine_tuning"]
EARLY_LAYERS_TUNING = config["params"]["early_layers_tuning"]
HISTOGRAM_STANDARDIZATION = config["params"]["standardize_histograms"]
GRAY2RGB = config["params"]["gray2rgb"]
MASK_IMAGES = config["params"]["mask_images"]
VIDEO = config["params"]["video"]
VOLUME = config["params"]["volume"]
TIME_STEP = config["params"]["time_step"]

ICD_CODE = str(config["dataset"]["icd_code"])
VIEW = config["dataset"]["view"]
DATASET = config["dataset"]["name"]

DEBUG = config["debug"]
TORCHIO_BACKEND = config["use_torchio"]

if config["dataset"]["val_prevalence"] == 0.5:
    BALANCED_VAL = False
else:
    BALANCED_VAL = False

checkpoint_metric = 'balanced_accuracy'

print(config)

device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

start_time = time.time()
config["params"]["fold"] = FOLD
traind_list, vald_list, testd_list = dataset.get_data_lists(config)
if config["dataset"]["use_synthetic"]:
    synthetic_datad_list = dataset.load_synthetic_dataset(config)
    traind_list += synthetic_datad_list
    print("Added {} synthetic samples to the training set".format(len(synthetic_datad_list)))
    # Print number of positive and negative cases
    train_positive_ids = [sample["id"] for sample in traind_list if sample["label"][1] == 1]
    train_negative_ids = [sample["id"] for sample in traind_list if sample["label"][1] == 0]
    print("Final training set: Positive cases: {}, Negative cases: {}".format(len(train_positive_ids), len(train_negative_ids)))
    if not config["dataset"]["use_real"]:
        traind_list = synthetic_datad_list
        print("!!! Using only synthetic samples for training")

traind_list = dataset.set_sample_weights(traind_list, balance_by_sensitive_features=config["dataset"]["balance_by_sensitive_features"])
vald_list = dataset.set_sample_weights(vald_list, balance_by_sensitive_features=config["dataset"]["balance_by_sensitive_features"])

train_positive_ids = [sample["id"] for sample in traind_list if sample["label"][1] == 1]
train_negative_ids = [sample["id"] for sample in traind_list if sample["label"][1] == 0]
val_positive_ids = [sample["id"] for sample in vald_list if sample["label"][1] == 1]
val_negative_ids = [sample["id"] for sample in vald_list if sample["label"][1] == 0]
test_positive_ids = [sample["id"] for sample in testd_list if sample["label"][1] == 1]
test_negative_ids = [sample["id"] for sample in testd_list if sample["label"][1] == 0]
positive_ids = train_positive_ids + val_positive_ids + test_positive_ids
negative_ids = train_negative_ids + val_negative_ids + test_negative_ids

total_len = len(positive_ids) + len(negative_ids)

num_workers = config["num_workers"]
training_transforms = get_transforms_torchio(config, test=False)
validation_transforms = get_transforms_torchio(config, test=True)

# This will include additional slices from the same patient
traind_ds = get_dataset_torchio(traind_list, training_transforms)
vald_ds = get_dataset_torchio(vald_list, validation_transforms)

sample_weights = [sample["sample_weight"] for sample in traind_ds._subjects]
val_balanced_weights = [sample["sample_weight"] for sample in vald_ds._subjects]

print("Training count =", len(traind_ds), "Validation count =", len(vald_ds))

weighted_sampler = WeightedRandomSampler(
    weights=sample_weights, num_samples=len(sample_weights), replacement=True
)
val_weighted_sampler = WeightedRandomSampler(
    weights=val_balanced_weights, num_samples=len(val_balanced_weights), replacement=True
)
train_loader = torch.utils.data.DataLoader(
    traind_ds,
    sampler=weighted_sampler,
    batch_size=BATCH_SIZE,
    num_workers=num_workers,
)
val_loader = torch.utils.data.DataLoader(
    vald_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers
)
balanced_val_loader = torch.utils.data.DataLoader(
    vald_ds,
    sampler=val_weighted_sampler,
    batch_size=BATCH_SIZE,
    num_workers=num_workers,
)

# visualise(train_loader, config, directory="misc/synth_and_real/")

model = get_model(config)
model = model.to(device)

loss_function = torch.nn.CrossEntropyLoss()
if not WEIGHT_DECAY:
    WEIGHT_DECAY = 0.0
optimizer = torch.optim.Adam(model.parameters(), LEARN_RATE, weight_decay=WEIGHT_DECAY)

if EARLY_LAYERS_TUNING:
    optimizer = torch.optim.Adam(
        [
            {"params": model.stem.parameters(), "lr": LEARN_RATE / 100},
            {"params": model.fc.parameters()},
        ],
        LEARN_RATE,
    )
elif FINE_TUNING:
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True
    optimizer = torch.optim.Adam(model.parameters(), LEARN_RATE)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=20
)

# Start training

val_interval = 20
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []
writer = SummaryWriter(
    log_dir=EXPERIMENT_DIR
    + f"/ICD_{ICD_CODE}_MODEL_{MODEL}_VIEW_{VIEW}_3D_{VOLUME}_VIDEO_{VIDEO}_MASKS_{MASK_IMAGES}_FT_{FINE_TUNING}_HIST_{HISTOGRAM_STANDARDIZATION}_LR_{LEARN_RATE}_WD_{WEIGHT_DECAY}_PREV_{config['dataset']['train_prevalence']}"
)

train_metrics = get_metrics_collection(device)
val_metrics = get_metrics_collection(device)
balanced_val_metrics = get_metrics_collection(device)
test_metrics = get_metrics_collection(device)

val_step = 0
train_len = len(traind_ds)
val_len = len(vald_ds)

for epoch in range(MAX_EPOCHS):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{MAX_EPOCHS}")
    # model.train()
    epoch_loss = 0
    step = 0

    for train_batch in tqdm(train_loader):
        model.train()
        step += 1

        train_batch = prepare_batch(train_batch, config)
        inputs, targets = train_batch["image"].to(device), train_batch["label"].to(
            device
        )
        optimizer.zero_grad()
        preds = model(inputs)
        preds = torch.nn.functional.softmax(preds, dim=1)
        loss = loss_function(preds, targets)
        loss.backward()
        optimizer.step()
        loss = loss.item()
        epoch_loss += loss
        epoch_len = len(traind_ds) // train_loader.batch_size
        targets = torch.argmax(targets, dim=1)
        preds = torch.argmax(preds, dim=1)

        train_step_metrics = train_metrics(preds, targets)

        for name, metric in train_step_metrics.items():
            writer.add_scalar(f"{name}/train", metric.item(), epoch_len * epoch + step)

        train_accuracy = train_step_metrics["accuracy"].item()
        # print(
        #     f"{step}/{epoch_len}, train_loss: {loss:.4f} train_accuracy: {train_accuracy:.4f}"
        # )
        writer.add_scalar("Loss/train", loss, epoch_len * epoch + step)
        
        # writer.add_image(f'Train image label: {targets[0].item()} pred: {preds[0].item()}', inputs[0], epoch)

        if (step) % val_interval == 0:
            model.eval()
            val_loss = 0
            val_metrics.reset()
            balanced_val_metrics.reset()
            val_step += 1
            for val_batch in val_loader:
                val_batch = prepare_batch(val_batch, config)
                val_images, targets = val_batch["image"].to(device), val_batch[
                    "label"
                ].to(device)
                with torch.no_grad():
                    preds = model(val_images)
                    preds = torch.nn.functional.softmax(preds, dim=1)
                    loss = loss_function(preds, targets)
                    targets = torch.argmax(targets, dim=1)
                    preds = torch.argmax(preds, dim=1)
                    loss = loss.item()
                    val_loss += loss

                    val_metrics.update(preds, targets)

            scheduler.step(val_loss)

            val_metrics_values = val_metrics.compute()

            results_string = f"Current epoch: {epoch+1} "
            for name, metric in val_metrics_values.items():
                writer.add_scalar(f"{name}/val", metric, val_step)
                results_string += f"{name}: {metric:.4f} "

            auroc = 0.0

            metric = val_metrics_values[checkpoint_metric].item()

            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), f"{EXPERIMENT_DIR}/best_model.pth")
                print("saved new best metric model")

            print(results_string)

            print(f"Best checkpoint with {checkpoint_metric}: {best_metric:.4f} at epoch {best_metric_epoch}")

            index = val_step
            writer.add_scalar("Loss/val", val_loss, index)
            
            writer.add_scalar(
                "hparam/Learning rate", optimizer.param_groups[0]["lr"], index
            )

            # writer.add_image(f'label: {targets[0].item()} pred: {preds[0].item()}', val_images[0], epoch)

            if BALANCED_VAL:
                for val_batch in balanced_val_loader:
                    if TORCHIO_BACKEND:
                        val_batch = prepare_batch(val_batch, config)
                    val_images, targets = val_batch["image"].to(device), val_batch[
                        "label"
                    ].to(device)
                    with torch.no_grad():
                        preds = model(val_images)
                        preds = torch.nn.functional.softmax(preds, dim=1)
                        targets = torch.argmax(targets, dim=1)
                        preds = torch.argmax(preds, dim=1)
                        balanced_val_metrics.update(preds, targets)

                balanced_val_metrics_values = balanced_val_metrics.compute()

                results_string = f"Current epoch (balanced): {epoch+1} "
                for name, metric in balanced_val_metrics_values.items():
                    writer.add_scalar(f"{name}/val_balanced", metric, index)
                    results_string += f"{name}: {metric:.4f} "

                print(results_string)

    train_metrics.reset()
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

print(
    f"Training completed, checkpoint metric {checkpoint_metric}: {best_metric:.4f} at epoch: {best_metric_epoch}"
)

print("Testing the model...")

test_metrics_values, balanced_test_metrics_values = test(EXPERIMENT_DIR)

elapsed_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
print("Training time: ", elapsed_time)


hparams_dict = {}
hparams_dict["hparam/val_f1"] = best_metric

for test_metric_name, test_metric_value in test_metrics_values.items():
    writer.add_scalar(f"{test_metric_name}/test", test_metric_value, MAX_EPOCHS)
    hparams_dict[f"hparam/test_{test_metric_name}"] = test_metric_value

for balanced_test_metric_name, balanced_test_metric_value in balanced_test_metrics_values.items():
    writer.add_scalar(f"{balanced_test_metric_name}/test_balanced", balanced_test_metric_value, MAX_EPOCHS)
    hparams_dict[f"hparam/test_balanced_{balanced_test_metric_name}"] = balanced_test_metric_value

writer.add_hparams(
    {
        "ICD": ICD_CODE,
        "dataset": DATASET,
        "model": MODEL,
        "positive_cases": len(positive_ids),
        "negative_cases": len(negative_ids),
        "segmentation masks": str(MASK_IMAGES),
        "batch_size": BATCH_SIZE,
        "epochs": MAX_EPOCHS,
        "learning_rate": LEARN_RATE,
        "weight_decay": WEIGHT_DECAY,
        "training time": elapsed_time,
        "video": VIDEO,
        "finetune": FINE_TUNING,
    },
    hparams_dict,
)

writer.close()
