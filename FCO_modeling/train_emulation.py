import datetime
import logging
import math
import os
import pickle
import shutil
import time
from statistics import mean
from typing import List, Tuple
import random
import glob
import sys
import yaml
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import PIL.Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import wandb
from typing import Union
from utils.nn_utils.create_cnn import CustomResNet
from utils.nn_utils.vit_encoder_decoder import ViTEncoderDecoder
from utils.polygon_to_tensor import create_bev_tensor
from utils.vit_pytorch import ViT
from utils.nn_utils.training_utils import lr_warmup_decay

torch.backends.cudnn.benchmark = True


class EmulationDataset(Dataset):
    def __init__(
        self,
        dataset_path: List[str],
        image_size: int,
        vehicle_representation: str,
        pre_generate_images: bool = False,
        balance_dataset: bool = False,
        remove_outliers: bool = True,
    ):  
        self.dataset = pd.DataFrame()
        if isinstance(dataset_path, str):
            dataset_path = [dataset_path]
        for p in dataset_path:
            df = pd.read_pickle(f'{p}dataset.pkl')
            self.dataset = pd.concat([self.dataset, df])
            
        # Only keep samples within a certain radius
        if remove_outliers:
            old_len = len(self.dataset)
            self.dataset = self.dataset[
                self.dataset["vector"].apply(
                    lambda x: math.sqrt(x[0] ** 2 + x[1] ** 2) < 50
                )
            ]
            print(f"Filtered dataset from {old_len} to {len(self.dataset)} samples")

        self.bev_ids = self.dataset["bev_pointer"].unique()

        # Calculate the balance between positive and negative samples
        positive_count = len(self.dataset[self.dataset["detected_label"] == 1])
        negative_count = len(self.dataset[self.dataset["detected_label"] == 0])
        print(f"Distribution from positive to negative samples: {positive_count/negative_count}")

        # Balance the dataset
        if balance_dataset:
            positive_samples = self.dataset[self.dataset["detected_label"] == 1]
            negative_samples = self.dataset[self.dataset["detected_label"] == 0]
            if len(negative_samples) > len(positive_samples):
                negative_samples = negative_samples.sample(
                    n=len(positive_samples), random_state=42
                )
            else:
                positive_samples = positive_samples.sample(
                    n=len(negative_samples), random_state=42
                )
            print(
                f"Balanced dataset with {len(positive_samples)} positive samples and {len(negative_samples)} negative samples"
            )
            self.dataset = pd.concat([positive_samples, negative_samples])

        self.bev_data = pd.DataFrame()
        for p in dataset_path:
            df = pd.read_pickle(f'{p}bev_data.pkl')
            self.bev_data = pd.concat([self.bev_data, df])

        self.bev_data.set_index("id", inplace=True)
        self.bev_data = self.bev_data[self.bev_data.index.isin(self.bev_ids)]

        print(f'There is a total of {len(self.bev_data)} BEV images')

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.image_size = image_size
        self.vehicle_representation = vehicle_representation

        self.radius = 50

        if pre_generate_images:
            self._generate_bev_images()
        self.pre_generate_images = pre_generate_images

    def _generate_bev_images(self):
        self.bev_tensors = {}
        for idx, data in tqdm(
            self.bev_data.iterrows(),
            total=len(self.bev_data),
            desc="Generating BEV Images",
            mininterval=10, # only update every 10 seconds
        ):
            bev_tensor = create_bev_tensor(
                building_polygons=data["building_polygons"],
                vehicle_infos=data["vehicle_infos"],
                rotation_angle=0,
                x_offset=data["ego_x"],
                y_offset=data["ego_y"],
                image_size=self.image_size,
                vehicle_representation=self.vehicle_representation,
                radius_covered=self.radius,
            )
            self.bev_tensors[idx] = bev_tensor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        data = self.dataset.iloc[idx]

        vector_tensor = torch.tensor(data["vector"], dtype=torch.float)
        vector_tensor = vector_tensor / self.radius  # Scale vector to [0,1]

        target_tensor = torch.tensor(data["detected_label"], dtype=torch.float)

        if self.pre_generate_images:
            bev_tensor = self.bev_tensors[data["bev_pointer"]]
        else:
            bev_data = self.bev_data.loc[data["bev_pointer"]]
            bev_tensor = create_bev_tensor(
                building_polygons=bev_data["building_polygons"],
                vehicle_infos=bev_data["vehicle_infos"],
                rotation_angle=0,
                x_offset=bev_data["ego_x"],
                y_offset=bev_data["ego_y"],
                image_size=self.image_size,
                vehicle_representation=self.vehicle_representation,
                radius_covered=self.radius,
            )

        return bev_tensor, vector_tensor, target_tensor


def binary_metrics(
    predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5
) -> Tuple[float, float, float, int]:
    binary_preds = (predictions >= threshold).float()
    count_ones = torch.sum(binary_preds == 1).item()

    true_positives = (binary_preds * targets).sum()
    false_positives = (binary_preds * (1 - targets)).sum()
    false_negatives = ((1 - binary_preds) * targets).sum()

    accuracy = (
        true_positives
        + (binary_preds.shape[0] - true_positives - false_positives - false_negatives)
    ) / binary_preds.shape[0]
    precision = true_positives / (true_positives + false_positives + 1e-9)
    recall = true_positives / (true_positives + false_negatives + 1e-9)

    return accuracy.item(), precision.item(), recall.item(), int(count_ones)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: LambdaLR,
        device: str,
        filename: str,
        mixed_precision: bool = False,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        if mixed_precision:
            self.scaler = GradScaler(device=device, enabled=mixed_precision)
        self.best_val_loss = float("inf")
        self.filename = filename
        self.mixed_precision = mixed_precision

    def train(self, num_epochs: int):
        for epoch in range(num_epochs):
            self.train_epoch(epoch)
            self.scheduler.step()

            if epoch % 5 == 0:
                eval_metrics = self.evaluate(epoch)
                avg_eval_loss = mean(eval_metrics["eval_loss"])
                if avg_eval_loss < self.best_val_loss:
                    self.best_val_loss = avg_eval_loss
                    self.save_best_model()

    def train_epoch(self, epoch: int):
        self.model.train()
        running_loss = 0.0
        acc, pre, rec, ones = [], [], [], []

        for images, vectors, labels in tqdm(
            self.train_loader, desc=f"Epoch {epoch} [Train]", 
            mininterval=10, # only update every 10 seconds
        ):
            images = images.to(self.device)
            vectors = vectors.to(self.device)
            labels = labels.to(self.device).unsqueeze(1).float()
            self.optimizer.zero_grad()

            with torch.autocast(enabled=self.mixed_precision, device_type=self.device):
                outputs = self.model(images, vectors)
                outputs = torch.clamp(outputs, min=-10, max=10)
                loss = self.criterion(outputs, labels)
            
            if self.mixed_precision:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            running_loss += loss.item()

            metrics = binary_metrics(outputs, labels)
            acc.append(metrics[0])
            pre.append(metrics[1])
            rec.append(metrics[2])
            ones.append(metrics[3])

        self.log_training_metrics(epoch, running_loss, acc, pre, rec, ones)

    def evaluate(self, epoch: int):
        self.model.eval()
        eval_metrics = {"eval_acc": [], "eval_pre": [], "eval_rec": [], "eval_loss": []}
        with torch.no_grad():
            for images, vectors, labels in tqdm(
                self.val_loader, desc=f"Epoch {epoch} [Eval]",
                mininterval=10, # only update every 10 seconds
            ):
                images = images.to(self.device)
                vectors = vectors.to(self.device)
                labels = labels.to(self.device).unsqueeze(1).float()

                outputs = self.model(images, vectors)
                loss = self.criterion(outputs, labels)
                eval_metrics["eval_loss"].append(loss.item())

                metrics = binary_metrics(outputs, labels)
                eval_metrics["eval_acc"].append(metrics[0])
                eval_metrics["eval_pre"].append(metrics[1])
                eval_metrics["eval_rec"].append(metrics[2])

        self.log_evaluation_metrics(epoch, eval_metrics)
        torch.save(self.model, 'model.pt')
        return eval_metrics

    def save_best_model(self):
        model_dir = os.path.join("trained_emulation_models", self.filename)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        torch.save(
            self.model.state_dict(),
            os.path.join(model_dir, "best_val_model_state_dict.pth"),
        )
        torch.save(self.model, os.path.join(model_dir, "best_val_model.pt"))
        print(f"Saved best model to {model_dir}")

    def log_training_metrics(
        self,
        epoch: int,
        running_loss: float,
        acc: List[float],
        pre: List[float],
        rec: List[float],
        ones: List[int],
    ):
        average_acc = sum(acc) / len(acc) if acc else 0
        average_pre = sum(pre) / len(pre) if pre else 0
        average_rec = sum(rec) / len(rec) if rec else 0
        average_ones = sum(ones) / len(ones) if ones else 0

        print(
            f"Epoch: {epoch} - Loss: {running_loss:.4f}, Accuracy: {average_acc:.4f}, Precision: {average_pre:.4f}, Recall: {average_rec:.4f}, Ones: {average_ones:.4f}, LR: {self.optimizer.param_groups[0]['lr']}"
        )
        wandb.log(
            {
                "epoch": epoch,
                "train_loss": running_loss,
                "train_acc": average_acc,
                "train_pre": average_pre,
                "train_rec": average_rec,
                "train_ones": average_ones,
            }
        )

    def log_evaluation_metrics(self, epoch: int, eval_metrics: dict):
        average_eval_acc = (
            sum(eval_metrics["eval_acc"]) / len(eval_metrics["eval_acc"])
            if eval_metrics["eval_acc"]
            else 0
        )
        average_eval_pre = (
            sum(eval_metrics["eval_pre"]) / len(eval_metrics["eval_pre"])
            if eval_metrics["eval_pre"]
            else 0
        )
        average_eval_rec = (
            sum(eval_metrics["eval_rec"]) / len(eval_metrics["eval_rec"])
            if eval_metrics["eval_rec"]
            else 0
        )
        average_eval_loss = (
            sum(eval_metrics["eval_loss"]) / len(eval_metrics["eval_loss"])
            if eval_metrics["eval_loss"]
            else 0
        )
        print(
            f"Epoch: {epoch} - Eval Loss: {average_eval_loss:.4f}, Eval Accuracy: {average_eval_acc:.4f}, Eval Precision: {average_eval_pre:.4f}, Eval Recall: {average_eval_rec:.4f}"
        )
        wandb.log(
            {
                "epoch": epoch,
                "eval_loss": average_eval_loss,
                "eval_acc": average_eval_acc,
                "eval_pre": average_eval_pre,
                "eval_rec": average_eval_rec,
            }
        )


def setup_training_components(model: nn.Module, config: dict) -> Tuple[nn.Module, optim.Optimizer, LambdaLR, str]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    criterion = nn.BCEWithLogitsLoss()  

    optimizer = optim.AdamW(
        model.parameters(), lr=config["network_config"]["init_lr"], weight_decay=1e-2
    )

    total_epochs = config["network_config"]["num_epochs"]
    warmup_epochs = 10
    scheduler = LambdaLR(
        optimizer, lr_lambda=lr_warmup_decay(warmup_epochs, total_epochs)
    )

    return criterion, optimizer, scheduler, device


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(cfg_file: str = "configs/config_networks.yaml"):
    with open(cfg_file, "r") as f:
        config = yaml.safe_load(f)
    # Set up configurations and logging
    set_seed()
    filename = f'{config["network_config"]["network_type"]}_{config["network_config"]["dataset_path"].split("/")[-1]}_{config["network_config"]["file_extension"]}'
    model_dir = os.path.join("trained_emulation_models", filename)
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    os.makedirs(model_dir)

    # Copy the config file to the model directory
    shutil.copy("configs/config_networks.yaml", os.path.join(model_dir, "config_networks.yaml"))

    logging.basicConfig(
        filename=os.path.join(model_dir, "log.log"), level=logging.INFO
    )
    logging.info("Starting training at " + time.strftime("%H:%M:%S", time.localtime()))

    # Initialize model
    if config["network_config"]["network_type"] == "ViT":
        model = ViT(
            **config["vit_config"],
            image_size=config["network_config"]["image_size"],
            sigmoid_activation=False,
        )
    elif config["network_config"]["network_type"] == "ResNet":
        model = CustomResNet(sigmoid_activation=False)
    elif config["network_config"]["network_type"] == "ViTEncoderDecoder":
        model = ViTEncoderDecoder(
            **config["vit_config"],
            image_size=config["network_config"]["image_size"],
            sigmoid_activation=False,
            decoder_depth=2,
        )
    else:
        raise ValueError("Model type not supported")
    
    pre_trained_path = config["network_config"].get("pretrained_path", None)
    if pre_trained_path is not None:
        print(f"Loading pre-trained model from {pre_trained_path}")
        state_dict = torch.load(pre_trained_path)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print("The following keys were missing and not loaded into the model:")
            print("\n".join(missing_keys))
        else:
            print("No missing keys.")

        if unexpected_keys:
            print("The following keys in the loaded state_dict were not expected by the model:")
            print("\n".join(unexpected_keys))
        else:
            print("No unexpected keys.")

    # Initialize wandb
    wandb.init(
        project="emulation",
        name=f"{datetime.datetime.now().strftime('%m-%d-%H-%M')}",
        mode="disabled",
    )
    # Create all dirs for the train and val datasets
    train_dirs, val_dirs, _ = create_train_val_dirs(config["network_config"]["dataset_path"], config["network_config"]["val_dataset_path"], config["network_config"]["test_dataset_path"], allow_overlapping=False)

    # Create datasets and loaders
    train_set = EmulationDataset(
        train_dirs,
        config["network_config"]["image_size"],
        config["network_config"]["vehicle_representation"],
        pre_generate_images=True,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=config["network_config"]["batch_size"],
        shuffle=True,
        num_workers=0,
    )

    val_set = EmulationDataset(
        val_dirs,
        config["network_config"]["image_size"],
        config["network_config"]["vehicle_representation"],
        pre_generate_images=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config["network_config"]["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    # Setup training components
    criterion, optimizer, scheduler, device = setup_training_components(model, config)

    # Get the information if the training is mixed precision
    mixed_precision = config["network_config"].get("mixes_precision", False)

    # Initialize Trainer
    trainer = Trainer(
        model, train_loader, val_loader, criterion, optimizer, scheduler, device, filename, mixed_precision
    )

    # Start training
    trainer.train(config["network_config"]["num_epochs"])

    # Save final model
    torch.save(
        trainer.model.state_dict(),
        os.path.join("trained_emulation_models", filename, "final_model_state_dict.pth"),
    )
    torch.save(
        trainer.model,
        os.path.join("trained_emulation_models", filename, "final_model.pth"),
    )
    wandb.save(os.path.join("trained_emulation_models", filename, "final_model.pth"))
    wandb.save(
        os.path.join("trained_emulation_models", filename, "final_model_state_dict.pth")
    )
    logging.info(
        f"Finished training at {time.strftime('%H:%M:%S', time.localtime())}"
    )

    # Test the model (assuming test_model is defined)
    from test_emulation import test_model

    average_test_acc, average_test_pre, average_test_rec = test_model(filename)
    wandb.log(
        {
            "average_test_acc": average_test_acc,
            "average_test_pre": average_test_pre,
            "average_test_rec": average_test_rec,
        }
    )


def process_dataset_paths(dataset_paths: Union[str, List[str]]) -> List[str]:
    dirs = []

    # Ensure dataset_paths is a list
    if isinstance(dataset_paths, str):
        dataset_paths = [dataset_paths]

    for path in dataset_paths:
        if path.endswith('*'):
            # Expand wildcard patterns using glob
            matched_paths = glob.glob(path)
            for matched_path in matched_paths:
                if matched_path.endswith('dataset.pkl'):
                    dirs.append(matched_path[:-11])
                elif matched_path.endswith('bev_data.pkl'):
                    dirs.append(matched_path[:-12])
        else:
            dirs.append(path)
    dirs = list(set(dirs))
    return dirs

def create_train_val_dirs(
    train_dataset_path: Union[str, List[str]],
    val_dataset_path: Union[str, List[str]],
    test_dataset_path: Union[str, List[str]],
    allow_overlapping: bool = False
) -> Tuple[List[str], List[str], List[str]]:
    # Process each dataset path using the helper function
    train_dirs = process_dataset_paths(train_dataset_path)
    val_dirs = process_dataset_paths(val_dataset_path)
    test_dirs = process_dataset_paths(test_dataset_path)

    if not allow_overlapping:
        # Remove overlapping directories
        train_set = set(train_dirs)
        val_set = set(val_dirs)
        test_set = set(test_dirs)

        # Exclude train dirs from val and test dirs and train and val dirs from test dirs
        train_set = train_set - val_set - test_set
        val_set = val_set# - test_set
        train_dirs, val_dirs, test_dirs = list(train_set), list(val_set), list(test_set)
    
    # check that the train, val and test directories are not empty
    if not train_dirs:
        raise ValueError('No valid training directories found.')
    if not val_dirs:
        raise ValueError('No valid validation directories found.')

    print(f'Training directories: {train_dirs}')
    print(f'Validation directories: {val_dirs}')
    print(f'Test directories: {test_dirs}')
    return train_dirs, val_dirs, test_dirs
            

    

if __name__ == "__main__":
    main()
