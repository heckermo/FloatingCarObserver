import logging
import os
import time
from typing import Dict, Tuple, Union
import yaml
from pathlib import Path
from codecarbon import EmissionsTracker
import csv 

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

from models import MaskedSequenceTransformer
from utils.autoencoder_utils import prepare_path_structure
from utils.path_utils import generate_file_name
from utils.criterion_utils import CombinedLoss, TrafficPositionLoss, SingleTrafficPositionLoss
from utils.dataset_utils import SequenceTfcoDataset
from utils.scheduler_utils import create_scheduler
from utils.train_utils import set_seed
from utils.wandb_utils import start_wandb
from utils.intraining_evaluation import InTrainingEvaluator
torch.backends.cudnn.benchmark = True


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: Union[CombinedLoss, TrafficPositionLoss],
        optimizer: optim.Optimizer,
        scheduler: lr_scheduler._LRScheduler,
        device: torch.device,
        evaluator: InTrainingEvaluator,
        config: Dict,
        path: str,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.evaluator = evaluator
        self.config = config
        self.path = path
        self.best_val_loss = float('inf')
        self.use_scaler = False

        if self.use_scaler:
            self.scaler = torch.cuda.amp.GradScaler()

    def process_batch(self, batch: Tuple[torch.Tensor, torch.Tensor], train: bool = True):
        input_tensor, target_tensor, indexes = batch
        input_tensor, target_tensor = input_tensor.to(self.device), target_tensor.to(self.device)

        if self.use_scaler:
            with torch.amp.autocast(device_type=self.device.type):
                outputs = self.model(input_tensor)
                loss, additional_information = self.criterion(outputs, target_tensor)
        else:
            outputs = self.model(input_tensor)
            loss, additional_information = self.criterion(outputs, target_tensor)

        if train:
            if self.use_scaler:
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
            else:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
        
        self.evaluator.collect(loss.item(), additional_information, batch, outputs, collect_raw_data=not train)

        return loss, additional_information

    def train_epoch(self, epoch: int):
        self.model.train()
        progress_bar = tqdm(total=len(self.train_loader), desc=f'Epoch {epoch}', leave=False, mininterval=10)

        for batch in self.train_loader:
            self.process_batch(batch, train=True)
            progress_bar.update(1)
        progress_bar.close()
        self.evaluator.return_collection(print_output=True, train=True)

    def validate_epoch(self, epoch: int):
        self.model.eval()
        progress_bar = tqdm(total=len(self.val_loader), desc=f'Validation', leave=False, mininterval=10)
        with torch.no_grad():
            for batch in self.val_loader:
                self.process_batch(batch, train=False)
                progress_bar.update(1)
        progress_bar.close()
        avg_val_loss = self.evaluator.return_collection(print_output=True, train=False, return_loss=True)

        return avg_val_loss

    def save_model(self, epoch: int):
        save_path = os.path.join(self.path, f'model_epoch_{epoch}.pth')
        torch.save(self.model.state_dict(), save_path)
        logging.getLogger(__name__).info(f'Model saved at {save_path}')

    def train(self, config, filepath):

        file_dir = os.path.join(filepath, "additional_information")
        csv_file = os.path.join(file_dir, "epoch_time.csv")
        csv_fieldnames = ["epoch", "duration"]

        with open (csv_file, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
            writer.writeheader()
        
        for epoch in range(self.config['num_epochs']):
            tracker = EmissionsTracker(project_name=config["project_name"], output_dir=file_dir, save_to_file=config["track_emissions"], log_level="critical", measure_power_secs=60)
            tracker.start()
            start_time = time.time()

            self.train_epoch(epoch)

            if epoch % self.config['validation_frequency'] == 0:
                val_loss = self.validate_epoch(epoch)

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_model(epoch)
            tracker.stop()
            epoch_duration = time.time() - start_time
            csv_row = {"epoch": epoch, "duration": round(epoch_duration/60, 2)}

            with open (csv_file, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
                writer.writerow(csv_row)

def main(config_file: str):
    base_root = Path(__file__).resolve().parents[2]

    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    network_configs = config['network_configs']
    set_seed(config["seed"]) 

    filename = generate_file_name(config["sequence_len"], config["min_timesteps_seen"], config["dataset_name"])
    path = prepare_path_structure(filename, base_path='trained_models', config_file=config_file)

    log_file = os.path.join(path, 'training.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO) 
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    with open(os.path.join(base_root, config["dataset_path"], config["dataset_name"][0], "config.yaml"), 'r') as f:
        data_config = yaml.load(f, Loader=yaml.FullLoader)

    num_grids = 0 #data_config["num_grids"]
    if num_grids > 1:
        overlap_mode = True 
    else:
        overlap_mode = False    


    filter_parameter = list()
    if config["filter"]:
        try:
            filter_mode = config["vehicle_filter_mode"]
            selection_mode = config["vehicle_selection_mode"]
            k = config["num_vehicles"]

            filter_parameter = [filter_mode, selection_mode, k]
        except KeyError as e:
            print(f"Check Config {e}")
    else:
        filter_parameter = None

    print(filter_parameter)

    train_dataset = SequenceTfcoDataset(
        dataset_path=[os.path.join(base_root, config['dataset_path'], n) for n in config['dataset_name']],
        sequence_len=config['sequence_len'],
        max_vehicles=config['max_vehicles'],
        min_timesteps_seen=config['min_timesteps_seen'],
        split=config['train_split'],
        radius=config['radius'],
        centerpoint=config['centerpoint'],
        overlap_mode=overlap_mode,
        filter = filter_parameter
    )
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=8)

    val_dataset = SequenceTfcoDataset(
        dataset_path=[os.path.join(base_root, config['dataset_path'], n) for n in config['dataset_name']],
        sequence_len=config['sequence_len'],
        max_vehicles=config['max_vehicles'],
        min_timesteps_seen=config['min_timesteps_seen'],
        split=config['val_split'],
        radius=config['radius'],
        centerpoint=config['centerpoint'],
        overlap_mode=overlap_mode,
        filter = filter_parameter
    )
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=8)

    start_wandb(config, network_configs, filename, project_name=config["project_name"], mode=config["wandb_mode"])

    # Create the model
    logger.info('Creating model')
    model = MaskedSequenceTransformer(sequence_len=config['sequence_len'], max_vehicles=config['max_vehicles'], **network_configs['MaskedTransformer'])

    model.to(device)

    if config['load_complete_model']:
        model.load_state_dict(torch.load(config['load_complete_model']), strict=False)

    criterion = SingleTrafficPositionLoss(distance_weight=config['distance_weight'], class_weight=config['class_weight'], soft_weight=config["soft_weight"])

    # get the total count of trainable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Total trainable parameters: {total_params}')
    optimizer = optim.AdamW(model.parameters(), lr=config['scheduler']['init_lr'])
    scheduler = create_scheduler(optimizer, config)

    evaluator = InTrainingEvaluator(config=config, path=path)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        evaluator=evaluator,
        config=config,
        path=path,
    )

    logger.info('Starting the training')
    trainer.train(config=config, filepath=path)
    logger.info('Finished training')

if __name__ == '__main__':
    main('configs/train_config.yaml')
