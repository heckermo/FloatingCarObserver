import logging
import os
from datetime import datetime
from typing import Dict, Tuple
import importlib
import yaml
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import MaskedSequenceTransformer
from baseline_models import KalmanFilter, LastKnowledge
from utils.dataset_utils import SequenceTfcoDataset
from utils.intraining_evaluation import InTrainingEvaluator
from utils.train_utils import set_seed
from utils.wandb_utils import start_wandb
from utils.path_utils import generate_file_name
import shutil
from utils.criterion_utils import SingleTrafficPositionLoss  # Import your loss function

torch.backends.cudnn.benchmark = True

class Tester:
    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
        evaluator: InTrainingEvaluator,
        cfg: Dict,
        path: str,
    ):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.criterion = criterion
        self.device = device
        self.evaluator = evaluator
        self.cfg = cfg
        self.path = path
        # Initialize GradScaler for mixed precision if needed
        self.scaler = torch.amp.GradScaler()
    
    def process_batch(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        input_tensor, target_tensor, indexes = batch
        input_tensor, target_tensor = input_tensor.to(self.device), target_tensor.to(self.device)

        # Mixed precision context if applicable
        with torch.amp.autocast(device_type=self.device.type):
            outputs = self.model(input_tensor)
            loss, additional_information = self.criterion(outputs, target_tensor)

        self.evaluator.collect(loss.item(), additional_information, batch, outputs)
        return loss.item(), additional_information

    def test(self):
        self.model.eval()
        progress_bar = tqdm(total=len(self.test_loader), desc='Testing', leave=False, mininterval=10)
        with torch.no_grad():
            for batch in self.test_loader:
                self.process_batch(batch)
                progress_bar.update(1)
        progress_bar.close()
        self.evaluator.return_collection(print_output=True, train=False)

def main(model_type: str, model_path: str, data_path: str):
    base_root = Path(__file__).resolve().parents[2]

    assert model_type in ['MaskedSequenceTransformer'], f"Model type {model_type} not supported"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    path = os.path.join(base_root, model_path, 'test_results')
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    
    # use the cfg from the trained model
    cfg_path = os.path.join(base_root, "fco/TFCO", "configs/train_config.yaml")
    with open(cfg_path, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader) 

    set_seed(config["seed"]) 

    log_file = os.path.join(path, 'test.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO) 
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    # Prepare the test dataset
    full_data_path = os.path.join(base_root, data_path)
    test_dataset = SequenceTfcoDataset(
        dataset_path=[full_data_path],
        sequence_len=config['sequence_len'],
        max_vehicles=config['max_vehicles'],
        min_timesteps_seen=config['min_timesteps_seen'],
        split=config['test_split'],  # Ensure 'test_split' is defined in your config
        radius=config['radius'],
        centerpoint=config['centerpoint']
    )
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=8)

    # Load the trained model
    logger.info('Loading model')
    if model_type == 'MaskedSequenceTransformer':
        model = MaskedSequenceTransformer(
        sequence_len=config['sequence_len'],
        max_vehicles=config['max_vehicles'],
        **config['network_configs']['MaskedTransformer']
        )
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)

    # Define the criterion (loss function)
    criterion = SingleTrafficPositionLoss(
        distance_weight=config['distance_weight'],
        class_weight=config['class_weight']
    )
    name = config["project_name"] + "_test"
    filename = generate_file_name(config["sequence_len"], config["min_timesteps_seen"], config["dataset_name"])

    start_wandb(config, config["network_configs"], filename, project_name=name, mode='online')
    evaluator = InTrainingEvaluator(config=config, path=path)

    tester = Tester(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
        evaluator=evaluator,
        cfg=config,
        path=path,
    )

    logger.info('Starting the testing')
    tester.test()
    logger.info('Finished testing')

if __name__ == '__main__':
    main(model_type='MaskedSequenceTransformer', model_path='trained_models/skc_r100_o0.7_pr0.1_seq10_mint3_20-06_10-24-43/model_epoch_8.pth', data_path="data/tfco_datasets/pen01_radius250")

