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
from utils.criterion_utils import SingleTrafficPositionLoss  

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

    def test(self, path_test):
        self.model.eval()
        progress_bar = tqdm(total=len(self.test_loader), desc='Testing', leave=False, mininterval=10)
        with torch.no_grad():
            for batch in self.test_loader:
                self.process_batch(batch)
                progress_bar.update(1)
        progress_bar.close()
        self.evaluator.return_collection(print_output=True, train=False, return_loss=True, save_in_file=True, filepath=path_test)

def main(config_path: str):
    base_root = Path(__file__).resolve().parents[2]

    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    set_seed(config["seed"]) 
    model_type = config["model_type"]


    assert model_type in ['MaskedSequenceTransformer'], f"Model type {model_type} not supported"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    path = os.path.join(base_root, "fco", "TFCO", "test_results")
    if not os.path.exists(path):
        os.makedirs(path)

    filename = generate_file_name(config["sequence_len"], config["min_timesteps_seen"], config["dataset_name"])
    path_test = os.path.join(path, filename)
    if not os.path.exists(path_test):
        os.makedirs(path_test)

    log_file = os.path.join(path_test, 'test.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO) 
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    with open(os.path.join(path_test, "config.yaml"), 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    # Prepare the test dataset
    full_data_path = os.path.join(base_root, config["data_path"])
    test_dataset = SequenceTfcoDataset(
        dataset_path=[full_data_path],
        sequence_len=config['sequence_len'],
        max_vehicles=config['max_vehicles'],
        min_timesteps_seen=config['min_timesteps_seen'],
        split=config['test_split'],  
        radius=config['radius'],
        centerpoint=config['centerpoint']
    )
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=8)

    # Load the trained model
    logger.info('Loading model')
    if config["model_type"] == 'MaskedSequenceTransformer':
        model = MaskedSequenceTransformer(
        sequence_len=config['sequence_len'],
        max_vehicles=config['max_vehicles'],
        **config['network_configs']['MaskedTransformer']
        )
        model.load_state_dict(torch.load(config["model_path"], map_location=device))
        model.to(device)

    # Define the criterion (loss function)
    criterion = SingleTrafficPositionLoss(
        distance_weight=config['distance_weight'],
        class_weight=config['class_weight'],
        soft_weight=config["soft_weight"]
    )
    name = config["project_name"] + "_test"
  
    start_wandb(config, config["network_configs"], filename, project_name=name, mode=config["wandb_mode"])
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
    tester.test(path_test)
    logger.info('Finished testing')

if __name__ == '__main__':
    main(config_path="configs/test_config.yaml")

