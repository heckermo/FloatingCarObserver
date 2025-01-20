import numpy as np
import torch
from typing import List, Tuple
import wandb
from collections import defaultdict
import math
from tools.visualize_tensors import create_bev_from_rawtensor, create_output_target_comparison
import logging
import pickle
import matplotlib.pyplot as plt
from itertools import chain

class InTrainingEvaluator:
    def __init__(self, config, path, mode: str = 'raw'):
        """
        Initialize the evaluator.

        Args:
            mode (str): Evaluation mode, can be 'raw' or 'bev'.
        """
        self.config = config
        self.path = path
        self.mode = mode  # can be 'raw' or 'bev'
        self.loss = []
        self.additional_losses = defaultdict(list)
        self.additional_information = defaultdict(list)
        self.epoch_counter = 0
        self.results_storage = {'ones': [], 'zeros': []}

    def collect(
        self,
        loss: float,
        additional_loss_information: dict,
        batch: List[torch.Tensor],
        outputs: torch.Tensor,
        collect_raw_data: bool = False,
    ):
        """
        Collects loss and additional information during training.

        Args:
            loss (float): The loss value for the current batch.
            additional_loss_information (dict): Additional loss information.
            batch (List[torch.Tensor]): Batch data, tuple of (input_tensor, target_tensor, indexes).
            outputs (torch.Tensor): Model outputs.
        """
        self.loss.append(loss)

        if self.mode == 'raw':
            # Collect class and regression losses
            self.additional_losses['class_loss'].append(
                additional_loss_information.get('class_loss', 0.0)
            )
            self.additional_losses['regression_loss'].append(
                additional_loss_information.get('position_loss', 0.0)
            )

            target_tensor = batch[1]

            relevant_mask = target_tensor[:, :, 0] == 1

            outputs = outputs[relevant_mask]

            # Apply sigmoid and threshold to outputs
            outputs_binary = (torch.sigmoid(outputs[:, 0]) > 0.5).float()
            outputs_one_mask = outputs_binary == 1

            # get the number of vehicles that are 1 in the outputs_binary
            self.additional_information['count_acc'].append((outputs_binary == 1).sum().item()/(outputs_binary.shape[0] + 1e-6))

            # Calculate the mean euclidean distance between predicted and target positions
            if target_tensor[relevant_mask].shape[0] == 0:
                distance = 0
            else:  
                distance = torch.mean(torch.norm(outputs[:, 1:] - target_tensor[relevant_mask][:, 1:].to(outputs.device), dim=1)).item()
                if collect_raw_data:
                    self.results_storage['zeros'].append(torch.norm(outputs[:, 1:] - target_tensor[relevant_mask][:, 1:].to(outputs.device), dim=1).detach().cpu().numpy())
            self.additional_information['distance'].append(distance)

            # Calculate the mean euclidean distance between predicted and target positions for vehicles that are 1 in the outputs_binary
            target_tensor_device = target_tensor[relevant_mask].to(outputs.device)
            if outputs_one_mask.sum() == 0:
                distance_one = 0
            else:
                distance_one = torch.mean(torch.norm(outputs[outputs_one_mask][:, 1:] - target_tensor_device[outputs_one_mask][:, 1:], dim=1)).item()
                if collect_raw_data:
                    self.results_storage['ones'].append(torch.norm(outputs[outputs_one_mask][:, 1:] - target_tensor_device[outputs_one_mask][:, 1:], dim=1).detach().cpu().numpy())
            self.additional_information['distance_ones'].append(distance_one)

            # Calculate the mean euclidean distance between predicted and target positions for vehicles that are 0 in the outputs_binary
            if (~outputs_one_mask).sum() == 0:
                distance_zero = 0
            else:
                distance_zero = torch.mean(torch.norm(outputs[~outputs_one_mask][:, 1:] - target_tensor_device[~outputs_one_mask][:, 1:], dim=1)).item()
            self.additional_information['distance_zeros'].append(distance_zero)

        elif self.mode == 'bev':
            raise NotImplementedError("BEV mode is not implemented.")
        else:
            raise NotImplementedError(f"Mode '{self.mode}' is not implemented.")

    def return_collection(
        self, print_output: bool = False, train: bool = True, return_loss: bool = False
    ):
        """
        Returns the collected statistics and logs them.

        Args:
            print_output (bool, optional): Whether to print the output. Defaults to False.
            train (bool, optional): Whether it's training mode. Defaults to True.
            return_loss (bool, optional): Whether to return the loss. Defaults to False.

        Returns:
            float, optional: The mean loss if return_loss is True.
        """
        # Increment the epoch counter
        if train:
            self.epoch_counter += 1

        return_dict = {}
        # Compute mean loss
        loss = np.mean(self.loss).item()
        return_dict['loss'] = loss

        # Compute mean of additional losses
        for key, values in self.additional_losses.items():
            return_dict[key] = np.mean(values).item()

        # Compute mean of additional information
        for key, values in self.additional_information.items():
            return_dict[key] = np.mean(values).item()

        # Reset the stored losses and information
        self._reset()

        # Add 'val_' prefix if in validation mode
        if not train:
            return_dict = {('val_' + key): value for key, value in return_dict.items()}

        # Print the output if requested
        if print_output:
            logger = logging.getLogger(__name__)
            logger.info(", ".join(f"{key}: {value:.4f}" for key, value in return_dict.items()))

        # Log the results to wandb
        wandb.log(return_dict, step=self.epoch_counter)

        # Save the raw results
        if len(self.results_storage['ones']) > 0:
            with open(f'{self.path}/raw_results_{self.epoch_counter}.pkl', 'wb') as f:
                pickle.dump(self.raw_results, f)
        
           # create a distribution plot of the distances
            fig, ax = plt.subplots()
            ax.hist(list(chain.from_iterable(self.results_storage['ones'])), bins=20, alpha=0.5, label='ones')
            ax.hist(list(chain.from_iterable(self.results_storage['ones'])), bins=20, alpha=0.5, label='zeros')
            ax.legend()
            plt.savefig(f'{self.path}/hist_{self.epoch_counter}.png')
            logger.info(f"Saved histogram of distances to {self.path}/hist_{self.epoch_counter}.png")
            plt.close()

        if return_loss:
            return loss

    def _reset(self):
        """Resets the stored losses and additional information."""
        self.loss.clear()
        self.additional_losses.clear()
        self.additional_information.clear()
        self.raw_results = {'ones': [], 'zeros': []}

if __name__ == "__main__":
    raise NotImplementedError("This script is not meant to be executed")

else:
    logger = logging.getLogger(__name__)