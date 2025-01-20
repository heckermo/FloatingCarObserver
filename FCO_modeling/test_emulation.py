import os
import sys
import yaml
import pickle
import random
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.vit_pytorch import ViT
from utils.nn_utils.create_cnn import CustomResNet
from utils.emulation_visualization import visualize_frame
from utils.nn_utils.vit_encoder_decoder import ViTEncoderDecoder

def test_model(filename: str):
    """
    Function to test the trained model on a test dataset.

    Args:
        filename (str): The name of the model directory where the trained model is saved.

    Returns:
        average_test_acc (float): Average test accuracy.
        average_test_pre (float): Average test precision.
        average_test_rec (float): Average test recall.
    """
    with open(os.path.join(filename, 'config_networks.yaml'), 'r') as f:    
        config = yaml.safe_load(f)

    # Device configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create test directory
    test_dir = os.path.join(filename, 'test_results')
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(os.path.join(test_dir, 'test_images'), exist_ok=True)

    # Initialize test df
    test_df = pd.DataFrame(columns=['id', 'label', 'prediction', 'vector_distance'])
    id_counter = 0

    model = torch.load(os.path.join(filename, 'best_val_model.pt'))
    # reinitialize the model
    model.to(device)
    model.eval()

    # Create the test dataset and dataloader
    test_dataset = EmulationDataset(
        dataset_path=config['network_config']['test_dataset_path'],
        image_size=config['network_config']['image_size'],
        vehicle_representation=config['network_config']['vehicle_representation'],
        pre_generate_images=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['network_config']['batch_size'],
        shuffle=False,
        num_workers=0
    )

    # Initialize metrics
    test_acc = []
    test_pre = []
    test_rec = []
    test_loss = []

    criterion = torch.nn.BCEWithLogitsLoss()

    # Evaluate the model on the test dataset
    with torch.no_grad():
        for images, vectors, labels in tqdm(test_loader, desc='Testing'):
            images, vectors, labels = images.to(device), vectors.to(device), labels.to(device)
            outputs = model.forward(images, vectors)
            labels = labels.unsqueeze(1).float()
            raw_labels = labels.clone()
            loss = criterion(outputs, labels)
            test_loss.append(loss.item())

            # Create test images and save data to test df
            for image, vector, labels, output in zip(images, vectors, labels, outputs):
                image, vector, label, output = image.cpu(), vector.cpu(), labels.cpu(), output.cpu()
                prediction = 1 if output.item() > 0.5 else 0
                vector_distance = torch.norm(vector).item()
                test_datapoint = pd.DataFrame({
                    'id': [id_counter],
                    'label': [int(label.item())],
                    'prediction': [prediction], 
                    'vector_distance': [vector_distance]
                })
                test_df = pd.concat([test_df, test_datapoint], ignore_index=True)
                visualize_frame(image, vector*50, 50, os.path.join(test_dir, 'test_images', f'{id_counter}.png'))
                id_counter += 1

            # Compute metrics
            metrics = binary_metrics(outputs, raw_labels)
            accuracy, precision, recall, _ = metrics
            test_acc.append(accuracy)
            test_pre.append(precision)
            test_rec.append(recall)

    average_test_acc = sum(test_acc) / len(test_acc) if test_acc else 0
    average_test_pre = sum(test_pre) / len(test_pre) if test_pre else 0
    average_test_rec = sum(test_rec) / len(test_rec) if test_rec else 0
    average_test_loss = sum(test_loss) / len(test_loss) if test_loss else 0

    # Print test metrics
    print(f"Test Loss: {average_test_loss:.4f}, Test Accuracy: {average_test_acc:.4f}, "
          f"Test Precision: {average_test_pre:.4f}, Test Recall: {average_test_rec:.4f}")
    
    # Save test results
    test_df.to_csv(os.path.join(test_dir, 'test_results.csv'), index=False)

    return average_test_acc, average_test_pre, average_test_rec


if __name__ == '__main__':
   from train_emulation import EmulationDataset, binary_metrics
   average_test_acc, average_test_pre, average_test_rec = test_model('trained_emulation_models/ViTEncoderDecoder_*_init')
