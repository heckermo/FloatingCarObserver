# SUMO_FCO

This section of the repository contains the source code for running various FCO modeling approaches within the SUMO simulation where Carla co-simulation is not required (i.e., 2D raytracing, 3D raytracing, emulation). This README is divided into two parts: running the modeling approaches with pre-trained weights for the emulation approach, and training the emulation networks.

We provide a conda environment to run all the code, specified in `environment.yaml`.

## Running FCO Modeling

All FCO modeling approaches can be initialized with the `detector_factory` function. Relevant configurations are provided as keyword arguments to the `detector_factory` function. For example, to initialize a detector using the emulation modeling approach:

```python
detector = detector_factory('nn', model_path='/path_to_trained_model_weights', building_polygons='/path_to_sumo_polygon_file')
```

Once initialized, the detector can determine which vehicles are seen by the current FCOs at each SUMO timestep. For example:

```python
detected_vehicles = detector.detect(fcos)
```

Here, `fcos` is a list of vehicles acting as floating car observers, and the function returns a dictionary of detected vehicles for each FCO.

We provide two example scripts:
- `FCO.py`: Defines and uses a single FCO for detections.
- `multi_FCO.py`: Sets a penetration rate and area of interest, and configures the FCOs accordingly. This script also generates the pre-training dataset for the TFCO.

## Training the Emulation Modeling Approach

The emulation modeling approach is based on neural networks, which need to be trained. A dataset can be generated with `generate_emulation_dataset.py`, using configurations in `configs/config_dataset.yaml`. The dataset is saved in `emulation_datasets/`. For faster generation, the process can run in parallel; configure this according to your computational resources.

Train the neural network architectures for a specific dataset with `train_emulation.py`, using configurations in `configs/config_networks.yaml`. Checkpoints are saved under `trained/`.

Evaluate the test dataset with `test_emulation.py`. This creates a `test_results` directory in the training directory, containing `test_results.csv` with insights into the test set results. Additionally, visualizations for each datapoint are saved to analyze the model's performance.

### Credits

The encoder-decoder architecture is built upon the VIT implementation of [vit-pytorch](https://github.com/lucidrains/vit-pytorch). Further, as outlined in the paper, the 2D-raytracing method is based on the implementation of [FCO_Sim](https://github.com/TUM-VT/FTO-Sim).