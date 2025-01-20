"""
This file is used to convert the datapoints from the Carla-SUMO Co-Simulation to the KITTI format in order to train the 3D object detection model.
"""

import os
import sys
import numpy as np
import tqdm
import shutil
import yaml

def analyze_label(label_path, label_storage):
    # load the label file
    class_counters = dict.fromkeys(label_storage.keys(), 0)
    with open(label_path, "r") as f:
        lines = f.readlines()
    if len(lines) == 0:
        class_counters['None'] = 1
    else:
        for line in lines:
            v_type = line.split(" ")[0]
            assert v_type in class_counters.keys(), f"Unknown label type {v_type}"
            class_counters[v_type] += 1
    for key in class_counters.keys():
        label_storage[key].append(class_counters[key])
    return label_storage



if __name__ == "__main__":
    cosimulation_recordings = "Co-simulation/recordings"
    name = "training_TRC"
    base_path = "Co-simulation/kitti_recording"
    target_path = os.path.join(base_path, name)
    training_testing_split = 0.9

    dataset_meta = {
        "name": name,
        "training_testing_split": training_testing_split,}

    if os.path.exists(target_path):
        shutil.rmtree(target_path)
    
    os.makedirs(target_path)

    total_dirs = []
    for dir in os.listdir(cosimulation_recordings):

        shutil.copy(os.path.join(cosimulation_recordings, dir, "co-simulation.yaml"), os.path.join(target_path, f"co-simulation-{dir}.yaml"))
        shutil.copy(os.path.join(cosimulation_recordings, dir, "sensor_setups.yaml"), os.path.join(target_path, f"sensor_setups-{dir}.yaml"))

        # get all dirs in the cosimulation_recordings 
        scenario_dirs = [d for d in os.listdir(os.path.join(cosimulation_recordings, dir)) if os.path.isdir(os.path.join(os.path.join(cosimulation_recordings, dir, d)))]
        for scenario in scenario_dirs:
            timestep_dirs = [d for d in os.listdir(os.path.join(cosimulation_recordings, dir, scenario)) if os.path.isdir(os.path.join(os.path.join(cosimulation_recordings, dir, scenario, d)))]
            for t_dir in timestep_dirs:
                camera_dirs = [d for d in os.listdir(os.path.join(cosimulation_recordings, dir, scenario, t_dir)) if os.path.isdir(os.path.join(os.path.join(cosimulation_recordings, dir, scenario, t_dir, d)))]
                for camera in camera_dirs:
                    total_dirs.append(os.path.join(cosimulation_recordings, dir, scenario, t_dir, camera))
    
    # randomly split the data into training and testing
    np.random.shuffle(total_dirs)
    split_idx = int(len(total_dirs) * training_testing_split)
    training_dirs = total_dirs[:split_idx]
    testing_dirs = total_dirs[split_idx:]
    
    # prepare the target_path
    if os.path.exists(target_path):
        os.system(f"rm -r {target_path}")
    os.makedirs(target_path)


    # iterate over the recordings and save the data
    splits = ['training', 'testing']
    split_counter = {}
    class_counter = {'DontCare': [], 
                    'Car': [],
                    'None': [],}
    for split in splits:
        t_path = os.path.join(target_path, split)
        os.makedirs(os.path.join(t_path, "image_2"))
        os.makedirs(os.path.join(t_path, "label_2"))
        os.makedirs(os.path.join(t_path, "calib"))
        counter = 0
        i = 0 
        for recording_dir in tqdm.tqdm(training_dirs if split == 'training' else testing_dirs):
            img_path = os.path.join(recording_dir, "rgb_image.png")
            label_path = os.path.join(recording_dir, "kitti_datapoint.txt")
            calib_path = os.path.join(recording_dir, "calib.txt")
            class_counter = analyze_label(label_path, class_counter)
            if class_counter['Car'][-1] > 0:
                i+=1
                # copy the data to the target path
                os.system(f"cp {img_path} {os.path.join(t_path, 'image_2', f'{i:06d}.png')}")
                os.system(f"cp {label_path} {os.path.join(t_path, 'label_2', f'{i:06d}.txt')}")
                os.system(f"cp {calib_path} {os.path.join(t_path, 'calib', f'{i:06d}.txt')}")
                counter += 1
        split_counter[split] = counter
    
    dataset_meta['class_mean'] = {key: np.mean(class_counter[key]).item() for key in class_counter.keys()}
    
    with open(os.path.join(target_path, "dataset_metas.yaml"), "w") as f:
        yaml.dump(dataset_meta, f)


