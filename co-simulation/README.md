This section of the repository implements the co-simulation modeling approach using a SUMO-Carla setup. Vehicles in the SUMO simulation become detectable based on a specific sensor configuration and a 3D detection algorithm that processes raw sensor data to produce 3D bounding boxes. These predicted bounding boxes are then mapped back to the simulation vehicles using an IoU-based matching procedure.

Prerequisites:
• Carla must be installed.
• A running Carla server is needed when executing the scripts (e.g., cd {path_to_carla} and ./CarlaUE4.sh if you are on a machine without screen run ./CarlaUE4.sh --RenderOffScreen)

The co-simulation scripts follow a callback structure. For each co-simulation step, a designated callback is invoked, processing the relevant data from the SUMO-Carla simulation.

Below are the main files:

1. generate_kitti_dataset.py
    This script performs a co-simulation run, attaching a chosen sensor setup to vehicles. At each step, raw sensor data is processed using KITTI dataset criteria and saved accordingly. Two configuration files are crucial: 
    • configs/sensor_setup.yaml – defines camera and lidar placements. 
    • configs/co-simulation.yaml – sets sensor penetration rate, max sensors, Carla map, Carla port, etc.
    Since multiple cameras and vehicles can be equipped with sensors, data is stored in the format:
        {map_name}__{weather}/
             {timestep}/
                  {vehicle_id1}/
                        {camera_position_1}/
                             calib.txt
                             kitti_datapoint.txt
                             lidar_points.pcd
                             rgb_image.png
                        {camera_position_2}/
                             ...
                  {vehicle_id2}/
                        ...
    This makes it easier to manage the generated dataset.

2. train_3d_detector.py
    Uses the synthetic KITTI dataset created above to train either a camera-based detector (MonoCon) or a LiDAR-based detector (PointPillar). Before training, the dataset is converted and reorganized into the original KITTI format (numeric indexing). This script permits both algorithms to be started from the same entry point, with shared settings consolidated in configs/d3_detection.yaml (e.g., pre-trained paths, number of epochs).

3. create_map_polgygons.py
    Carla does not provide direct access to building data. To obtain polygons for a top-down representation in SUMO, this script drives a vehicle equipped with lidar around a Carla map to collect point clouds. The collected data is merged, filtered, and clustered using DBSCAN. Polygons are then created and saved in .add.xml format for SUMO.

4. generate_emulation_dataset.py
    Generates a dataset specific to a sensor setup and 3D detection algorithm. The trained detector and its weights must be specified in the script. The output provides all necessary data for the emulation approach, along with a binary label indicating whether a traffic participant is detectable based on 3D detection and the subsequent IoU mapping. This dataset can be used in /FCO-modeling/train_emulation.py.

5. run_detections.py
    Allows all detection modeling approaches (2D ray tracing, 3D ray tracing, co-simulation, emulation) to run simultaneously, enabling performance comparisons for each method against the co-simulation approach.

## Credits

The basic co-simulation setup is provided by the [Carla](https://github.com/carla-simulator/carla) developers. We further utilize the [CARLA-KITTI](https://github.com/fnozarian/CARLA-KITTI) implementation for the evaluation of the KITTI criteria and extended it to our needs. Additionally, we utilize the existing implementations for the 3D detection algorithms, namely [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) and [MonoCon](https://github.com/2gunsu/monocon-pytorch). 
