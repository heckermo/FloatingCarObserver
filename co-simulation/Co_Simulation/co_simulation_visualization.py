import carla
import traci
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import open3d as o3d
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import open3d as o3d
import numpy as np
import cv2

class CosimulationVisualization:
    def __init__(self, synchonization):
        self.synchronization = synchonization
        client = carla.Client('localhost', 3000)    
        self.world = client.get_world()
        self.spectator_camera = self._setup_spectator_camera()

        self.carla_id = None
        self.sumo_id = None
        self.simulation_time = None
        self.image_data = None  # Store the latest image data

        self.base_path = "visualization_images"
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)
    
    def set_vehicle_id(self, sumo_id):
        """
        Sets the new vehicle that should be tracked by the visualization 
        """
        self.sumo_id = sumo_id
        self.carla_id = self.synchronization.synchronization.sumo_carla_idmapping[sumo_id]
    
    
    def update(self):
        """
        Captures the current state of the simulation and saves the visualization
        """
        self.simulation_time = traci.simulation.getTime()

        if self.carla_id is None or self.sumo_id is None:
            print("No vehicle set for the visualization")
            return

        # Update the spectator camera to the position of the vehicle
        vehicle = self.world.get_actor(self.carla_id)
        if vehicle:
            vehicle_transform = vehicle.get_transform()
            spectator_location = vehicle_transform.location + carla.Location(z=40)
            spectator_rotation = carla.Rotation(pitch=-90, yaw=vehicle_transform.rotation.yaw, roll=0)
            spectator_transform = carla.Transform(spectator_location, spectator_rotation)
            self.spectator_camera.set_transform(spectator_transform)
        
        # Save the current spectator image
        self._save_spectator_image()

        # Set the sumo gui view to the vehicle and save the image
        traci.gui.trackVehicle("View #0", self.sumo_id) 
        traci.gui.setZoom("View #0", 400)
        traci.gui.screenshot("View #0", os.path.join(self.base_path, f"sumo_view_{int(self.simulation_time)}.png"))
        

    def _setup_spectator_camera(self):
        """
        Setup the spectator camera in the CARLA simulation
        """
        blueprint_library = self.world.get_blueprint_library()
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '1920')
        camera_bp.set_attribute('image_size_y', '1080')
        camera_bp.set_attribute('fov', '110')

        # Spawn the camera at the initial spectator position
        spectator = self.world.get_spectator()
        camera = self.world.spawn_actor(camera_bp, spectator.get_transform())

        # Set a listener to process each image frame
        camera.listen(lambda image: self._process_image(image))
        return camera
    
    def _process_image(self, image):
        """
        Converts the raw CARLA image to an OpenCV image format and stores it
        """
        # Convert to numpy array and reshape to BGRA format
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))

        # Convert BGRA to BGR for OpenCV
        self.image_data = array[:, :, :3]
    
    def _save_spectator_image(self):
        """
        Saves the current image of the spectator camera as PNG
        """
        if self.image_data is not None:
            filename = os.path.join(self.base_path, f"carla_view_{int(self.simulation_time)}.png")
            cv2.imwrite(filename, self.image_data)

def read_pcd_file(pcd_file_path):
    # Read the point cloud data from the PCD file
    pcd = o3d.io.read_point_cloud(pcd_file_path)
    return pcd

def generate_bev_image(pcd, resolution=0.1, side_range=(-50, 50), fwd_range=(-50, 50), enhance_contrast=True):
    points = np.asarray(pcd.points)
    intensities = np.asarray(pcd.colors)[:, 0] if pcd.colors else None

    x_points = points[:, 0]
    y_points = points[:, 1]
    z_points = points[:, 2]
    side_filter = np.logical_and(y_points > side_range[0], y_points < side_range[1])
    fwd_filter = np.logical_and(x_points > fwd_range[0], x_points < fwd_range[1])
    point_filter = np.logical_and(side_filter, fwd_filter)

    x_points = x_points[point_filter]
    y_points = y_points[point_filter]
    z_points = z_points[point_filter]
    if intensities is not None:
        intensities = intensities[point_filter]
def generate_bev_image(pcd, resolution=0.1, side_range=(-50, 50), fwd_range=(-50, 50), enhance_contrast=True):
    points = np.asarray(pcd.points)
    intensities = np.asarray(pcd.colors)[:, 0] if pcd.colors else None

    x_points = points[:, 0]
    y_points = points[:, 1]
    z_points = points[:, 2]
    side_filter = np.logical_and(y_points > side_range[0], y_points < side_range[1])
    fwd_filter = np.logical_and(x_points > fwd_range[0], x_points < fwd_range[1])
    point_filter = np.logical_and(side_filter, fwd_filter)

    x_points = x_points[point_filter]
    y_points = y_points[point_filter]
    z_points = z_points[point_filter]
    if intensities is not None:
        intensities = intensities[point_filter]

    # Convert to pixel positions
    x_img = (-y_points / resolution).astype(np.int32)
    y_img = (-x_points / resolution).astype(np.int32)

    # Flip horizontally by negating the x_img values
    x_img = -x_img

    x_img -= int(np.floor(side_range[0] / resolution))
    y_img -= int(np.floor(fwd_range[0] / resolution))

    img_width = int((side_range[1] - side_range[0]) / resolution)
    img_height = int((fwd_range[1] - fwd_range[0]) / resolution)
    bev_image = np.zeros((img_height, img_width), dtype=np.uint8)

    max_height = np.max(z_points) if z_points.size > 0 else 1
    min_height = np.min(z_points) if z_points.size > 0 else 0
    height_range = max_height - min_height if max_height != min_height else 1
    pixel_values = ((z_points - min_height) / height_range * 255).astype(np.uint8)

    bev_image[y_img, x_img] = pixel_values

    if enhance_contrast:
        bev_image = cv2.equalizeHist(bev_image)

    return bev_image

def crop_center(img, target_width, target_height):
    """
    Crop the image horizontally to the target width, keeping the center.
    """
    original_width = img.shape[1]
    if original_width <= target_width:
        return img
    
    original_height = img.shape[0]  
    if original_height <= target_height:
        return img

    # Calculate the cropping box to center the image
    left = (original_width - target_width) // 2
    right = left + target_width
    img = img[:, left:right]

    top = (original_height - target_height) // 2
    bottom = top + target_height
    img = img[top:bottom, :]
    return img

def run_cosimulation_dashboard(vehicle_id: str):
    bev_path = "visualization_images"
    recording_path = "Co-simulation/recordings/KIVI_visualization"
    output_dir = "dashboard_frames"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    bev_files = os.listdir(bev_path)
    sumo_bev_files = [f for f in bev_files if f.startswith("sumo_view")]
    timesteps = sorted([int(f.split(".")[0].split("_")[-1]) for f in sumo_bev_files])

    for timestep in timesteps:
        sumo_file = os.path.join(bev_path, f"sumo_view_{timestep}.png")
        carla_file = os.path.join(bev_path, f"carla_view_{timestep}.png")
        recordings_dir = os.path.join(recording_path, f"{timestep}_0", vehicle_id)
        lidar_point_cloud_path = os.path.join(recordings_dir, "lidar_points.pcd")
        front_camera = os.path.join(recordings_dir, "front", "bounding_box_image.png")

        if not (os.path.exists(sumo_file) and os.path.exists(carla_file) and os.path.exists(lidar_point_cloud_path) and os.path.exists(front_camera)):
            continue

        sumo_img = plt.imread(sumo_file)
        carla_img = plt.imread(carla_file)
        front_camera_img = plt.imread(front_camera)
        pcd = read_pcd_file(lidar_point_cloud_path)
        lidar_bev_img = generate_bev_image(pcd)

        # Crop the SUMO image to match the CARLA image width
        target_width = carla_img.shape[1]
        target_height = carla_img.shape[0]  
        sumo_img = crop_center(sumo_img, target_width, target_height)

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        axes[0, 0].imshow(sumo_img)
        axes[0, 0].set_title("SUMO View")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(carla_img)
        axes[0, 1].set_title("CARLA View")
        axes[0, 1].axis("off")

        axes[1, 0].imshow(front_camera_img)
        axes[1, 0].set_title("Front Camera View")
        axes[1, 0].axis("off")

        axes[1, 1].imshow(lidar_bev_img, cmap="gray")
        axes[1, 1].set_title("Lidar BEV Image")
        axes[1, 1].axis("off")

        output_path = os.path.join(output_dir, f"frame_{timestep}.png")
        plt.savefig(output_path)
        plt.close(fig)

        print(f"Saved frame for timestep {timestep} to {output_path}")

    create_video_from_images(output_dir, output_video="cosimulation_dashboard_video.mp4", fps=1)

def create_video_from_images(image_dir="dashboard_frames", output_video="cosimulation_dashboard_video.mp4", fps=1):
    images = sorted([img for img in os.listdir(image_dir) if img.endswith(".png")])
    if not images:
        print("No images found to create a video.")
        return

    frame = cv2.imread(os.path.join(image_dir, images[0]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_dir, image)))

    cv2.destroyAllWindows()
    video.release()
    print(f"Video saved as {output_video}")

if __name__ == "__main__":
    run_cosimulation_dashboard(vehicle_id="pv_6_166_0")