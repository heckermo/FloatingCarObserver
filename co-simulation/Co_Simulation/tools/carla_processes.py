import subprocess
import psutil
import os
import carla
import time
import logging
from Co_Simulation.tools.mappings import weather_mappings

def is_process_running(process_name):
    for proc in psutil.process_iter(['name', 'status']):
        try:
            if process_name.lower() in proc.info['name'].lower() and proc.info['status'] == psutil.STATUS_RUNNING:
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return False

def start_carla_server(carla_root, screen, carla_server_port=2000):
    # Expand the full path, resolving '~' to the home directory
    full_carla_root = os.path.expanduser(carla_root)

    if is_process_running('CarlaUE4-Linux-Shipping'):
        print('Carla is already running')
        return
    
    # Form the complete command
    if screen:
        command = f'{full_carla_root}/CarlaUE4.sh -carla-port={carla_server_port}'
    else:
        command = f'{full_carla_root}/CarlaUE4.sh -RenderOffScreen -carla-port={carla_server_port}'
    
    # Run the command, note that we don't prepend 'bash' here since we're executing a script directly
    carla_process = subprocess.Popen(command, shell=True)#, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    return carla_process

def stop_carla_server(process_name):
    if is_process_running(process_name):
        for proc in psutil.process_iter(['name', 'status']):
            try:
                if process_name in proc.info['name'] and proc.info['status'] == psutil.STATUS_RUNNING:
                    proc.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass

def adapt_carla_map(map_name: str, weather: str, carla_port=3000, sleep_time=3):
    carla_client = carla.Client('localhost', carla_port)
    carla_client.set_timeout(10.0)

    # Set the chosen carla map
    availabel_maps = [map_name.split('/')[-1] for map_name in carla_client.get_available_maps()]
    assert map_name in availabel_maps, f'Map {map_name} not available. Available maps are: {availabel_maps}'
    world = carla_client.load_world(map_name)

    # Set the weather
    assert weather in weather_mappings, f'Weather {weather} not supported'
    weather = weather_mappings[weather]
    world.set_weather(weather)
    logger.info(f'Loaded map: {world.get_map().name} with weather {weather}, will sleep for {sleep_time} seconds to allow the map to load')

    # reset the simulation to synchronous mode
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    settings.max_substeps = 5
    settings.max_substep_delta_time = 0.01
    world.apply_settings(settings)

if __name__ == '__main__':
    process_name = 'CarlaUE4-Linux-Shipping'
    stop_carla_server(process_name)
    carla_root = '~/CARLA_Shipping_0.9.14_KIVI/LinuxNoEditor'
    start_carla_server(carla_root, True)
    time.sleep(10)
    adapt_carla_map('Town01')
    # Small delay to allow the process to start
    time.sleep(5)  # Adjust as necessary
    if is_process_running(process_name):
        print(f'{process_name} is running')
    else:
        print(f'{process_name} is not running')
    time.sleep(10)
    stop_carla_server(process_name)
    if is_process_running(process_name):
        print(f'{process_name} is running')
    else:
        print(f'{process_name} is not running')
else:
    logger = logging.getLogger(__name__)