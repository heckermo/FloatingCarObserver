import random
import xml.etree.ElementTree as ET
import os

INCOMING_EDGES = {'KIVI': {'incoming_edges': ['-64#0', '-20#0', '-2#3', '71', '-1', '-59#0', '-67#0'],
                           'outgoing_edges': ['-58', '59#0', '66', '-71', '1', '-12#2', '-55']},
                }

def parse_network_edges(net_file):
    tree = ET.parse(net_file)
    root = tree.getroot()
    edges = []

    for edge in root.findall('edge'):
        if edge.get('function') is None:
            edges.append(edge.get('id'))
    
    return edges, edges

def generate_random_trips(incoming_edges_list, outgoing_edges_list: list, num_trips: int=100):
    trips = []
    for _ in range(num_trips):
        start_edge = random.choice(incoming_edges_list)
        end_edge = random.choice(outgoing_edges_list)
        while start_edge == end_edge:
            end_edge = random.choice(outgoing_edges_list)
        trips.append((start_edge, end_edge))
    return trips

def write_trips_file(trips, output_file, start_time=0, end_time=1000, head_start_vehicles=100):
    # generate random depart times
    depart_times = [random.uniform(start_time, end_time) for _ in range(len(trips)-head_start_vehicles)]
    depart_times.sort()
    with open(output_file, 'w') as f:
        f.write('<routes>\n')
        for i in range(head_start_vehicles):
            start_edge, end_edge = trips[i]
            f.write(f'    <trip id="{i}" from="{start_edge}" to="{end_edge}" depart="0" />\n')
        for i in range(len(trips)-head_start_vehicles):
            start_edge, end_edge = trips[i+head_start_vehicles]
            f.write(f'    <trip id="{i+head_start_vehicles}" from="{start_edge}" to="{end_edge}" depart="{depart_times[i]}" />\n')
        f.write('</routes>\n')

def create_route_from_trips(trips_file, route_file, net_file):
    os.system(f'duarouter --net-file {net_file} --route-files {trips_file} --output-file {route_file}')


def edit_vclasses(route_file, vclass_list):
    tree = ET.parse(route_file)
    root = tree.getroot()
    for vehicle in root.findall('vehicle'):
        vehicle.set('type', random.choice(vclass_list))
    tree.write(route_file)


if __name__ == '__main__':
    town = 'Town05'
    vclass_list = ['vehicle.ford.mustang']
    if town not in INCOMING_EDGES:
        print(f'No incoming edges defined for town {town}, taking all edges from net file')
        incoming_edges, outgoing_edges = parse_network_edges(f'Co-simulation/sumo_files/net/{town}.net.xml')
    else:
        incoming_edges = INCOMING_EDGES[town]['incoming_edges']
        outgoing_edges = INCOMING_EDGES[town]['outgoing_edges']
    trips = generate_random_trips(incoming_edges, outgoing_edges, 150)
    write_trips_file(trips, f'Co-simulation/sumo_files/rou/{town}.trips.xml', 0, 20)
    create_route_from_trips(f'Co-simulation/sumo_files/rou/{town}.trips.xml', f'Co-simulation/sumo_files/rou/{town}_own.rou.xml', f'Co-simulation/sumo_files/net/{town}.net.xml')
    edit_vclasses(f'Co-simulation/sumo_files/rou/{town}_own.rou.xml', vclass_list)
    #write_route_file(trips, vclass_list, 'Co-simulation/sumo_files/rou/{town}_own.rou.xml', 0, 100)

    # start the simulation to check if the route file is working (without sumo gui)
    cfg_file = f'Co-simulation/sumo_files/{town}.sumocfg'
    sumo_cmd = f'sumo -c {cfg_file}'
    sumo_gui_cmd = f'sumo-gui -c {cfg_file}'
    os.system(sumo_cmd)
