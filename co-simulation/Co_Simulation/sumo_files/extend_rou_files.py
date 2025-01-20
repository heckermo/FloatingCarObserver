import xml.etree.ElementTree as ET
from xml.dom import minidom

# Configuration parameters
copies_to_add = 5  # how many sets of current vehicles you want to add
time_increment = 35.0  # how much time to increment departure time for each repeated set

# Load the existing routes file
tree = ET.parse('rou/Town05_own.rou.xml')
root = tree.getroot()

# Extract IDs and departure times
all_vehicles = root.findall('vehicle')
existing_ids = [int(v.get('id')) for v in all_vehicles]
max_id = max(existing_ids)
depart_times = [float(v.get('depart')) for v in all_vehicles]
max_depart = max(depart_times)

# Duplicate vehicles
for copy_index in range(copies_to_add):
    for v in all_vehicles:
        old_id = int(v.get('id'))
        old_depart = float(v.get('depart'))

        # Create new vehicle element
        new_vehicle = ET.Element('vehicle')
        new_id = old_id + (copy_index + 1) * len(all_vehicles)
        new_vehicle.set('id', str(new_id))

        # Adjust departure time
        new_depart = old_depart + (copy_index + 1) * time_increment + max_depart
        new_vehicle.set('depart', f"{new_depart:.2f}")
        new_vehicle.set('type', v.get('type'))

        # Copy route
        old_route = v.find('route')
        new_route = ET.Element('route')
        new_route.set('edges', old_route.get('edges'))
        new_vehicle.append(new_route)

        root.append(new_vehicle)

# Convert ElementTree to string
xml_str = ET.tostring(root, encoding='utf-8')

# Pretty print using minidom
parsed = minidom.parseString(xml_str)
pretty_str = parsed.toprettyxml(indent="    ")

# Write to output file
with open('rou/Town05_own_extended.rou.xml', 'w', encoding='utf-8') as f:
    f.write(pretty_str)