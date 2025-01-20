import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

"""
This script analyzes and visualizes traffic demand over a 24-hour simulation period
using vehicle departure times extracted from a SUMO route file.

The script performs the following tasks:
1. Parses an XML route file to extract vehicle departure times.
2. Sorts the departure times and calculates the number of vehicles departing at each second.
3. Computes rolling window counts of vehicle departures over hourly intervals.
4. Aggregates the rolling window counts into hourly data.
5. Visualizes the traffic demand over the 24-hour simulation period using a line plot.

The resulting plot is saved as 'traffic_demand.png'.
"""

# Path to your route file
route_file = "motorized_routes_2020-09-16_24h_dist.rou.xml"

# Parse the XML
tree = ET.parse(os.path.join('sumo_sim', route_file))
root = tree.getroot()

# Extract departure times
departure_times = []
for vehicle in root.findall('vehicle'):
    depart = float(vehicle.get('depart'))
    departure_times.append(depart)

# Sort the departure times
departure_times.sort()

# Define simulation length in seconds (24 hours = 86400 seconds)
T = 24 * 3600
window_length = 3600

# Initialize an array to hold counts of how many vehicles departed at each second
counts = [0] * (T + 1)

# Populate the counts array
for d in departure_times:
    sec = int(d)
    if 0 <= sec <= T:
        counts[sec] += 1

# Compute cumulative sums to quickly calculate rolling windows
cumulative = [0] * (T + 1)
cumulative[0] = counts[0]
for i in range(1, T + 1):
    cumulative[i] = cumulative[i - 1] + counts[i]

# Compute rolling window counts
rolling_window = [0] * (T + 1)
for t in range(window_length, T + 1):
    rolling_window[t] = cumulative[t] - cumulative[t - window_length]

# Aggregate hourly data
hourly_windows = {}
for hour in range(1, 25):
    t = hour * 3600
    hourly_windows[hour] = rolling_window[t]

# Visualization
plt.figure(figsize=(12, 6))

# Rolling window plot
rolling_x = range(window_length, T + 1)
rolling_y = [rolling_window[t] for t in rolling_x]
plt.plot([t / 3600 for t in rolling_x], rolling_y, label="Rolling Window (Hourly)", alpha=0.7)

# Labels and title
plt.xlabel("Time", fontsize=14)
plt.ylabel("Number of Vehicles in Simulation", fontsize=14)
plt.xticks(range(0, 25, 1))
plt.legend()
plt.grid(alpha=0.3)

# Show the plot
plt.tight_layout()
plt.savefig('traffic_demand.png', dpi=500)
plt.savefig('traffic_demand.pdf')
plt.savefig('traffic_demand.svg')