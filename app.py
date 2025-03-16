
import matplotlib
from flask import Flask, Response
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import io
import requests
import matplotlib
from scipy.ndimage import maximum_filter

matplotlib.use('agg')

app = Flask(__name__)

def fetch_tile(api_url):
    response = requests.get(api_url)
    return response.json()["message"]["data"]

import numpy as np

def assemble_map(api_url):
    height_map = np.zeros((256, 256), dtype=int)
    
    tile_indices = [(i, j) for i in range(4) for j in range(4)]

    for i, j in tile_indices:
        if (i, j) == (1, 1): 
            
            tile = np.full((64, 64), 255, dtype=int)
        else:
            
            tile = fetch_tile(api_url)
            tile = np.array(tile).reshape(64, 64)
        
        height_map[i * 64:(i + 1) * 64, j * 64:(j + 1) * 64] = tile

    return height_map


def find_peaks(height_map, size=32):

    local_max = maximum_filter(height_map, size=size) == height_map
    peaks = np.column_stack(np.where(local_max))
    return peaks

def place_stations(peaks, height_map, station_type_cooper_radius=32, station_type_engel_radius=64):
    stations = []
    used_peaks = set()

    for peak in peaks:
        if tuple(peak) not in used_peaks:
            neighbors = [p for p in peaks if np.linalg.norm(p - peak) <= station_type_engel_radius]
            if len(neighbors) > 1:
                stations.append({"type": "Энгель", "position": (peak[0], peak[1]), "radius": station_type_engel_radius})
                used_peaks.update(tuple(p) for p in neighbors)
            else:
                stations.append({"type": "Купер", "position": (peak[0], peak[1]), "radius": station_type_cooper_radius})
                used_peaks.add(tuple(peak))

    return stations


def plot_3d_map(height_map, modules=None, stations=None, show_coverage=False):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    x = np.arange(height_map.shape[1])
    y = np.arange(height_map.shape[0])
    x, y = np.meshgrid(x, y)
    z = height_map
    ax.plot_surface(x, y, z, cmap='terrain', alpha=0.8)
    if modules:
        for module in modules:
            ax.scatter(module[1], module[0], height_map[module[0], module[1]],
                       c='green', marker='x', s=100, label='Модуль')

    if stations:
        for station in stations:
            color = 'blue' if station['type'] == 'Купер' else 'red'
            ax.scatter(station['position'][1], station['position'][0],
                       height_map[station['position'][0], station['position'][1]],
                       c=color, s=100, label=station['type'])
            if show_coverage:
                u = np.linspace(0, 2 * np.pi, 100)
                v = np.linspace(0, np.pi, 100)
                x_sphere = station['position'][1] + station['radius'] * np.outer(np.cos(u), np.sin(v))
                y_sphere = station['position'][0] + station['radius'] * np.outer(np.sin(u), np.sin(v))
                z_sphere = height_map[station['position'][0], station['position'][1]] + station['radius'] * np.outer(
                    np.ones(np.size(u)), np.cos(v))
                ax.plot_surface(x_sphere, y_sphere, z_sphere, color=color, alpha=0.2)

    ax.set_xlabel('X (пиксели)')
    ax.set_ylabel('Y (пиксели)')
    ax.set_zlabel('Высота')
    ax.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    plt.close()
    return buf

@app.route('/map', methods=['GET'])
def get_map():
    buf = plot_3d_map(height_map)
    return Response(buf.getvalue(), mimetype='image/png')

@app.route('/map/modules', methods=['GET'])
def get_map_with_modules():
    buf = plot_3d_map(height_map, modules=modules)
    return Response(buf.getvalue(), mimetype='image/png')

@app.route('/map/stations', methods=['GET'])
def get_map_with_stations():
    buf = plot_3d_map(height_map, stations=stations)
    return Response(buf.getvalue(), mimetype='image/png')

@app.route('/map/stations/coverage', methods=['GET'])
def get_map_with_stations_coverage():
    buf = plot_3d_map(height_map, stations=stations, show_coverage=True)
    return Response(buf.getvalue(), mimetype='image/png')

if __name__ == '__main__':
    api_url = "https://olimp.miet.ru/ppo_it/api"

    height_map = assemble_map(api_url)

    peaks = find_peaks(height_map)
    stations = place_stations(peaks, height_map)

    app.run(host="0.0.0.0", port=8080, debug=True)
