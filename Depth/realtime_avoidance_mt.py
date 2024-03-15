import re

import numpy as np
from metavision_core.event_io import EventsIterator
import time
import torch
from utils.loading_utils import load_model, get_device
from utils.event_readers import VoxelGridDataset
from os.path import join, basename
import json
import argparse
import shutil
import os
from depth_prediction import DepthEstimator
from options.inference_options import set_depth_inference_options
import cv2
from sklearn.cluster import DBSCAN
from shapely.geometry import MultiPoint, Polygon
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
from sklearn.metrics import silhouette_score
import math
import metavision_hal as mv_hal
import threading



def downscale(events):
    # events = np.array(events)  # Convert list of events to a NumPy array
    events = np.array([list(event) for event in events])

    mask = (events[:,0] % 4 == 0) & (events[:,1] % 4 == 0)  # Create a mask for the rows where both x and y are divisible by 4

    scaled_events = events[mask].copy()  # Select only those rows
    scaled_events[:, 0] = scaled_events[:, 0] / 4 + 3  # Scale x
    scaled_events[:, 1] = scaled_events[:, 1] / 4 + 40  # Scale y
    scaled_events = scaled_events.astype(np.int32)  # Convert to int

    time_values = np.unique(scaled_events[:, 3])  # Get unique time values
    time_values = time_values[np.where(np.diff(time_values) > 0)[0] + 1]  # Get increasing time values

    return scaled_events, time_values


def fill_polygon(polygon):
    # Calculate the bounding box of the polygon
    min_x, min_y = np.min(polygon, axis=0)
    max_x, max_y = np.max(polygon, axis=0)

    interior_points = []

    # Iterate through all points within the bounding box
    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):
            point = np.array([x, y], dtype=np.int32)

            # Check if the point is inside the polygon
            if cv2.pointPolygonTest(polygon, (x, y), False) >= 0:
                interior_points.append(point)

    return interior_points


def events_to_voxel_grid(events, num_bins, width, height):
    """
    Build a voxel grid with bilinear interpolation in the time domain from a set of events.
    :param events: a [N x 4] NumPy array containing one event per row in the form: [x, y, polarity, timestamp]
    :param num_bins: number of bins in the temporal axis of the voxel grid
    :param width, height: dimensions of the voxel grid
    """

    assert (events.shape[1] == 4)
    assert (num_bins > 0)
    assert (width > 0)
    assert (height > 0)


    voxel_grid = np.zeros((num_bins, height, width), np.float32).ravel()

    # normalize the event timestamps so that they lie between 0 and num_bins
    last_stamp = events[-1, 3]
    first_stamp = events[0, 3]
    deltaT = last_stamp - first_stamp
    print(deltaT)

    if deltaT == 0:
        deltaT = 1.0

    ts = (num_bins - 1) * (events[:, 3] - first_stamp) / deltaT
    # print(ts)
    xs = events[:, 0].astype(np.int32)
    ys = events[:, 1].astype(np.int32)
    pols = events[:, 2]
    pols[pols == 0] = -1  # polarity should be +1 / -1

    tis = ts.astype(np.int32)
    dts = ts - tis
    vals_left = pols * (1.0 - dts)
    vals_right = pols * dts

    valid_indices = tis < num_bins
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width +
              tis[valid_indices] * width * height, vals_left[valid_indices])


    valid_indices = (tis + 1) < num_bins
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width +
              (tis[valid_indices] + 1) * width * height, vals_right[valid_indices])


    voxel_grid = np.reshape(voxel_grid, (num_bins, height, width))

    return voxel_grid, first_stamp, last_stamp


def store_ev(mv):

    for ev_idx, ev in enumerate(mv):
        np.save("../test/events/" + 'event_iterator_' + "{:010}".format(
            ev_idx), ev)




raw_path = mv_hal.DeviceDiscovery.open('')
print(raw_path)
mv_it = EventsIterator(raw_path, delta_t=50000)
height, width = mv_it.get_size()
i = 0

parser = argparse.ArgumentParser(
    description='Evaluating a trained network')

parser.add_argument('-c', '--path_to_model', default="./E2DEPTH_si_grad_loss_mixed.pth.tar",
                    type=str,
                    help='path to model weights')
parser.add_argument('-i', '--input_folder', default="../test/voxels",
                    type=str,
                    help="name of the folder containing the voxel grids")
parser.add_argument('--start_time', default=0.0, type=float)
parser.add_argument('--stop_time', default=0.0, type=float)
# parser.add_argument('--save_numpy', default=True)

set_depth_inference_options(parser)

args = parser.parse_args()
print(torch.cuda.is_available())

model = load_model(args.path_to_model)
device = get_device(args.use_gpu)
model = model.to(device)
# model = torch.load(args.path_to_model, map_location="cpu")
estimator = DepthEstimator(model, 260, 326, model.num_bins, args)
model.eval()

index = 0
colors_list = [(0, 0, 255),  # Red
               (0, 255, 0),  # Green
               (255, 0, 0),
               (100, 100, 100),
               (200, 200, 200),
               (50, 50, 50),
               (150, 150, 150),
               (20, 80, 20)]

Dmax = 7.9
alpha = 3.70378
epsilon = 1

print_every_n = 50

cell = int(326/5)
cell_1 = 65
cell_2 = 130
cell_3 = 195
cell_4 = 260

def main_loop():
    with open('../test/voxels/timestamps.txt', 'w') as file:
        with open(
                '../test/voxels/boundary_timestamps.txt',
                'w') as boundary_file:
            index_mv = 0
            while True:
                files = [f for f in os.listdir('../test/events/') if f.startswith("event_iterator_")]
                numbers = [int(re.findall(r'\d+', f)[0]) for f in files]
                if len(files) == 0:
                    time.sleep(0.05)
                    continue
                else:
                    max_number = max(numbers)
                    max_filename = f'event_iterator_{str(max_number).zfill(10)}.npy'
                    print(max_filename)
                    with open(os.path.join('../test/events', max_filename), 'rb') as current_file:
                        ev = np.load(current_file)
                events = downscale(ev)[0]
                timestamps = downscale(ev)[1]

                try:
                    voxels, first_stamp, last_stamp = events_to_voxel_grid(events, 5, 326, 260)
                except IndexError:
                    continue

                np.save(
                    "../test/voxels/" + 'event_tensor_' + "{:010}".format(
                        index_mv), voxels)
                for index, timestamp in enumerate(timestamps):
                    timestamp = float(timestamp / 1000000)
                    file.write(f"{index_mv + index} {timestamp}\n")

                base_folder = os.path.dirname(args.input_folder)
                event_folder = os.path.basename(args.input_folder)
                try:
                    dummy_dataset = VoxelGridDataset(base_folder,
                                                     event_folder,
                                                     args.start_time,
                                                     args.stop_time,
                                                     transform=None)
                except OSError and AssertionError:
                    time.sleep(0.05)
                    continue
                data = dummy_dataset[0]
                _, height_1, width_1 = data['events'].shape
                height_1 = height_1 - args.low_border_crop


                events_dataset = VoxelGridDataset(base_folder=base_folder,
                                                  event_folder=event_folder,
                                                  start_time=args.start_time,
                                                  stop_time=args.stop_time,
                                                  transform=None)

                output_dir = args.output_folder
                dataset_name = args.dataset_name
                # print(events_dataset)
                data = events_dataset[0]
                event_tensor = data['events'][:, :height_1, :]
                out = estimator.update_reconstruction(event_tensor, 0)
                index_mv += 1

                img = np.zeros((260, 326, 3), dtype=np.uint8)
                xy = np.column_stack((events[:, 0], events[:, 1]))
                print(max(events[:, 0]))

                depth_log = out

                K = 5
                nbrs = NearestNeighbors(n_neighbors=K, algorithm='ball_tree').fit(xy)
                distances, indices = nbrs.kneighbors(xy)
                kth_distances = distances[:, -1]
                sorted_distances = np.sort(kth_distances)

                x = np.arange(len(sorted_distances))

                # Use the KneeLocator to find the elbow point
                knee_locator = KneeLocator(x, sorted_distances, curve="convex", direction="increasing")
                elbow_index = knee_locator.knee

                # Get the corresponding KNN distance as an estimate for `eps`
                eps_estimate = int(sorted_distances[elbow_index])

                eps_value = 5  # Adjust this value based on your specific use case
                min_samples_value = 40  # Adjust this value based on your specific use case

                dbscan = DBSCAN(eps=eps_estimate, min_samples=min_samples_value)

                clusters = dbscan.fit_predict(xy)

                unique_clusters = np.unique(clusters)

                unique_clusters = unique_clusters[unique_clusters != -1]
                colors = [tuple(map(int, np.random.randint(0, 255, 3))) for _ in range(len(unique_clusters))]

                depth = Dmax * np.exp(-alpha * (1 - depth_log))
                depth_sum = {cluster: 0.0 for cluster in unique_clusters}
                points_count = {cluster: 0 for cluster in unique_clusters}

                cluster_events = {cluster: [] for cluster in unique_clusters}
                mean_optical_flow = {cluster: [] for cluster in unique_clusters}

                # Iterate through the depth_map and clusters_reshaped and accumulate depth values and points count
                for i, (x, y) in enumerate(xy):
                    cluster_label = clusters[i]
                    if cluster_label != -1:
                        # Use the event coordinates to index into the depth map and assign colors to each cluster
                        color = colors_list[cluster_label % len(colors_list)]  # Use modulo to avoid index out of range
                        img[int(y), int(x)] = color
                        cluster_events[cluster_label].append([x, y])

                        points_count[cluster_label] += 1

                for j in range(len(unique_clusters)):
                    hull = MultiPoint(cluster_events[j]).convex_hull
                    cluster_items = np.array(cluster_events[j])
                    min_x, min_y = np.min(cluster_items[:, 0]), np.min(cluster_items[:, 1])
                    max_x, max_y = np.max(cluster_items[:, 0]), np.max(cluster_items[:, 1])
                    # print(min_x, max_x, min_y, max_y)
                    centroid = hull.centroid
                    centroid_x, centroid_y = int(centroid.x), int(centroid.y)
                    # print("cluster" + str(j), [centroid_x, centroid_y])
                    depth_value = depth[centroid_y, centroid_x]
                    # print(depth_value)
                    depth_value_1 = depth[centroid_y, centroid_x - 1]
                    depth_value_2 = depth[centroid_y, centroid_x + 1]
                    depth_value_3 = depth[centroid_y - 1, centroid_x]
                    depth_value_4 = depth[centroid_y + 1, centroid_x]
                    depth_value_5 = depth[centroid_y - 1, centroid_x - 1]
                    depth_value_6 = depth[centroid_y + 1, centroid_x + 1]
                    depth_value_7 = depth[centroid_y - 1, centroid_x + 1]
                    depth_value_8 = depth[centroid_y + 1, centroid_x - 1]
                    mean_depth_value = (depth_value + depth_value_1 + depth_value_2 + depth_value_3 + depth_value_4 + depth_value_5 + depth_value_6 + depth_value_7 + depth_value_8) / 9

                    label = f"Mean Depth: {mean_depth_value:.2f}"

                    cv2.rectangle(img, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
                    cv2.putText(img, label, (centroid_x, centroid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.putText(img, "obstacle", (max_x, min_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    if mean_depth_value < 2:
                        min_bound = np.min(cluster_events[j], axis=0)
                        max_bound = np.max(cluster_events[j], axis=0)

                        if max_bound[0] - min_bound[0] > 195 and max_bound[0] > cell_4:
                            print("Object too big, not applicable")
                        else:
                            if min_bound[0] >= cell_1:
                                cv2.arrowedLine(img, (163, 250), (32, 130), (0, 255, 0), 2)
                            elif max_bound[0] <= cell_4:
                                cv2.arrowedLine(img, (163, 250), (292, 130), (0, 255, 0), 2)

                    else:
                        pass

                cv2.imshow("Clustered Image", img)
                key = cv2.waitKey(1) & 0xFF
                index += 1
                if key == ord('q'):
                    break
                boundary_file.write(f"{index_mv} {first_stamp / 1000000} {last_stamp / 1000000}\n")
                i += 1

            time.sleep(0.05)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


# Create two threads.
thread1 = threading.Thread(target=store_ev, args=(mv_it,))
thread2 = threading.Thread(target=main_loop,)

# Start the threads.
thread1.start()
thread2.start()

# Wait for both threads to finish.
thread1.join()
thread2.join()

print("Both threads are finished!")
