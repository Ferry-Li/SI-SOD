import os
import numpy as np
import cv2
from skimage import measure
from tqdm import tqdm
from utils import binarize, apply_colormap, create_colormap, image_filter
import math
import torch

def binary(label):
    # convert a number to binary system
    if label == 0:
        return 0
    else:
        return math.pow(2, label - 1)




'''
find the connected components of a (0, 1) mask
return: number of connected components, sizes of connected components, the labeled mask with each connected component a label
'''
def find_connection(mask, mask_path, epsilon):

    label_mask = np.zeros_like(mask)
    size_list = []
    labeled_mask, _ = measure.label(mask, connectivity=1, background=0, return_num=True)
    properties = measure.regionprops(labeled_mask)

    num_features = 0
    for prop in properties:
        clean_mask = np.zeros_like(mask)
        # remove noise in the mask
        if prop.area >= epsilon:
            num_features += 1
            for coords in prop.coords:
                y, x = coords
                clean_mask[y, x] = num_features
            label_mask = label_mask + clean_mask
            size_list.append(prop.area)

    if num_features == 0:
        num_features = 1
        clean_mask = np.zeros_like(mask)

        if len(properties) == 0:
            return None, None, None
        for coords in properties[0].coords:
            y, x = coords
            clean_mask[y, x] = num_features
        return num_features, [properties[0].area], clean_mask


    return num_features, size_list, label_mask


def compute_weight(labeled_mask):

    def set_value(origin_value_list, target_value):
        zero_idx = (origin_value_list == 0)
        non_zero_idx = (origin_value_list > 0)
        origin_value_list[zero_idx] = target_value

        target_value_list = target_value * np.ones_like(origin_value_list[non_zero_idx])
        origin_value_list[non_zero_idx] = np.max(np.array((target_value_list, origin_value_list[non_zero_idx])), axis=0)

        return origin_value_list


        
    weight_graph = np.zeros_like(labeled_mask).astype(float)
    step_list = np.unique(labeled_mask)
    h, w = labeled_mask.shape
    
    num_features = int(len(step_list))
    for label in step_list:
        connected_block = (labeled_mask == label).astype(int)
        connected_block_size = np.sum(connected_block > 0)
        rows, cols = np.where(connected_block > 0)
        assert len(rows) * len(cols) > 0, "connected_block_size = 0 !"

        xmin, xmax = min(cols), max(cols)
        ymin, ymax = min(rows), max(rows)
        frame_size = (xmax - xmin + 1) * (ymax - ymin + 1)

        # mark the area of a object frame with binary(label)
        weight_graph[ymin:(ymax+1), xmin:(xmax+1)] += binary(label)

    return weight_graph

# compute connected components in a mask
def generate_connection(root_dir, size, epsilon):
    h = w = size
    mask_dir = os.path.join(root_dir, 'mask')
    connection_dir = os.path.join(root_dir, 'connection')
    if not os.path.exists(connection_dir):
        os.mkdir(connection_dir)
    for mask in tqdm(os.listdir(mask_dir), desc="Generating connection ..."):
        mask_path = os.path.join(mask_dir, mask)
        mask_gt = cv2.imread(mask_path)
        mask_gt = binarize(mask_gt[..., 0], 255 / 2)
        num_components, sizes, labeled_mask = find_connection(mask_gt, mask_path, epsilon)
        mask_name = mask.rsplit('.')[0]
        np.save(os.path.join(connection_dir, mask_name + ".npy"), labeled_mask)


def generate_single_connection(root_dir, size, epsilon):
    h = w = size
    mask = 'sun_azranmirjvukkycc.png'
    mask_dir = os.path.join(root_dir, 'mask')
    connection_dir = os.path.join(root_dir, 'connection')
    mask_path = os.path.join(mask_dir, mask)
    mask_gt = cv2.imread(mask_path)
    mask_gt = binarize(mask_gt[..., 0], 255 / 2)
    num_components, sizes, labeled_mask = find_connection(mask_gt, mask_path, epsilon)
    mask_name = mask.rsplit('.')[0]
    np.save(os.path.join(connection_dir, mask_name + ".npy"), labeled_mask)

            
# generate weight for masks in the dir
def generate_weight(root_dir, size=384, visualize=False):
    h = w = size
    labeled_mask_dir = os.path.join(root_dir, "connection")
    weight_graph_dir = os.path.join(root_dir, "weight")
    if not os.path.exists(weight_graph_dir):
        os.mkdir(weight_graph_dir)
    labeled_mask_list = os.listdir(labeled_mask_dir)
    if visualize:
        weight_visualize_dir = os.path.join(root_dir, "weight_visualize")
        if not os.path.exists(weight_visualize_dir):
            os.mkdir(weight_visualize_dir)

    for item in tqdm(labeled_mask_list, desc="Generating weight ..."):

        labeled_mask = np.load(os.path.join(labeled_mask_dir, item))
        labeled_mask = cv2.resize(labeled_mask, (h, w), interpolation=cv2.INTER_NEAREST)

        weight_graph = compute_weight(labeled_mask)
        weight_name = item.rsplit('.')[0]
        np.save(os.path.join(weight_graph_dir, weight_name + ".npy"), weight_graph)
        # visualize the generated weight file
        if visualize:
            num_components = len(np.unique(weight_graph))
            assert num_components > 0, f"cannot visualize file: {weight_name}.npy"
            colormap = create_colormap(num_components)
            colored_image = apply_colormap(weight_graph, colormap)
            cv2.imwrite(os.path.join(weight_visualize_dir, weight_name + ".png"), colored_image)
