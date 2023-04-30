import json
import torch
from PIL import Image
import numpy as np


def get_data_from_index(data, index):
    camera_angle_x = data["camera_angle_x"]
    frame = data["frames"][index]

    file_path = frame["file_path"]
    rotation = frame["rotation"]
    transform_matrix = torch.tensor(frame["transform_matrix"])

    return transform_matrix, rotation, file_path, camera_angle_x


def load_data(data):
    transform_matricies = []
    file_paths = []
    camera_angle_x = 0.0
    for i in range(len(data["frames"])):
        transform_matrix, rotation, file_path, camera_angle_x = get_data_from_index(data, i)
        transform_matricies.append(transform_matrix.unsqueeze(0))
        file_paths.append(file_path)

    transform_matricies = torch.cat(transform_matricies, 0)

    return transform_matricies, file_paths, camera_angle_x


def read_data(data_path):
    with open(data_path, "r") as f:
        data = json.load(f)
    return data


def load_image_data(data_folder, object_folder, split="train"):
    with open(f"{data_folder}/{object_folder}/transforms_{split}.json", "r") as f:
        data = json.load(f)

    imgs = [Image.open(f'{data_folder}/{object_folder}/{split}/{frame["file_path"].split("/")[-1]}.png') for frame in
            data["frames"]]
    imgs = np.array([np.array(img) for img in imgs])
    imgs = torch.tensor(imgs, dtype=torch.float)
    imgs = (imgs / 255)

    return data, imgs


def load_image_data_from_path(path, transformpath):
    with open(transformpath, "r") as f:
        data = json.load(f)

    imgs = [Image.open(f'{path}/{frame["file_path"].split("/")[-1]}.png') for frame in data["frames"]]
    imgs = np.array([np.array(img) for img in imgs])
    imgs = torch.tensor(imgs, dtype=torch.float)
    imgs = (imgs / 255)

    return data, imgs
