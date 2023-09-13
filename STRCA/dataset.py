import logging
import random
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def to_tensor(x):
    return torch.from_numpy(x).type(torch.FloatTensor)


def filter_pod_faults(data, target):
    target = target[:, 16:, 6:]
    indices_to_keep = np.where(np.any(target != 0, axis=1))[0]
    data, target = data[indices_to_keep, :, 224:], target[indices_to_keep]
    indices = np.argmax(target, axis=2)
    target = np.where(
        target[np.arange(target.shape[0])[:, np.newaxis], np.arange(target.shape[1]), indices] == 0, 0, indices + 1
    )
    return data.astype(float), target.astype(int)


def filter_node_faults(data, target):
    target = target[:, :6, :6]
    indices_to_keep = np.where(np.any(target != 0, axis=1))[0]
    data, target = data[indices_to_keep, :, :144], target[indices_to_keep]
    indices = np.argmax(target, axis=2)
    target = np.where(
        target[np.arange(target.shape[0])[:, np.newaxis], np.arange(target.shape[1]), indices] == 0, 0, indices + 1
    )
    return data.astype(float), target.astype(int)


def drop_repeat(data, target):
    n_samples, n_steps, n_features = data.shape
    flattened_data = data.reshape(n_samples, -1)
    unique_indices = np.unique(flattened_data, axis=0, return_index=True)[1]
    data, target = data[unique_indices], target[unique_indices]
    return data.reshape(-1, n_steps, n_features), target


def sample_labeled_data(data, target, ratio, seed):
    y_fault_type = np.max(target, axis=1)
    data_lb, data_ulb, target_lb, target_ulb = train_test_split(
        data, target, stratify=y_fault_type, train_size=ratio, random_state=seed
    )
    unique_elements, counts = np.unique(target_lb, return_counts=True)
    for element, count in zip(unique_elements, counts):
        logger.debug(f"Number {element} appears {count} times.")
    return data_lb, target_lb, data_ulb, target_ulb


def transform_pod(data, target):
    n_samples, n_steps, n_features = data.shape
    pod_num = 40
    n_metric = n_features // pod_num
    data = data.reshape(n_samples, n_steps, pod_num, n_metric)
    data = data.transpose(0, 2, 1, 3)
    data = data.reshape(-1, n_steps, n_metric)
    target = target.flatten()
    return data, target


def transform_node(data, target):
    n_samples, n_steps, n_features = data.shape
    node_num = 6
    n_metric = n_features // node_num
    data = data.reshape(n_samples, n_steps, node_num, n_metric)
    data = data.transpose(0, 2, 1, 3)
    data = data.reshape(-1, n_steps, n_metric)
    target = target.flatten()
    return data, target


def transform_raw_data_for_train(data, target, ratio, seed, filter, transform):
    data, target = filter(data, target)
    data, target = drop_repeat(data, target)
    data_lb, target_lb, data_ulb, target_ulb = sample_labeled_data(data, target, ratio, seed)
    data_ulb, target_ulb = transform(data_ulb, target_ulb)
    data_lb, target_lb = transform(data_lb, target_lb)
    return data_lb, target_lb, data_ulb, target_ulb


def transform_raw_data(data, target, filter, transform):
    data, target = filter(data, target)
    data, target = transform(data, target)
    return data, target


def jitter(time_series, noise_level=0.001):
    noise = np.random.normal(0, noise_level, size=time_series.shape)
    noisy_time_series = time_series + noise
    return noisy_time_series


def scaling(time_series, sigma=0.01):
    return time_series * np.random.normal(1, sigma, size=time_series.shape)


def permutation(time_series, max_segments=3):
    segments = np.random.randint(1, max_segments)
    split_indices = np.random.randint(1, time_series.shape[0], size=segments)
    split_indices.sort()
    split_indices = np.concatenate([[0], split_indices, [time_series.shape[0]]])
    indices = list(range(len(split_indices) - 1))
    random.shuffle(indices)
    shuffled_time_series = np.concatenate(
        [time_series[split_indices[i] : split_indices[i + 1]] for i in indices], axis=0
    )
    return shuffled_time_series


def shuffle_time_series(time_series):
    split_index = np.random.randint(1, time_series.shape[0])
    return np.concatenate([time_series[split_index:], time_series[:split_index]], axis=0)


class TimeSeriesDataset(Dataset):
    def __init__(self, data, targets, transform=None) -> None:
        super().__init__()
        self.data = data
        self.targets = targets
        self.transform = transform

    def __getitem__(self, index):
        ts, target = self.data[index], self.targets[index]

        if self.transform:
            ts = self.transform(ts)

        if type(ts) is tuple:
            ts = tuple(map(to_tensor, ts))
        else:
            ts = to_tensor(ts)

        return ts, target

    def __len__(self):
        return len(self.data)


def get_aiops(file_path, entity_type, ratio, seed):
    with open(file_path, "rb") as f:
        dataset = pickle.load(f)
    data = dataset["data"]

    if entity_type == "pod":
        filter = filter_pod_faults
        transform = transform_pod
    elif entity_type == "node":
        filter = filter_node_faults
        transform = transform_node

    # set lb_data, ulb_data, eval_data, test_data
    lb_data, lb_target, ulb_data, ulb_target = transform_raw_data_for_train(
        data["x_metric_train"], data["y_train"], ratio, seed, filter, transform
    )
    eval_data, eval_target = transform_raw_data(data["x_metric_valid"], data["y_valid"], filter, transform)
    test_data, test_target = transform_raw_data(data["x_metric_test"], data["y_test"], filter, transform)

    return lb_data, lb_target, ulb_data, ulb_target, eval_data, eval_target, test_data, test_target
