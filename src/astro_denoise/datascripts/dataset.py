import copy

import cv2
import numpy as np
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(
        self, file_paths, transform=None, ratio=0.9, size_data=(256, 256, 3), size_window=(5, 5)
    ):
        self.file_paths = file_paths
        self.transform = transform

        self.ratio = ratio
        self.size_data = size_data
        self.size_window = size_window

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        print(file_path)

        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)["image"]

        label = image
        input, mask = self.generate_mask(copy.deepcopy(label))
        data = {"image": label, "input": input, "masked": mask}

        return data

    def generate_mask(self, input):
        ratio = self.ratio
        size_window = self.size_window
        size_data = input.shape
        num_sample = int(size_data[0] * size_data[1] * (1 - ratio))

        mask = np.ones(size_data)
        output = input

        for ich in range(size_data[2]):
            idy_msk = np.random.randint(0, size_data[0], num_sample)
            idx_msk = np.random.randint(0, size_data[1], num_sample)

            idy_neigh = np.random.randint(
                -size_window[0] // 2 + size_window[0] % 2,
                size_window[0] // 2 + size_window[0] % 2,
                num_sample,
            )
            idx_neigh = np.random.randint(
                -size_window[1] // 2 + size_window[1] % 2,
                size_window[1] // 2 + size_window[1] % 2,
                num_sample,
            )

            idy_msk_neigh = idy_msk + idy_neigh
            idx_msk_neigh = idx_msk + idx_neigh

            idy_msk_neigh = (
                idy_msk_neigh
                + (idy_msk_neigh < 0) * size_data[0]
                - (idy_msk_neigh >= size_data[0]) * size_data[0]
            )
            idx_msk_neigh = (
                idx_msk_neigh
                + (idx_msk_neigh < 0) * size_data[1]
                - (idx_msk_neigh >= size_data[1]) * size_data[1]
            )

            id_msk = (idy_msk, idx_msk, ich)
            id_msk_neigh = (idy_msk_neigh, idx_msk_neigh, ich)

            output[id_msk] = input[id_msk_neigh]
            mask[id_msk] = 0.0

        return output, mask
