import os
import pandas as pd
import glob as glob
import cv2
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data import Subset


class BaseDataset(Dataset):
    def __init__(self, paths=[["data/Deteling/graphics"],["data/Deteling/details"]],transform=None, num_samples = 10000):
        self.transform = transform
        self.paths = paths

        files = []
        labels = []
        all_data = []
        for i, p in zip(range(len(paths)), paths):
            tmp_files =  []
            for pp in paths[i]:

                files_single_dir = glob.glob(f"{pp}/*.png")
                tmp_files += files_single_dir

            tmp_files = sorted(tmp_files, key=os.path.getsize, reverse=True)
            num_samples = min(len(tmp_files), num_samples) if num_samples > 0 else len(tmp_files)
            files += tmp_files[:num_samples]
            labels += [i] * num_samples

        for label, path in zip(files, labels):
            all_data.append({"label": label, "path": path})

        self.df = pd.DataFrame(all_data)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        path = self.df.iloc[index, 0]
        label = self.df.iloc[index, 1]
        image = cv2.imread(path)
        image_out = image.copy()
        if self.transform:
            image_out = self.transform(image)

        return {'working_image': image_out, 'orig_image': image, 'label': label, 'path': path}