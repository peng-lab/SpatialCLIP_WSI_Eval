from torch.utils.data import Dataset
from PIL import Image
import pandas as pd


class SlideDataset(Dataset):
    def __init__(self, slide, coordinates: pd.DataFrame, patch_size: int, transform:list =None):
        super(SlideDataset, self).__init__()
        self.slide=slide
        self.coordinates=coordinates
        self.patch_size=patch_size
        self.transform = transform

    def __len__(self):
        return len(self.coordinates)

    def __getitem__(self, idx):
        entry=self.coordinates.iloc[idx]
        x=entry['x']
        y=entry['y']

        patch = self.slide[x:x+self.patch_size, y:y+self.patch_size, :]
        img = Image.fromarray(patch)  # Convert image to RGB

        if self.transform:
            img = self.transform(img)

        return img


class PatchDataset(Dataset):
    def __init__(self, image_paths, transform:list =None):
        super(PatchDataset, self).__init__()
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx])

        if self.transform:
            img = self.transform(img)

        return img