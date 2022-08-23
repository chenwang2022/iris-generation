import os
from PIL import Image
from torch.utils.data import Dataset


class IrisDataset(Dataset):
    def __init__(self, path, transform):

        self.path = path
        self.transform = transform
        self.imglist = os.listdir(path)

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, index):
        name = self.imglist[index]
        img = Image.open(os.path.join(self.path, name)).convert("RGB")
        img = self.transform(img)

        return img

