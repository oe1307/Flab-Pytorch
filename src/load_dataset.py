import os
from glob import glob

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.datasets.imagenet import parse_devkit_archive, parse_val_archive


class MyDataset(Dataset):
    def __init__(self, path):
        """
        set path of images and transform
        """
        self.path = glob(path)
        self.labels = os.listdir(os.path.dirname(self.path[0]))
        self.n_classes = len(self.labels)
        meta_data = torch.load("../storage/meta.bin")
        print(meta_data[0])
        self.transform = T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        """set number of images in dataset"""
        return len(self.path)

    def __getitem__(self, idx):
        """get image and label per index"""
        image = Image.open(self.path[idx]).convert("RGB")
        image = self.transform(image)
        label = self.path[idx].split("/")[-2]
        return image, label


if __name__ == "__main__":
    if not os.path.exists("../storage/val"):
        print("Expanding imagenet dataset...")
        parse_devkit_archive("../storage")
        parse_val_archive("../storage")

    dataset = MyDataset("../storage/val/*/*.JPEG")
    print("n_examples =", len(dataset))
    print("n_classes =", dataset.n_classes)

    # check 0th data
    data, label = dataset[0]
    print(data)
    print(data.shape)
    print(label)
