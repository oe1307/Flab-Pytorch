import json
import os
from glob import glob
from urllib.request import urlretrieve

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.datasets.imagenet import parse_devkit_archive, parse_val_archive


class MyDataset(Dataset):
    def __init__(self, path):
        """set path of images and transform"""
        self.path = glob(path)
        class_index = json.load(open("../storage/imagenet_class_index.json", "r"))
        self.label2index = {v[0]: int(k) for k, v in class_index.items()}
        self.n_classes = len(self.label2index)
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
        index = self.label2index[label]
        return image, label, index


if __name__ == "__main__":
    if not os.path.exists("../storage/val"):
        print("Expanding imagenet dataset...")
        parse_devkit_archive("../storage")
        parse_val_archive("../storage")
        urlretrieve(
            "https://s3.amazonaws.com/deep-learning-models/"
            + "image-models/imagenet_class_index.json",
            "../storage/imagenet_class_index.json",
        )

    dataset = MyDataset("../storage/val/*/*.JPEG")
    print("n_examples =", len(dataset))
    print("n_classes =", dataset.n_classes)

    # check 0th data
    data, label, index = dataset[0]
    print(data)
    print(data.shape)
    print("label:", label)
    print("index:", index)
