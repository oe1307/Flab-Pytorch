import os
from glob import glob

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.datasets import ImageNet


class MyDataset(Dataset):
    def __init__(self, path):
        self.path = glob(path)
        self.transform = T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.path)

    def __getitem__(self, idx):
        image = Image.open(self.path[idx]).convert("RGB")
        image = self.transform(image)
        label = self.path[idx].split("/")[-2]
        return image, label


if __name__ == "__main__":
    if not os.path.exists("../storage/imagenet/val"):
        print("Expanding imagenet dataset...")
        ImageNet("../storage/imagenet", split="val")
    dataset = MyDataset("../storage/imagenet/val/*/*.JPEG")
    data, label = dataset[0]
    print(data.shape)
    print(label)
