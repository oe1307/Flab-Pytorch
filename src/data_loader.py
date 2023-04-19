from torch.utils.data import DataLoader

from dataset import MyDataset

if __name__ == "__main__":
    # below is the example of how to use DataLoader

    dataset = MyDataset("../storage/imagenet/val/*/*.JPEG")
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

    for batch, (data, label) in enumerate(dataloader):
        print(data)
        print(data.shape)
        print(label)
