import timm
from torch import nn
from torch.utils.data import DataLoader

from load_dataset import MyDataset


class ResNet50(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.base_model = timm.create_model("resnet50", pretrained=True)
        self.base_model.fc.out_features = n_classes
        self.fc = nn.Linear(self.base_model.fc.in_features, n_classes)

    def forward(self, x):
        x = self.base_model(x)
        return x

    def forward_1(self, x):
        """other example of forward function"""
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.act1(x)
        x = self.base_model.maxpool(x)
        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        feature = x.clone()  # extract feature of layer3
        x = self.base_model.layer4(x)
        x = self.base_model.global_pool(x)
        x = nn.Dropout(p=self.base_model.drop_rate)(x)  # add dropout
        # x = self.base_model.fc(x)
        x = self.fc(x)  # use the new fc layer
        return x, feature


if __name__ == "__main__":
    dataset = MyDataset("../storage/val/*/*.JPEG")
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    model = ResNet50(dataset.n_classes)

    # code for checking the layers of the model
    print("layers: ")
    layers = dict(model.named_modules()).keys()
    layers = [layer for layer in layers if len(layer.split(".")) == 2]
    for layer in layers:
        print(layer)
    print()

    for images, labels, indexes in dataloader:
        outputs = model(images)
        print(outputs)
        print(outputs.shape)
        print("prediction:", outputs.argmax(dim=1))
        print("ground_truth:", indexes)
        break
