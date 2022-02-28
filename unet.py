from collections import OrderedDict

import torch
import torch.nn as nn
import dataset as ds
from loss import DiceLoss
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
# from utils import *
import matplotlib.pyplot as plt
import numpy as np

verbose = True
show_images_on_train = False


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet, self).__init__()

        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )
        eps = 1e-10
        self.threshold = torch.nn.Threshold(0.5, 0)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return self.threshold(torch.sigmoid(self.conv(dec1)))

    @staticmethod
    def _block(in_channels, features, name, padding=1):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=padding,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=padding,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )


def run_train(model, epochs, dataset, device):
    dataset = ds.datasetAsPatches(dataset)
    lr = 1e-3
    batch_size = 16
    vis_freq = 1

    #train_set, val_set = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.1),
    #                                                             int(len(dataset) - int(len(dataset) * 0.1))])
    train_set = dataset

    loader_train = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    #loader_valid = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=2)
    loaders = {"train": loader_train, "valid": None}

    print("Dataset size : ", len(train_set))
    print("Steps per epoch : ", len(train_set) // batch_size)

    unet = model
    unet.to(device)

    dsc_loss = DiceLoss()

    optimizer = optim.Adam(unet.parameters(), lr=lr)

    loss_train = []
    step = 0

    for epoch in tqdm(range(epochs), total=epochs):
        print("Current epoch:", epoch)
        for phase in ["train"]:
            if phase == "train":
                unet.train()
            else:
                unet.eval()

            for i, data in enumerate(loaders[phase]):
                if phase == "train":
                    step += 1
                x, y_true, x_pos, y_pos = data
                x, y_true = x.to(device), y_true.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    y_pred = unet(x)

                    loss = dsc_loss(y_pred, y_true)
                    if phase == "train":
                        loss_train.append(loss.item())
                        loss.backward()
                        optimizer.step()
                        if loss.item() < 0 and show_images_on_train == True:
                            plt.imshow(x)
                            plt.title("Prediction")
                            plt.show()
                            plt.imshow(y_pred)
                            plt.title("Prediction")
                            plt.show()
                            plt.imshow(y_true)
                            plt.title("Truth")
                            plt.show()

                if phase == "train" and (step + 1) % 10 == 0 and verbose:
                    print("Step", step, ":", np.mean(loss_train))
                    loss_train = []

        torch.save(unet.state_dict(), "unet_intermediary.pt")
    return unet
