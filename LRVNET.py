import torch
from PIL import Image
import numpy
import math

patch_size = 5


class LRFFCNCNN(torch.nn.Module):
    def __init__(self):
        super(LRFFCNCNN, self).__init__()
        self.unetDown = UNetDown()
        self.unetUp = UNetUp()
        # self.lowres = lowResolutionBranch()
        self.fireSqueeze = FireSqueeze()
        self.cropLayer = CropLayer()

    def forward(self, pred_large_image, patch, coordx, coordy):
        patch_coords = (coordx, coordy)
        pred_large_image = self.cropLayer(pred_large_image, patch_coords)
        down, transverse1, transverse2, transverse3, transverse4 = self.unetDown(patch)
        pred = self.fireSqueeze(down, pred_large_image)
        pred = self.unetUp(pred, transverse1, transverse2, transverse3, transverse4)
        return pred


class CropLayer(torch.nn.Module):
    def __init__(self):
        super(CropLayer, self).__init__()
        self.output_layer = torch.nn.Conv2d(256, 256, (1, 1))

    def forward(self, x, coordinates_b):
        batch = torch.tensor([]).cuda().float()
        for index in range(len(coordinates_b[0])):
            coordinates = coordinates_b[0][index], coordinates_b[1][index],

            x = x.reshape(85, 85, 256)
            pred = x[coordinates[1].item():coordinates[1].item() + patch_size,
                   coordinates[0].item():coordinates[0].item() + patch_size]
            pred = pred.reshape(1, 256, patch_size, patch_size)
            batch = torch.cat([batch, pred], 0)
        return self.output_layer(batch)


class UNetDown(torch.nn.Module):
    def __init__(self):
        super(UNetDown, self).__init__()
        self.input = UNetInputLayers()
        self.transverse1 = Transverse(64, 32)
        self.godown1 = UNetGoDown()
        self.down1 = UNetDown1Layers()
        self.transverse2 = Transverse(64, 32)
        self.godown2 = UNetGoDown()
        self.down2 = UNetDown2Layers()
        self.transverse3 = Transverse(128, 64)
        self.godown3 = UNetGoDown()
        self.down3 = UNetDown3Layers()
        self.transverse4 = Transverse(256, 128)
        self.down4 = MixedPooling()

    def forward(self, x):
        x = self.input(x)
        transverse1 = self.transverse1(x)

        x = self.godown1(x)

        x = self.down1(x)
        transverse2 = self.transverse2(x)
        x = self.godown2(x)
        x = self.down2(x)
        transverse3 = self.transverse3(x)
        x = self.godown3(x)
        x = self.down3(x)
        transverse4 = self.transverse4(x)
        x = self.down4(x)
        return x, transverse1, transverse2, transverse3, transverse4


class UNetUp(torch.nn.Module):
    def __init__(self):
        super(UNetUp, self).__init__()
        self.up1 = UNetUp1()
        self.goup1 = UNetGoUp((21, 21))
        self.up2 = UNetUp2()
        self.goup2 = UNetGoUp((42, 42))
        self.up3 = UNetUp3()
        self.goup3 = UNetGoUp((85, 85))
        self.output = UNetOutputLayers()

    def forward(self, x, transverse1, transverse2, transverse3, transverse4):
        x = torch.cat([transverse4, x], 1)
        x = self.up1(x)
        x = self.goup1(x)
        x = torch.cat([transverse3, x], 1)
        x = self.up2(x)
        x = self.goup2(x)
        x = torch.cat([transverse2, x], 1)
        x = self.up3(x)
        x = self.goup3(x)
        x = torch.cat([transverse1, x], 1)
        return self.output(x)


class FireSqueeze(torch.nn.Module):
    def __init__(self):
        super(FireSqueeze, self).__init__()
        self.layer1 = torch.nn.Conv2d(512, 256, (1, 1))
        self.selu = torch.nn.SELU()
        self.splitlayer = Splitted3Layers(256, 512)

        self.layers = torch.nn.Sequential(
            torch.nn.SELU(),
            torch.nn.Conv2d(512, 256, (3, 3), padding=1),
            torch.nn.SELU(),
            torch.nn.Conv2d(256, 128, (1, 1)),
            torch.nn.Upsample((10, 10))
        )

    def forward(self, fromUp, fromLowRes):
        x = torch.cat([fromUp, fromLowRes], 1)
        x = self.layer1(x)
        x = self.selu(x)
        x = self.splitlayer(x)
        x = self.layers(x)

        return x


class UNetInputLayers(torch.nn.Module):
    def __init__(self):
        super(UNetInputLayers, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(1, 6, (1, 1)),
            torch.nn.SELU(),
            torch.nn.Conv2d(6, 6, (1, 1)),
            torch.nn.SELU(),
            torch.nn.Conv2d(6, 32, (3, 3), padding=1),
            torch.nn.SELU(),
            torch.nn.Conv2d(32, 64, (3, 3), padding=1),
            torch.nn.SELU()
        )

    def forward(self, x):
        return self.layers(x)


class UNetDown1Layers(torch.nn.Module):
    def __init__(self):
        super(UNetDown1Layers, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(64, 32, (1, 1)),
            torch.nn.SELU(),
            torch.nn.Conv2d(32, 64, (3, 3), padding=1),
            torch.nn.SELU()
        )

    def forward(self, x):
        return self.layers(x)


class UNetDown2Layers(torch.nn.Module):
    def __init__(self):
        super(UNetDown2Layers, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, (1, 1)),
            torch.nn.SELU(),
            torch.nn.Conv2d(64, 128, (3, 3), padding=1),
            torch.nn.SELU()
        )

    def forward(self, x):
        return self.layers(x)


class UNetDown3Layers(torch.nn.Module):
    def __init__(self):
        super(UNetDown3Layers, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(128, 128, (1, 1)),
            torch.nn.SELU(),
            torch.nn.Conv2d(128, 256, (3, 3), padding=1),
            torch.nn.SELU()
        )

    def forward(self, x):
        return self.layers(x)


class UNetGoDown(torch.nn.Module):
    def __init__(self):
        super(UNetGoDown, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.AvgPool2d((2, 2)),
            torch.nn.SELU()
        )

    def forward(self, x):
        return self.layers(x)


class MixedPooling(torch.nn.Module):
    def __init__(self):
        super(MixedPooling, self).__init__()
        self.maxPooling = torch.nn.MaxPool2d((2, 2))
        self.avgPooling = torch.nn.AvgPool2d((2, 2))
        self.gamma = 0.5
        # TODO : implement gamma gradient descent learning

    def forward(self, x):
        return (self.gamma * self.maxPooling(x)) + ((1 - self.gamma) * self.avgPooling(x))


class Transverse(torch.nn.Module):
    def __init__(self, inC, outC):
        super(Transverse, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(inC, outC, (1, 1)),
            torch.nn.SELU()
        )

    def forward(self, x):
        return self.layers(x)


class UNetUp1(torch.nn.Module):
    def __init__(self):
        super(UNetUp1, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(256, 128, (3, 3), padding=1),
            torch.nn.SELU(),
            torch.nn.Conv2d(128, 64, (1, 1))
        )

    def forward(self, x):
        return self.layers(x)


class UNetUp2(torch.nn.Module):
    def __init__(self):
        super(UNetUp2, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(128, 64, (1, 1)),
            torch.nn.SELU(),
            torch.nn.Conv2d(64, 32, (3, 3), padding=1)
        )

    def forward(self, x):
        return self.layers(x)


class UNetUp3(torch.nn.Module):
    def __init__(self):
        super(UNetUp3, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(64, 32, (3, 3), padding=1),
            torch.nn.SELU(),
            torch.nn.Conv2d(32, 32, (1, 1))
        )

    def forward(self, x):
        return self.layers(x)


class UNetOutputLayers(torch.nn.Module):
    def __init__(self):
        super(UNetOutputLayers, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(64, 32, (1, 1)),
            torch.nn.BatchNorm2d(32),
            torch.nn.SELU(),
            torch.nn.Conv2d(32, 1, (1, 1)),
            torch.nn.BatchNorm2d(1),
            torch.nn.ReLU(),
            torch.nn.Sigmoid()
        )
        self.threshold = torch.nn.Threshold(0.8, 0)

    def forward(self, x):
        return self.threshold(self.layers(x))


class UNetGoUp(torch.nn.Module):
    def __init__(self, dim):
        super(UNetGoUp, self).__init__()
        self.layers = torch.nn.Upsample(dim)

    def forward(self, x):
        return self.layers(x)


class Splitted3Layers(torch.nn.Module):
    def __init__(self, inC, outC):
        super(Splitted3Layers, self).__init__()
        self.layers = [
            torch.nn.Sequential(torch.nn.Conv2d(inC, inC, (1, 1), padding=0).cuda(), torch.nn.SELU()),
            torch.nn.Sequential(torch.nn.Conv2d(inC, inC, (3, 3), padding=1).cuda(), torch.nn.SELU()),
            torch.nn.Sequential(torch.nn.Conv2d(inC, inC, (5, 5), padding=2).cuda(), torch.nn.SELU()),
        ]
        self.selu = torch.nn.Sequential(torch.nn.Conv2d(inC * 4, outC, (1, 1)), torch.nn.SELU())

    def forward(self, x):
        outs = [x]
        for layer in self.layers:
            outs.append(layer(x))
        x = torch.cat(outs, 1)
        return self.selu(x)


class lowResolutionBranch(torch.nn.Module):
    def __init__(self):
        super(lowResolutionBranch, self).__init__()
        self.subsampling = torch.nn.AvgPool2d(16, 16)
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(1, 6, kernel_size=(1, 1), padding=1),
            torch.nn.SELU(),
            torch.nn.Conv2d(6, 32, (3, 3)),
            torch.nn.SELU(),
            Splitted3Layers(32, 64),
            torch.nn.Conv2d(64, 64, (3, 3), padding=1),
            torch.nn.SELU(),
            Splitted3Layers(64, 128),
            torch.nn.Conv2d(128, 128, (3, 3), padding=1),
            torch.nn.SELU(),
            Splitted3Layers(128, 256),
            torch.nn.Conv2d(256, 256, (3, 3), padding=1),
            torch.nn.SELU(),
            torch.nn.Conv2d(256, 256, (3, 3), padding=1),
            torch.nn.SELU()
        )
        self.softmax = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, (1, 1)),
            torch.nn.BatchNorm2d(256),
            torch.nn.SELU(),
            torch.nn.Conv2d(256, 1, (1, 1)),
            torch.nn.BatchNorm2d(1),
            torch.nn.ReLU(),
            torch.nn.Sigmoid()
        )
        self.threshold = torch.nn.Threshold(0.8, 0)

    def forward(self, x):
        x = self.subsampling(x)
        return self.threshold(self.softmax(self.layers(x)))


class functionnalLowRes(torch.nn.Module):
    def __init__(self, lowres):
        super(functionnalLowRes, self).__init__()
        self.lowres = lowres

    def forward(self, x):
        return self.lowres.layers(self.lowres.subsampling(x))
