import torch
import dataset as ds
import numpy as np
from PIL import Image
import getopt
import os


class IllegalArgumentError(ValueError):
    pass


dirname = os.path.dirname(__file__)


def build_with_Unet(network, dataset, save_folder):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
    stack = []
    for i_image, batch in enumerate(dataloader):
        x = batch[0]
        y = batch[1]
        imagedset = ds.imageAsDataset(x, y, 256, 5)
        imageloader = torch.utils.data.DataLoader(imagedset, batch_size=16, shuffle=False, num_workers=2)

        image_pred = np.empty((x.shape[2], x.shape[3]))
        for batch_index, batch_sample in enumerate(imageloader):
            patchx, patchy, coordx, coordy = batch_sample
            patch_pred = network(patchx.cuda())
            patch_pred = np.where(patch_pred.cpu() > 0.5, 1, 0)
            for i, patch in enumerate(patch_pred):
                x_ = int(((coordx[i] / 5) * image_pred.shape[0]).item())
                y_ = int(((coordy[i] / 5) * image_pred.shape[1]).item())
                image_pred[y_: y_ + patchy[i].shape[2], x_: x_ + patchx[i].shape[1]] = patch[0]

        stack.append(image_pred)
        Image.fromarray(image_pred).save(save_folder + "UnetBuild" + str(i_image) + '.tif')
    return stack


def build_with_LRVNet(lowResBranch, network, dataset, save_folder):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
    stack = []
    for i_image, batch in enumerate(dataloader):
        x = batch[0]
        y = batch[1]
        lowres_pred = lowResBranch(x.float().cuda())
        imagedset = ds.imageAsDataset(x, y, 85, 16)
        imageloader = torch.utils.data.DataLoader(imagedset, batch_size=8, shuffle=False, num_workers=2)

        image_pred = np.empty((x.shape[2], x.shape[3]))
        for batch_index, batch_sample in enumerate(imageloader):
            patchx, patchy, coordx, coordy = batch_sample
            patch_pred = network(lowres_pred, patchx.float().cuda(), coordx, coordy)
            patch_pred = np.where(patch_pred.cpu() > 0.5, 1, 0)
            for i, patch in enumerate(patch_pred):
                x_ = int(((coordx[i] / 16) * image_pred.shape[0]).item())
                y_ = int(((coordy[i] / 16) * image_pred.shape[1]).item())
                image_pred[y_: y_ + patchy[i].shape[2], x_: x_ + patchx[i].shape[1]] = patch[0]

        stack.append(image_pred)
        Image.fromarray(image_pred).save(save_folder + "LRVNetBuild" + str(i_image) + '.tif')
    return stack


def usage(exception):
    print("build_stack usage:")
    print("   -must be specified:")
    print("     --input <input>              : the directory where the seqs are stored")
    print("     --model_type <model_type>    : 'Unet' or 'LRVNET'")
    print("     --model_location <model.pt>  :  directory of the model you want to use")
    print(
        "     --lowresbranch <model.pt>    :  directory of the low resolution branch you want to use. Only used if model_type is 'LRVNET'")
    print("     --save   <save_repository>   : the location the model should be saved at")
    print("     --damaged_dir <damaged_dir>  : name of the directory where inputs are located. Ex : ReconFBP_crop_300")
    print("   -other options")
    print(
        "     --segmented_dir <damaged_dir>  : name of the directory where inputs are located. Ex : ReconFBP_1800_SEGM")
    raise Exception


def parse_args(argv):
    input_data_repository = None
    model_type = None
    model_location = None
    lowresbranch = None
    save_repository = None
    damaged_dir = None
    segmented_dir = None

    opts, args = getopt.getopt(argv, "",
                               ["model_type =", "model_location =",
                                "lowresbranch =", "save =", "damaged_dir =", "input =", "segmented_dir ="])

    for opt, arg in opts:
        if opt == "--input ":
            input_data_repository = os.path.join(dirname, arg)
            if not os.path.isdir(input_data_repository):
                usage(IllegalArgumentError("--input " + input_data_repository + " is not a valid directory"))
        if opt == "--model_type ":
            if arg in ["Unet", "LRVNet"]:
                model_type = arg
            else:
                usage(IllegalArgumentError("--model_type must be in " + str(["Unet", "LRVnet"]) + " not " + arg))
        if opt == "--model_location ":
            model_location = os.path.join(dirname, arg)
            if not os.path.isfile(model_location):
                usage(IllegalArgumentError("--model_type " + model_location + " is not a valid file"))

        if opt == "--lowresbranch ":
            lowresbranch = os.path.join(dirname, arg)
            if not os.path.isfile(lowresbranch):
                usage(IllegalArgumentError("--lowresbranch " + lowresbranch + " is not a valid file"))

        if opt == "--save ":
            save_repository = os.path.join(dirname, arg)
            if not os.path.isdir(save_repository):
                usage(IllegalArgumentError("--save " + save_repository + " is not a valid directory"))

        if opt == "--damaged_dir ":
            damaged_dir = arg

        if opt == "--segmented_dir ":
            segmented_dir = arg
    if None in [model_type, model_location, save_repository, damaged_dir, input_data_repository]:
        usage(IllegalArgumentError("Mandatory argument missing!"))

    if lowresbranch is None and model_type == "LRVNet":
        usage(Exception("You selected LRVNet but didn't specify LowresBranchLocation"))

    return model_type, model_location, lowresbranch, save_repository, damaged_dir, input_data_repository, segmented_dir


def main(argv):
    model_type, model_location, lowresbranch_location, save_repository, damaged_dir, input_data_repository, segmented_dir = parse_args(
        argv)
    device = torch.device("cuda:0")

    if segmented_dir is None:
        segmented_dir = damaged_dir
    dataset = ds.SeparatedDataset(input_data_repository, input_data_repository, damaged_dir, segmented_dir)

    if model_type == "Unet":
        from unet import UNet
        model = UNet(in_channels=1, out_channels=1)
        model.load_state_dict(torch.load(model_location))
        model.to(device)
        build_with_Unet(model, dataset, save_repository)
    elif model_type == "LRVNet":
        from LRVNET import LRFFCNCNN, lowResolutionBranch, functionnalLowRes
        unet = LRFFCNCNN()
        lowresbranch = lowResolutionBranch()

        unet.load_state_dict(torch.load(model_location))
        lowresbranch.load_state_dict(torch.load(lowresbranch_location))
        funLowRes = functionnalLowRes(lowresbranch)
        unet.to(device)
        funLowRes.to(device)
        build_with_LRVNet(funLowRes, unet, dataset, save_repository)


if __name__ == '__main__':
    argv = ['--input', 'D:/Datasets', "--model_type", "LRVNet", "--damaged_dir", "ReconFBP_crop_300", "--segmented_dir",
            "ReconFBP_1800_SEGM", "--save", "E:/Builds/", "--model_location", "models/defaultLRVnet.pth", "--lowresbranch","models/defaultlowresBranch.pt"]
    main(argv)
