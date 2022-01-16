import sys
import getopt
import os
import torch
from dataset import SeparatedDataset, UnifiedDataset
import warnings

dirname = os.path.dirname(__file__)


class IllegalArgumentError(ValueError):
    pass


def usage(exception):
    print("-unet_training usage:")
    print("   -must be specified")
    print("     --input  <input_repository>  : the data used to train the model")
    print("     --target <target_repository> : the segmented images the model will reproduce")
    print("     --model_type <model_type>    : 'Unet' or 'LRVNET'")
    print("     --data_mode <data_mode>      : 'separated' or 'unified'")
    print("   -other options")
    print("     --damaged_dir <damaged_dir>  : name of the directory where inputs are located. Only for separated mode")
    print(
        "     --segmented_dir <segmented_dir>: name of the directory where targets are located. Only for separated mode")
    print("     --save   <save_repository>   : the location the model should be saved at")
    print("     --model  <pretrained_model>  : use this option to train an already trained model")
    print("     --epochs <epochs>            : the number of epochs")
    raise exception


def parse_args(argv):
    print(argv)
    opts, args = getopt.getopt(argv, "",
                               ["input =", "target =", "save =", "model =", "epochs =", "model_type =", "data_mode =",
                                "damaged_dir =", "segmented_dir ="])
    print(opts)
    input_data_repository = None
    target_repository = None
    model_type = None
    model_save_repository = None
    use_model_in_repo = None
    epochs = None
    data_mode = None
    damaged_dir = None
    segmented_dir = None

    for opt, arg in opts:
        if opt == "--input ":
            input_data_repository = os.path.join(dirname, arg)
            print(input_data_repository)
            if not os.path.isdir(input_data_repository):
                usage(IllegalArgumentError("--input " + input_data_repository + " is not a valid directory"))
        if opt == "--target ":
            target_repository = os.path.join(dirname, arg)
            if not os.path.isdir(input_data_repository):
                usage(IllegalArgumentError("--target " + target_repository + " is not a valid directory"))
        if opt == "--save ":
            model_save_repository = os.path.join(dirname, arg)
            if not os.path.isdir(input_data_repository):
                usage(IllegalArgumentError("--save " + model_save_repository + " is not a valid directory"))
        if opt == "--pretrained_model ":
            use_model_in_repo = os.path.join(dirname, arg)
            if not os.path.isdir(input_data_repository):
                usage(IllegalArgumentError("--model " + use_model_in_repo + " is not a valid directory"))
        if opt == "--model_type ":
            if arg in ["Unet", "LRVnet"]:
                model_type = arg
            else:
                usage(IllegalArgumentError("--model_type must be in " + str(["Unet", "LRVnet"]) + " not " + arg))
        if opt == "--epochs ":
            epochs = arg
            try:
                epochs = int(epochs)
            except:
                usage(IllegalArgumentError("--epochs should be int, " + epochs + " is not valid"))
        if opt == "--data_mode ":
            if arg in ["separated", "unified"]:
                data_mode = arg
        if opt == "--damaged_dir ":
            damaged_dir = arg
        if opt == "--segmented_dir ":
            segmented_dir = arg

    if input_data_repository is None or target_repository is None or model_type is None or data_mode is None:
        usage(IllegalArgumentError("--input, --target, --model_type and --data_mode should be specified"))

    # if data_mode == "separated" and input_data_repository != target_repository:
    #    usage(Exception("data_mode is separated, therefore --input and --target should be equal"))

    if data_mode == "separated" and (damaged_dir is None or segmented_dir is None):
        usage(IllegalArgumentError("data_mode is separated, therefore damaged_dir and segmented_dir must be specified"))

    if data_mode == "separated":
        images_input = []
        images_target = []
        for seq in os.listdir(input_data_repository):
            seq_repo = os.path.join(input_data_repository, seq)
            if os.path.isdir(seq_repo):
                input_repo = os.path.join(seq_repo, damaged_dir)
                if not os.path.isdir(input_repo):
                    usage(IllegalArgumentError(
                        input_repo + " doesn't exist. Arguments must be so <input>/directory/<damaged_dir> is a dir, for every dir in <input>"))
                images_input.append(len(os.listdir(input_repo)))

        for seq in os.listdir(target_repository):
            seq_repo = os.path.join(target_repository, seq)
            if os.path.isdir(seq_repo):
                target_repo = os.path.join(seq_repo, segmented_dir)
                if not os.path.isdir(target_repo):
                    usage(IllegalArgumentError(
                        target_repo + " doesn't exist. Arguments must be so <target>/directory/<segmented_dir> is a dir, for every dir in <target>"))
                images_target.append(len(os.listdir(target_repo)))

        if os.listdir(target_repository) != os.listdir(input_data_repository) or str(images_input) != str(
                images_target):
            usage(IllegalArgumentError("Seqs in input are different from seqs in target : \nSeqs target" + str(
                os.listdir(input_data_repository)) + "\nSeqs input " + str(
                os.listdir(target_repository)) + "\nNumber of images in inputs " + str(
                images_input) + "\nNumber of images in target " + str(images_target)))

    return input_data_repository, target_repository, model_type, model_save_repository, use_model_in_repo, epochs, data_mode, damaged_dir, segmented_dir


def main(argv):
    input_data_repository, target_repository, model_type, model_save_repository, use_model_in_repo, epochs, data_mode, damaged_dir, segmented_dir = parse_args(
        argv)
    warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)

    if model_type == "Unet":
        from unet import UNet, run_train
        model = UNet(in_channels=1, out_channels=1)

    elif model_type == "LRVNet":
        from LRVNET import LRFFCNCNN, run_train
        model = LRFFCNCNN()

    if use_model_in_repo is not None:
        model.load_state_dict(torch.load(use_model_in_repo))

    if data_mode == "separated":
        dataset = SeparatedDataset(input_data_repository, target_repository, damaged_dir, segmented_dir)
    elif data_mode == "unified":
        dataset = UnifiedDataset(input_data_repository, target_repository)

    if epochs is not None:
        epochs = 3


    device = torch.device("cuda:0")

    model = run_train(model, epochs, dataset, device)


if __name__ == '__main__':
    argv = ['--input', 'D:/Datasets', '--target', 'D:/Datasets', "--model_type", "Unet", "--data_mode", "separated",
            "--damaged_dir", "ReconFBP_crop_300", "--segmented_dir", "ReconFBP_1800_SEGM"]
    main(argv)
