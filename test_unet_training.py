import unet_training


def assertError(fun, args):
    try:
        fun(args)
    except unet_training.IllegalArgumentError as e:
        return True, e
    return False, None


def main_args_test(args, should_go_through):
    isException, error = assertError(unet_training.parse_args, args)
    if isException == should_go_through:
        if not should_go_through:
            raise Exception(str(args) + " should raise an exception")
        else:
            raise error


def test_wrong_arguments():
    main_args_test(['--input', 'repository'], False)
    main_args_test(['--input', 'D:/Datasets', '--target', 'repository'], False)
    main_args_test(['--input', 'D:/Datasets', '--target', 'D:/Datasets', "--save", "bugs"], False)
    main_args_test(['--input', 'D:/Datasets', '--target', 'D:/Datasets', "--model_type", "Unet"], False)


def test_correct_arguments():
    main_args_test(
        ['--input', 'D:/Datasets', '--target', 'D:/Datasets', "--model_type", "Unet", "--data_mode", "separated",
         "--damaged_dir", "ReconFBP_crop_300", "--segmented_dir", "ReconFBP_1800_SEGM"], True)


def test_training():
    argv = ['--input', 'D:/Datasets', '--target', 'D:/Datasets', "--model_type", "Unet", "--data_mode", "separated",
            "--damaged_dir", "ReconFBP_crop_300", "--segmented_dir", "ReconFBP_1800_SEGM"]
    try:
        unet_training.main(argv)
    except Exception as e:
        raise e
