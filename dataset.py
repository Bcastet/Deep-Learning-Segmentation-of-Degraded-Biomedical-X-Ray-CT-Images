import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy
import torchvision
import os

images_per_set = 1959
padder = torchvision.transforms.Pad(1, fill=0, padding_mode='constant')


class SeparatedDataset(Dataset):
    def __init__(self, damaged_data_source, segmented_data_source, input_source_dir, target_source_dir):
        self.input_path = damaged_data_source
        self.target_path = segmented_data_source
        self.input_source = input_source_dir
        self.target_source = target_source_dir
        self.sets = []
        self.len = 0
        self.len_sets = []

        for seq in os.listdir(damaged_data_source):
            if os.path.isdir(os.path.join(damaged_data_source, seq)):
                self.sets.append(seq)
                self.len += len(os.listdir(os.path.join(*[damaged_data_source, seq, input_source_dir])))
                self.len_sets.append(len(os.listdir(os.path.join(*[damaged_data_source, seq, input_source_dir]))))

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        if isinstance(item, torch.Tensor):
            item = item.item()

        set_idx = 0
        while item >= self.len_sets[set_idx]:
            item -= self.len_sets[set_idx]
            set_idx += 1

        img_idx = item

        file = os.path.join(*[self.input_path, str(self.sets[set_idx]), self.input_source, str(img_idx) + ".tif"])

        source = numpy.asarray(Image.open(file))

        target_file = os.path.join(
            *[self.input_path, str(self.sets[set_idx]), self.target_source, str(img_idx) + ".tif"])
        target = numpy.asarray(Image.open(target_file))

        source = source.reshape(1, source.shape[0], source.shape[1])
        target = target.reshape(1, target.shape[0], target.shape[1])

        source = source / 255
        target = target / 255

        source = torch.tensor(source).float()
        target = torch.tensor(target).float()

        return [source, target]


class UnifiedDataset(SeparatedDataset):
    def __init__(self, damaged_data_source, segmented_data_source):
        pass
        super().__init__()



class datasetAsPatches(Dataset):
    def __init__(self, standard_dataset: SeparatedDataset):
        self.source_dataset = standard_dataset

    def __len__(self):
        # return 101
        return len(self.source_dataset) * 5 * 5

    def __getitem__(self, item):
        if isinstance(item, torch.Tensor):
            item = item.item()

        image_idx = item // 25
        patch_idx = item % 25
        input_image, target_image = self.source_dataset[image_idx]
        return imageAsDataset(input_image, target_image, 256, 5)[patch_idx]


class imageAsDataset(Dataset):
    def __init__(self, tensor_image, target_image, size, divide_by):
        self.tensor_image = tensor_image.reshape((1360, 1360, 1))
        self.target_image = target_image.reshape((1360, 1360, 1))
        self.size = size
        self.divide_by = divide_by

    def __len__(self):
        return self.divide_by * self.divide_by

    def __getitem__(self, item):
        x = item % self.divide_by
        y = item // self.divide_by

        inp = self.tensor_image[y * self.size:(y + 1) * self.size, x * self.size:(x + 1) * self.size]
        tar = self.target_image[y * self.size:(y + 1) * self.size, x * self.size:(x + 1) * self.size]

        return inp.reshape((1, self.size, self.size)), \
               tar.reshape((1, self.size, self.size)), x, y
