import os
import torch
import torch.utils.data as dt
import torchvision.transforms as transforms
from PIL import Image


# Saving and loading models and model states
def save_model(network, path='state'):
    with open(path, 'w'):
        torch.save(network, path)


def load_model(path):
    return torch.load(path)


def save_state(network, path='state'):
    torch.save(network.state_dict(), path)


def load_state(class_, path):
    network = class_()
    network.load_state_dict(torch.load(path))
    return network


# Tensor processing
def class_max(tensor):
    _, _, height, width = tensor.shape
    output = torch.zeros(1, height, width)
    for row in range(height):
        for col in range(width):
            if tensor[0, 0, row, col] > tensor[0, 1, row, col]:
                output[0, row, col] = 1
            else:
                output[0, row, col] = 0
    return output


# Dataset processing
class CarvanaDataset(dt.Dataset):

    def __init__(self, data_path, mask_path, input_size=224):
        """
            data_path (string): Path to the images data files.
            mask_path (string): Path were images masks are placed
        """

        self.files = os.listdir(data_path)
        self.files.sort()
        self.mask_files = os.listdir(mask_path)
        self.mask_files.sort()
        self.data_path = data_path
        self.mask_path = mask_path
        assert (len(self.files) == len(self.mask_files))
        self.input_size = input_size

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.files)

    def pil_load(self, path, is_input=True):
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')

    def __getitem__(self, idx):
        f_name = os.path.join(self.data_path, self.files[idx])
        m_name = os.path.join(self.mask_path, self.mask_files[idx])

        if os.path.exists(f_name) is False:
            raise Exception('Missing file with name ' + f_name + ' in dataset')

        input_ = self.pil_load(f_name)
        target = self.pil_load(m_name, False)

        input_ = self.to_tensor(input_)
        target = self.to_tensor(target)
        target = torch.sum(target, dim=0).unsqueeze(0)
        target[torch.gt(target, 0)] = 1

        return input_, target
