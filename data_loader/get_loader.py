from .mydataset import ImageFolder, ImageFilelist
from .unaligned_data_loader import UnalignedDataLoader
import os
import torch
from torch.utils.data import DataLoader


def get_loader_test(source_path, target_path, evaluation_path, transforms, batch_size=32):
    source_folder = ImageFolder(os.path.join(source_path),
                                transforms[source_path])
    target_folder_train = ImageFolder(os.path.join(target_path),
                                      transform=transforms[target_path],
                                      return_paths=False,train=True)
    eval_folder_test = ImageFilelist(os.path.join(evaluation_path),
                                     '/data/ugui0/ksaito/VISDA_tmp/image_list_val.txt',
                                     transform=transforms[evaluation_path],
                                     return_paths=True)

    train_loader = UnalignedDataLoader()
    train_loader.initialize(source_folder, target_folder_train, batch_size)

    test_loader = torch.utils.data.DataLoader(
        eval_folder_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4)

    return train_loader, test_loader
def get_loader(source_path, target_path, evaluation_path, transforms, batch_size=32):
    source_folder = ImageFolder(os.path.join(source_path),
                                transforms[source_path])
    target_folder_train = ImageFolder(os.path.join(target_path),
                                      transform=transforms[target_path],
                                      return_paths=False)
    eval_folder_test = ImageFolder(os.path.join(evaluation_path),
                                     transform=transforms[evaluation_path],
                                     return_paths=True)

    train_loader = UnalignedDataLoader()
    train_loader.initialize(source_folder, target_folder_train, batch_size)

    test_loader = torch.utils.data.DataLoader(
        eval_folder_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4)

    return train_loader, test_loader
