import torch
from torch.utils.data import Dataset
import pickle
import cv2 as cv
import numpy as np


class CIFAR100Dataset(Dataset):
    def __init__(
            self,
            data_dir: str,
            meta_dir: str,
            train: bool = True
    ) -> None:
        super(CIFAR100Dataset, self).__init__()
        self.data_type = "train" if train else "test"
        with open(data_dir + self.data_type, "rb") as fo:
            data = pickle.load(fo, encoding="bytes")
            self.images = list(data.values())[4]
            self.fine_labels = list(data.values())[2]
            # self.coarse_labels = list(data.values())[3]
        with open(meta_dir, "rb") as fo:
            meta = pickle.load(fo, encoding="bytes")
            self.fine_labels_name = list(meta.values())[0]
            # self.coarse_labels_name = list(meta.values())[1]

    def __getitem__(self, index: int) -> dict:
        return {
            "image": torch.tensor(self.images[index]).resize(3, 32, 32),
            "fine_label": self.fine_labels[index],
            # "coarse_label": self.coarse_labels,
            "fine_label_name": self.fine_labels_name[self.fine_labels[index]],
            # "coarse_label_name": self.coarse_labels_name[self.coarse_labels[index]]
        }

    def __len__(self) -> int:
        return len(self.images)


dataset = CIFAR100Dataset(
    data_dir="./data/cifar-100-python/",
    meta_dir="./data/cifar-100-python/meta",
    train=True
)

image_num = 0
scale = 20
label = str(dataset[image_num]["fine_label_name"])
img = torch.permute(torch.tensor(dataset[image_num]["image"]).resize(3, 32, 32), [1, 2, 0]).numpy()[:, :, ::-1]
img = cv.resize(img, (img.shape[0] * scale, img.shape[1] * scale), interpolation=cv.INTER_LINEAR)
cv.imshow(label, np.uint8(img))
cv.waitKey()
