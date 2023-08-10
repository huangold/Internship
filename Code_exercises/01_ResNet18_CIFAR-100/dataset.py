import torch
from torch.utils.data import Dataset
from docset import DocSet
from imgaug import augmenters as iaa
import io
import numpy as np
import cv2 as cv


class CIFAR100Dataset(Dataset):
    def __init__(
            self,
            data_dir: str,
            train: bool = True
    ) -> None:
        super(CIFAR100Dataset, self).__init__()
        self.docs = DocSet(data_dir, 'r')
        self.aug = iaa.Sequential([
            iaa.Pad(4, keep_size=False),
            iaa.Fliplr(0.5),
            iaa.CropToFixedSize(32, 32),
        ]) if train else iaa.Identity()

    def __getitem__(self, index: int) -> dict:
        doc = self.docs[index]
        image = cv.imdecode(np.frombuffer(doc["image"], dtype=np.uint8), cv.IMREAD_COLOR)[:, :, ::-1]
        image = self.aug(image=image)
        label = doc["label"]
        return {'image': image, 'label': label}

    def __len__(self) -> int:
        return len(self.docs)


dataset = CIFAR100Dataset("./data/cifar-100-python/train.ds", True)
image = dataset[0]["image"]
image = cv.resize(image, (image.shape[0] * 10, image.shape[1] * 10), interpolation=cv.INTER_LINEAR)
cv.imshow("1", image)
cv.waitKey()