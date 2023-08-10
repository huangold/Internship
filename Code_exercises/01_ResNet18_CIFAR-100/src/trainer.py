import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataset_test import CIFAR100Dataset
# from datetime import datetime

class Trainer:
    def __init__(
            self,
            model,
            num_classes: int,
            data_dir: str,
            meta_dir: str,
            batch_size: int,
            learning_rate: float = 0.001,
            weight_decay: float = 0.001,
            num_epochs: int = 10,
            device: str = "cuda"
    ) -> None:
        self.device = device
        self.num_classes = num_classes
        self.model = model(self.num_classes).to(self.device)
        self.save_dir = "./ResNet18.pt"
        if data_dir:
            train_data_set = CIFAR100Dataset(data_dir, meta_dir, train=True)
            self.train_data_length = len(train_data_set)
            self.train_loader = DataLoader(
                train_data_set,
                batch_size=batch_size,
                shuffle=True
            )
            test_data_set = CIFAR100Dataset(data_dir, meta_dir, train=False)
            self.test_data_length = len(test_data_set)
            self.test_loader = DataLoader(
                test_data_set,
                batch_size=batch_size,
                shuffle=True
            )
        self.num_epochs = num_epochs
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()

    def train(self, images: torch.Tensor, labels: torch.Tensor) -> tuple:
        images, labels = images.to(self.device), labels.to(self.device)
        images = images.float()
        labels = F.one_hot(labels, self.num_classes).float()
        output = self.model(images)
        loss = self.criterion(output, labels)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        output = output.cpu()
        loss = loss.cpu()
        return output, loss

    def run_train(self) -> None:
        print("--训练开始--")
        for epoch in range(self.num_epochs):
            self.model.train()
            total_train_loss = 0
            total_train_acc = 0
            for i, batch_data in enumerate(self.train_loader):
                images = batch_data["image"]
                labels = batch_data["fine_label"]
                output, loss = self.train(images, labels)
                total_train_loss += loss.data.item()
                preds = torch.max(output, 1)[1].numpy()
                train_acc = (preds == labels.data.cpu().numpy()).astype(int).sum()
                total_train_acc += train_acc
            print("第{}轮，训练集loss：{}，准确率：{:.2f}%".format(
                epoch + 1,
                total_train_loss / self.train_data_length,
                100 * total_train_acc / self.train_data_length
            ))
            self.run_test()

    def predict(self, images: torch.Tensor, labels: torch.Tensor) -> tuple:
        with torch.no_grad():
            images, labels = images.to(self.device), labels.to(self.device)
            images = images.float()
            labels = F.one_hot(labels, self.num_classes).float()
            output = self.model(images)
            loss = self.criterion(output, labels)
            output = output.cpu()
            loss = loss.cpu()
            return output, loss

    def run_test(self) -> None:
        print("--测试开始--")
        self.model.eval()
        total_test_loss = 0
        total_test_acc = 0
        for i, batch_data in enumerate(self.test_loader):
            images = batch_data["image"]
            labels = batch_data["fine_label"]
            output, loss = self.predict(images, labels)
            total_test_loss += loss.data.item()
            preds = torch.max(output, 1)[1].numpy()
            test_acc = (preds == labels.data.cpu().numpy()).astype(int).sum()
            total_test_acc += test_acc
        print("测试集loss：{}，准确率：{:.2f}%".format(
            total_test_loss / self.test_data_length,
            100 * total_test_acc / self.test_data_length
        ))

    def save(self) -> None:
        # present_time = datetime.fromtimestamp(int(datetime.timestamp(datetime.now())))
        # self.save_dir = "./ResNet18-" + str(present_time).replace(' ', '-').replace(':', '') + ".pt"
        torch.save(
            self.model.state_dict(),
            self.save_dir
        )

    def load(self) -> None:
        # if self.save_dir:
        state_dict = torch.load(self.save_dir)
        self.model.load_state_dict(state_dict)
