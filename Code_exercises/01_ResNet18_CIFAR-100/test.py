import torch.cuda
from models.resnet18 import ResNet18
from src.trainer import Trainer

device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    model = ResNet18
    trainer = Trainer(
        model=model,
        num_classes=100,
        data_dir="./data/cifar-100-python/",
        meta_dir="./data/cifar-100-python/meta",
        batch_size=64,
        learning_rate=1e-2,
        weight_decay=0.01,
        num_epochs=150,
        device=device
    )
    trainer.load()
    trainer.run_test()
