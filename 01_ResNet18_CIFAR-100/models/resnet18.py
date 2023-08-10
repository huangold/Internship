from models.basic_block import *


class ResNet18(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super(ResNet18, self).__init__()
        self.in_ch = 64
        self.conv = nn.Sequential(
            nn.Conv2d(3, self.in_ch, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.in_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = self.make_layer(out_ch=64, num_blocks=2, stride_1st_block=1)
        self.layer2 = self.make_layer(out_ch=128, num_blocks=2, stride_1st_block=2)
        self.layer3 = self.make_layer(out_ch=256, num_blocks=2, stride_1st_block=2)
        self.layer4 = self.make_layer(out_ch=512, num_blocks=2, stride_1st_block=2)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, out_ch: int, num_blocks: int, stride_1st_block: int) -> nn.Sequential:
        strides = [stride_1st_block] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_ch, out_ch, stride))
            self.in_ch = out_ch
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.shape[0], -1)
        out = self.fc(out)
        return out
