import torch
import torchvision
import torchvision.transforms as T

IMAGE_SIZE=64
BATCH_SIZE=64

data = torchvision.datasets.MNIST(
    "../Datasets/MNIST_PyTorch",
    train=True,
    transform=T.Compose([
        T.Resize(IMAGE_SIZE),
        T.ToTensor()
    ]),
    download=True
)
dataloader = torch.utils.data.DataLoader(
    data,
    batch_size=BATCH_SIZE,
    shuffle=True
)
