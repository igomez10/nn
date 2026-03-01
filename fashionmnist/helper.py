
# Define model
from kagglehub import datasets
from torch import nn
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
class fashionClassifierMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        # flatten the image into a 784 element vector
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train_loop(epochs: int, dataloader: DataLoader, model: nn.Module, lossFn: nn.Module, optimizer: torch.optim.Optimizer):
    for i in range(epochs):
        for image, expected_label in dataloader:
            optimizer.zero_grad()
            pred = model(image)

            loss = lossFn(pred, expected_label)
            loss.backward()

            optimizer.step()
            if i % 10 == 0:
                print("loss", loss.item(), end='\r')
        print("epoch", i, "loss", loss.item())



def get_data() -> tuple[datasets.FashionMNIST, datasets.FashionMNIST]:
  print("Downloading training data...")
  training_data = datasets.FashionMNIST(
      root="../data",
      train=True,
      download=True,
      transform=ToTensor(),
  )

  # Download test data from open datasets.
  print("Downloading test data...")
  test_data = datasets.FashionMNIST(
      root="../data",
      train=False,
      download=True,
      transform=ToTensor(),
  )

  return training_data, test_data
