
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


def train_loop(epochs: int, dataloader: DataLoader, model: nn.Module, lossFn: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device):
    model.train()
    for i in range(epochs):
        for image, expected_label in dataloader:
            image, expected_label = image.to(device), expected_label.to(device)
            optimizer.zero_grad()
            pred = model(image)

            loss = lossFn(pred, expected_label)
            loss.backward()

            optimizer.step()
            if i % 10 == 0:
                print("epoch", i, "loss", loss.item(), end='\r')
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

class fashionClassifierCNN(nn.Module):
#  Determine what layers and their order in CNN object 
    def __init__(self, num_classes, in_channels=3):
        super().__init__()
        self.conv_layer1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3)
        self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.max_pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.conv_layer3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv_layer4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.max_pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.fc1 = nn.Linear(1024, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
    
    # Progresses data across layers    
    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.max_pool1(out)
        
        out = self.conv_layer3(out)
        out = self.conv_layer4(out)
        out = self.max_pool2(out)
                
        out = out.reshape(out.size(0), -1)
        
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out

def evaluateWithMax(model: nn.Module, dataloader: DataLoader, device: torch.device, ):
    model.eval()
    num_correct = 0
    num_samples = 0
    accuracy = 0
    with torch.inference_mode():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            _, predicted = torch.max(pred, 1) # get the index of the max log-probability
            num_correct += (predicted == y).sum()
            num_samples += predicted.size(0)



    accuracy = float(num_correct) / float(num_samples)
    print(f"Got {num_correct} / {num_samples} with accuracy {accuracy:.2f}") 


