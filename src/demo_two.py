""" Pipeline steps
0) Data Collection
1) Design Model (input, output size, forward pass)
2) Construct loss and optimizer
3) Training loop
    - forward pass: copmute prediction
    - backward pass: gradients 
    - update weights
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Neural Network
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

# Main Function
def main(input_size: int, hidden_size: int, num_classes: int, num_epochs: int, batch_size: int, learning_rate: float) -> None:
    # MINIST
    training_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        transform=transforms.ToTensor(),
        download=True)
    test_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(
        dataset=training_dataset,
        batch_size=batch_size,
        shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False)

    examples = iter(train_loader)
    samples, labels = next(examples)

    model = NeuralNet(input_size, hidden_size, num_classes)

    print(type(model))

    return None

if __name__ == '__main__':
    """ Default Hyper Parameters
        input_size = 784 (28x28)
        * hidden_size = 500 
        num_classes = 10
        * num_epochs = 2 
        * batch_size = 100
        * learning_rate = 0.001
    """
    main(784, 100, 10, 2, 100, 0.001) 

