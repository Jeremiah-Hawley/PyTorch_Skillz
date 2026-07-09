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
    def __init__():
        return None

    def forward():
        return None

# Main Function
def main(input_size: int, hidden_size: int, num_classes: int, num_epochs: int, batch_size: int, learning_rate: float) -> None:
    
    training_dataset = torchvision.datasets.MNIST(
        #Where to store the data
        root='./data',
        #Is is for training?
        train=True,
        # Transform the data
        transform=transforms.ToTensor(),
        #Download flag
        download=True)

    test_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(
        #what dataset should the loader use?
        dataset=training_dataset,
        #batch size during training?
        batch_size=batch_size,
        #should it shuffle (randomize positions) of the data?
        shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False)

    examples = iter(train_loader)
    samples, labels = next(examples)

    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.imshow(samples[i][0], cmap='gray')

    plt.show()

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
    main(784, 100, 10, 2, 500, 0.001)

