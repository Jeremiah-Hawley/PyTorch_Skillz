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
        #initialize variables and function pointers
        return None

    def forward(self, x):
        #write forward pass
        return None

# Main Function
def main(input_size: int, hidden_size: int, num_classes: int, num_epochs: int, batch_size: int, learning_rate: float) -> None:
    # 0) Data Collection
    train_dataset = None
    test_dataset = None

    train_loader = None
    test_loader = None

    # 1) Design Model
    model = NeuralNet(None).to(None)

    # 2) Loss and optimizer
    criterion = None
    optimizer = None

    # 3) Training Loop
    n_steps = None
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):  
            #train the model in this loop
            print("oops!")
        
            # CRITERION GOES HERE
            loss = None

            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_steps}], Loss: {loss.item():.4f}')            

    # 4) Testing Loop
    print("\n === Testing Now === \n")
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for images, labels in test_loader:
            # your code here
            print(None)

        acc = 100.0 * n_correct / n_samples        
        print(f'accuracy = {acc}')



    return None

if __name__ == '__main__':
    """ Default Hyper Parameters
        input_size = 784 (28x28)
        * hidden_size = 500 
        num_classes = 10
        * num_epochs = 2 
        * batch_size = 100
        * learning_rate = 0.001
        (feel free to change ones marked with "*" as you wish)
    """
    main(784, 500, 10, 2, 100, 0.001)

