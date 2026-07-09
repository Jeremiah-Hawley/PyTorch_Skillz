""" Pipeline steps
0) Data Collection
1) Design Model (input, output size, forward pass)
2) Construct loss and optimizer
3) Training loop
    - forward pass: copmute prediction
    - backward pass: gradients 
    - update weights
"""

#torch
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
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        # no activation or softmax - attached to CrossEntropyLoss
        return out

# Main Function
def main(input_size: int, hidden_size: int, num_classes: int, num_epochs: int, batch_size: int, learning_rate: float) -> None:
    # 0) Dataset
    train_dataset = torchvision.datasets.MNIST(root='./data', 
                                                train=True, 
                                                transform=transforms.ToTensor(),  
                                                download=True)

    test_dataset = torchvision.datasets.MNIST(root='./data', 
                                                train=False, 
                                                transform=transforms.ToTensor())

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                                batch_size=batch_size, 
                                                shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                                batch_size=batch_size, 
                                                shuffle=False)

    examples = iter(test_loader)
    example_data, example_targets = next(examples)

    # 1) Design Model
    model = NeuralNet(input_size, hidden_size, num_classes).to(device)

    # 2) Loss and Optimizer
    criterion = nn.CrossEntropyLoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    #3) Training loop
    n_steps = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):  
            # image tensor shape: 100, 1, 28, 28
            images = images.reshape(-1, 28*28).to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass + optimizer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i+1) % 100 == 0:
                print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_steps}], Loss: {loss.item():.4f}')            
    print("\n === Testing Now === \n")
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for images, labels in test_loader:
            images = images.reshape(-1, 28*28).to(device)
            labels = labels.to(device)
            outputs = model(images)
            # max returns (value ,index)
            _, predicted = torch.max(outputs.data, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

        acc = 100.0 * n_correct / n_samples        
        print(f'accuracy = {acc}')


if __name__ == '__main__':
    """ Default Hyper Parameters
        input_size = 784 (28x28)
        * hidden_size = 500 
        num_classes = 10
        * num_epochs = 2 
        * batch_size = 100
        * learning_rate = 0.001
    """
    main(784, 500, 10, 2, 100, 0.001)

