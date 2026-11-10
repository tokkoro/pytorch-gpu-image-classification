"""Steps 2-4: Define the CNN, loss function and optimizer, train, and save the model."""
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim

from common import PATH, Net, get_loaders

if __name__ == '__main__':
    # Optional args: device (default: GPU if available) and model output path
    device = torch.device(sys.argv[1] if len(sys.argv) > 1 else
                          (torch.accelerator.current_accelerator().type
                           if torch.accelerator.is_available() else 'cpu'))
    model_path = sys.argv[2] if len(sys.argv) > 2 else PATH
    print('Training on device:', device)

    trainloader, _ = get_loaders()

    # 2. Define a Convolutional Neural Network
    net = Net()
    net.to(device)
    print(net)

    # 3. Define a Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # 4. Train the network
    start = time.perf_counter()
    for epoch in range(2):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}', flush=True)
                running_loss = 0.0

    print(f'Finished Training in {time.perf_counter() - start:.1f} s')

    torch.save(net.state_dict(), model_path)
    print(f'Saved model to {model_path}')
