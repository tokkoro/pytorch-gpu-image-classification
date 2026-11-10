"""Steps 2-4: Define the CNN, loss function and optimizer, train, and save the model."""
import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim

from common import MODEL_FILENAME, Net, batch_size, get_loaders, log_metadata, output_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='auto',
                        help="'cpu', 'cuda', ... (default: 'auto' = GPU if available)")
    parser.add_argument('--model_path', default=None,
                        help=f'where to save the model (default: {MODEL_FILENAME} in the outputs directory)')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--batch_size', type=int, default=batch_size)
    parser.add_argument('--num_workers', type=int, default=2)
    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device(torch.accelerator.current_accelerator().type
                              if torch.accelerator.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    model_path = args.model_path or output_path(MODEL_FILENAME)
    print('Training on device:', device)

    trainloader, _ = get_loaders(args.batch_size, args.num_workers)

    # 2. Define a Convolutional Neural Network
    net = Net()
    net.to(device)
    print(net)

    # 3. Define a Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=args.momentum)

    # 4. Train the network
    start = time.perf_counter()
    for epoch in range(args.epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            epoch_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}', flush=True)
                log_metadata(epoch=epoch + 1, batch=i + 1, loss=running_loss / 2000)
                running_loss = 0.0
        log_metadata(epoch=epoch + 1, epoch_loss=epoch_loss / len(trainloader))

    train_time = time.perf_counter() - start
    print(f'Finished Training in {train_time:.1f} s')
    log_metadata(train_time_s=round(train_time, 1), device=str(device))

    torch.save(net.state_dict(), model_path)
    print(f'Saved model to {model_path}')
