"""Step 5: Test the network on the test data, then check the GPU device."""
import sys

import torch
import torchvision

from common import PATH, Net, classes, get_loaders, imshow

if __name__ == '__main__':
    model_path = sys.argv[1] if len(sys.argv) > 1 else PATH
    print('Testing model:', model_path)
    _, testloader = get_loaders()

    dataiter = iter(testloader)
    images, labels = next(dataiter)

    imshow(torchvision.utils.make_grid(images), 'test_samples.png')
    print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

    net = Net()
    net.load_state_dict(torch.load(model_path, weights_only=True, map_location='cpu'))

    outputs = net(images)
    _, predicted = torch.max(outputs, 1)
    print('Predicted:   ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(4)))

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

    # Training on GPU: check which device is available
    device = torch.device(torch.accelerator.current_accelerator().type
                          if torch.accelerator.is_available() else 'cpu')
    print('Device:', device)
