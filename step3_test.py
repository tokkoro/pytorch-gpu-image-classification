"""Step 5: Test the network on the test data, then check the GPU device."""
import argparse

import torch
import torchvision

from common import (PATH, Net, batch_size, classes, get_loaders, imshow, input_dir,
                    log_metadata, output_path)


def default_model_path():
    """On Valohai, use the file in the `model` input; locally, ./cifar_net.pt."""
    model_input = input_dir('model')
    if model_input is not None and model_input.is_dir():
        models = sorted(p for p in model_input.rglob('*') if p.is_file())
        if models:
            return str(models[0])
    return PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model_path', default=None,
                        help='model to evaluate (default: Valohai input "model", or ./cifar_net.pt)')
    parser.add_argument('--batch_size', type=int, default=batch_size)
    parser.add_argument('--num_workers', type=int, default=2)
    args = parser.parse_args()

    model_path = args.model_path or default_model_path()
    print('Testing model:', model_path)
    _, testloader = get_loaders(args.batch_size, args.num_workers)

    dataiter = iter(testloader)
    images, labels = next(dataiter)

    imshow(torchvision.utils.make_grid(images), output_path('test_samples.png'))
    print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(len(labels))))

    net = Net()
    net.load_state_dict(torch.load(model_path, weights_only=True, map_location='cpu'))

    outputs = net(images)
    _, predicted = torch.max(outputs, 1)
    print('Predicted:   ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(len(labels))))

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the {total} test images: {100 * correct // total} %')

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

    class_accuracy = {}
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        class_accuracy[f'accuracy_{classname}'] = round(accuracy, 1)
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

    log_metadata(accuracy=round(100 * correct / total, 2), **class_accuracy)

    # Training on GPU: check which device is available
    device = torch.device(torch.accelerator.current_accelerator().type
                          if torch.accelerator.is_available() else 'cpu')
    print('Device:', device)
