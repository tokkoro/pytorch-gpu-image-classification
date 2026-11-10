"""Step 1: Load and normalize CIFAR10, show some training images."""
import argparse

import torchvision

from common import batch_size, classes, get_loaders, imshow, log_metadata, output_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch_size', type=int, default=batch_size)
    parser.add_argument('--num_workers', type=int, default=2)
    args = parser.parse_args()

    trainloader, testloader = get_loaders(args.batch_size, args.num_workers)
    print(f'Train samples: {len(trainloader.dataset)}, test samples: {len(testloader.dataset)}')
    log_metadata(train_samples=len(trainloader.dataset), test_samples=len(testloader.dataset))

    # get some random training images
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    imshow(torchvision.utils.make_grid(images), output_path('train_samples.png'))
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(len(labels))))
