"""Step 1: Load and normalize CIFAR10, show some training images."""
import torchvision

from common import batch_size, classes, get_loaders, imshow

if __name__ == '__main__':
    trainloader, testloader = get_loaders()
    print(f'Train samples: {len(trainloader.dataset)}, test samples: {len(testloader.dataset)}')

    # get some random training images
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    imshow(torchvision.utils.make_grid(images), 'train_samples.png')
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))
