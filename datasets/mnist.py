from torchvision import datasets, transforms

def get_mnist_datasets(data_dir='datasets/data', train = True):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load the training dataset
    dataset = datasets.MNIST(
        root=data_dir,
        train=train,
        download=True,
        transform=transform
    )
    return dataset

if __name__ == "__main__":
    mnist_dataset = get_mnist_datasets()