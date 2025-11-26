from torchvision import datasets
# datasets.CIFAR10(root="/home/structlearning/mayukh.mondal/projects/deep_double_descent/data", train=True, download=True)
# datasets.CIFAR10(root="/home/structlearning/mayukh.mondal/projects/deep_double_descent/data", train=False, download=True)
root="/home/structlearning/mayukh.mondal/projects/deep_double_descent/data"
datasets.MNIST(root=root, train=True, download=True)
datasets.MNIST(root=root, train=False, download=True)
print("Done. Data is in:", root)