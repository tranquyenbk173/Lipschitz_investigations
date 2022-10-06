import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from sklearn.model_selection import train_test_split

def cifar_data_loader(batch_size):

    transform = transforms.Compose(
        [transforms.ToTensor(),
        #  transforms.Resize(70),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    bs = batch_size
    
    cifar_train = datasets.CIFAR10('./data/data_cifar10',train=True,transform=transform, download=True)
    
    cifar_train_batch = DataLoader(cifar_train,batch_size = bs,shuffle = True)
    
    cifar_test = datasets.CIFAR10('./data/data_cifar10',train=False,transform=transform, download=True) 
    
    cifar_test_batch = DataLoader(cifar_test,batch_size = bs,shuffle = False)
    
    return cifar_train_batch, cifar_test_batch
    

def fmnist_data_loader(batch_size):

    transform = transforms.Compose(
        [transforms.ToTensor(),
        #  transforms.Resize(70),
         transforms.Normalize((0.5,), (0.5,))])
    
    bs = batch_size
    
    fmnist_train = datasets.FashionMNIST('./data/data_fmnist',train=True,transform=transform, download=True)
    
    fmnist_train_batch = DataLoader(fmnist_train,batch_size = bs,shuffle = True)
    
    fmnist_test = datasets.FashionMNIST('./data/data_fmnist',train=False,transform=transform, download=True) 
    
    fmnist_test_batch = DataLoader(fmnist_test,batch_size = bs,shuffle = False)
    
    return fmnist_train_batch, fmnist_test_batch
    

def mnist_data_loader(batch_size):

    transform = transforms.Compose(
        [transforms.ToTensor(),
        #  transforms.Resize(70),
         transforms.Normalize((0.5,), (0.5,))])
    
    bs = batch_size
    
    fmnist_train = datasets.MNIST('./data/data_mnist',train=True,transform=transform, download=True)
    
    fmnist_train_batch = DataLoader(fmnist_train,batch_size = bs,shuffle = True)
    
    fmnist_test = datasets.MNIST('./data/data_mnist',train=False,transform=transform, download=True) 
    
    fmnist_test_batch = DataLoader(fmnist_test,batch_size = bs,shuffle = False)
    
    return fmnist_train_batch, fmnist_test_batch