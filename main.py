import comet_ml
from comet_ml import Experiment
from utils.manager import Manager
from torch.utils.data import DataLoader
from utils.arguments import *
import torch.nn as nn
import numpy as np
import random
import torch
import sys

args =  get_args()

print('=' * 100)
print('Arguments =')
for arg in vars(args):
    print('\t' + arg + ':', getattr(args, arg))
print('=' * 100)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

###############################################---------------------------------------------

# Seed
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
else:
    print('[CUDA unavailable]'); sys.exit()
    
# Experiment
if args.experiment == 'fmnist':
    from utils.datasets import fmnist_data_loader as dataloader
elif args.experiment == 'mnist':
    from utils.datasets import mnist_data_loader as dataloader
elif args.experiment == 'cifar10':
    from utils.datasets import cifar_data_loader as dataloader

train_loader, test_loader = dataloader(batch_size=args.batch_size)
if args.regularization == 'none':
    args.tunning = False

# print(args.tunning)
if args.tunning == True:
    # print(len(train_loader.dataset))
    train_set, valid_set = torch.utils.data.random_split(train_loader.dataset, (len(train_loader.dataset)-10000, 10000), generator=torch.Generator().manual_seed(args.seed)) ### return 2 Dataset objects
    train_loader = DataLoader(train_set,batch_size = args.batch_size,shuffle = True)
    test_loader = DataLoader(valid_set,batch_size = args.batch_size,shuffle = False)
    

# Network
if args.experiment == 'fmnist':
    if args.arch == 'mlp':
        from models.fmnist_models import MLP as network
    elif args.arch == 'lenet':
        from models.fmnist_models import LeNett as network
    elif args.arch == 'alexnet':
        from models.fmnist_models import AlexNett as network
    elif args.arch == 'resnet20':
        from models.fmnist_models import ResNet20 as network
        
elif args.experiment == 'cifar10':
    if args.arch == 'mlp':
        from models.cifar_models import MLP as network
    elif args.arch == 'lenet':
        from models.cifar_models import LeNett as network
    elif args.arch == 'alexnet':
        from models.cifar_models import AlexNett as network
    elif args.arch == 'resnet20':
        from models.cifar_models import ResNet20 as network
        
if args.experiment == 'mnist':
    if args.arch == 'mlp':
        from models.fmnist_models import MLP as network
    elif args.arch == 'lenet':
        from models.fmnist_models import LeNett as network
    elif args.arch == 'alexnet':
        from models.fmnist_models import AlexNett as network
    elif args.arch == 'resnet20':
        from models.fmnist_models import ResNet20 as network

if args.lamb != '':
    lamb = float(args.lamb)
else:
    lamb = ''

if args.arch != 'mlp' or 'mnist' not in args.experiment:    
    model = network(num_classes=10, reg = args.regularization, p=lamb).to(device)
else:
    model = network(num_classes=10, reg = args.regularization, p=lamb, hidden_size=args.hidden_size, num_layers=args.num_layers).to(device)

# Optimizer and Criterion:
if args.optimizer == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                        momentum=0.9)
elif args.optimizer == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

is_mse = False
if args.criterion == 'CE':
    criterion = nn.CrossEntropyLoss()
elif args.criterion == 'MSE':
    criterion = nn.MSELoss()
    is_mse = True
elif args.criterion == 'hingle_loss':
    criterion = nn.MultiMarginLoss()
    # is_mse = True
elif args.criterion == 'KL':
    criterion = nn.KLDivLoss(reduction='batchmean')
    is_mse = True
    
# Others:
maxEpoch = args.max_epoch
lr = args.lr
bs = args.batch_size


    
L_model_thres = args.L_model_threshold
L_loss_thres = args.L_loss_threshold
    
    
name_exp = '{}_{}_{}_{}_{}_{}_{}_{}'.format(args.experiment, args.arch, args.regularization, args.criterion, lamb, maxEpoch, args.seed, lr)
if args.tunning:
    name_exp += '_tune'

if args.arch == 'mlp' and args.arch == 'fmnist':
    hidden_size = model.hidden_size
    num_layers = model.num_layers
    name_exp += 'hiddensize' + str(hidden_size) + '_numlayers' + str(num_layers)
    
manager = Manager(model, criterion, optimizer, name_exp, lr, maxEpoch, bs, train_loader, test_loader, is_mse, args.regularization, lamb, args.interval, L_model_thres, L_loss_thres, args.tunning)

manager.run()