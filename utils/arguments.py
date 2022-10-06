import argparse

def get_args():
    parser = argparse.ArgumentParser(description='>>>> Lipschitz <<<<')
    # Arguments
    parser.add_argument('--seed', type=int, default=0, help='(default=%(default)d)', required=True)
    parser.add_argument('--experiment', default='fmnist', type=str, required=True,
                        choices=['fmnist',
                                 'mnist',
                                 'cifar10'], help='(default=%(default)s)')
    parser.add_argument('--arch', default='mlp', type=str, required=True,
                        choices=['mlp',
                                 'lenet',
                                 'alexnet',
                                 'resnet20'])
    parser.add_argument('--hidden_size', type=int, help='Hidden size of MLP', default=400)
    parser.add_argument('--num_layers', type=int, help='Num_layers of MLP', default=3)
    parser.add_argument('--regularization', default='none', type=str, required=True,
                        choices=['none',
                                 'dropout',
                                 'l1',
                                 'l2',
                                 'frobreg',
                                 'lcc',
                                 'bn'], help='(default=%(default)s)')
    parser.add_argument('--lamb', default='_', type=str, required=False, help='(default=%(default)s)')
    parser.add_argument('--optimizer', default='SGD', type=str, required=True,
                        choices=['SGD', 
                                 'Adam'], help='(default=%(default)s)')
    parser.add_argument('--criterion', default='CE', type=str, required=True,
                        choices=['CE',
                                 'MSE',
                                 'hingle_loss',
                                 'KL'], help='(default=%(default)s)')
    parser.add_argument('--max_epoch', default=500, type=int, required=True, help='(default=%(default)d)')
    parser.add_argument('--batch_size', default=128, type=int, required=True, help='(default=%(default)d)')
    parser.add_argument('--lr', default=0.01, type=float, required=True, 
                            help='(default=(default))')
    parser.add_argument('--interval', type=int, required=False, default=5,
                            help='Testing interval')
    parser.add_argument('--L_model_threshold', type=float, help='L_model threshold', default=0.0)
    parser.add_argument('--L_loss_threshold', type=float, help='L_loss threshold', default=0.0)
    parser.add_argument('--tunning', type=bool, help='Tunning mode or not?', default=False)
    args=parser.parse_args()
    return args