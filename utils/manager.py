import comet_ml
from comet_ml import Experiment
API_KEY = 'SapPaHUFNrFE9hfqfYZwx3sQp'
PROJ_NAME = 'chay-bo-sung-lipschitz'
WORKSPACE = 'tranquyenbk173'

from tqdm import tqdm
import torch
from utils import estimate_L_loss, estimate_L_model
import pandas as pd
import time
import os
import pickle
import math

device = "cuda" if torch.cuda.is_available() else "cpu"

class Manager:
    
    def __init__(self, model, criterion, optimizer, name_exp, lr, max_epoch, bs, train_loader, test_loader, is_mse, regularization, lamb, interval, L_model_thres, L_loss_thres, tunning):
        self.net = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.name_exp = name_exp
        self.lr = lr
        self.bs = bs
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.max_epoch = max_epoch
        self.is_mse = is_mse
        self.reg = regularization
        self.lamb = lamb
        self.scheduler = None
        self.interval = interval
        self.L_model_thres = L_model_thres
        self.L_loss_thres = L_loss_thres
        self.tunning = tunning
    
    def run(self):
        
        max_epoch = self.max_epoch
        epoch_i = -1
        self.net = self.net.to(device)
        print(self.net)
        if device == 'cuda':
              self.net = torch.nn.DataParallel(self.net)
              torch.backends.cudnn.benchmark = True
        
        # Check to see if there is a key in environment:
        key_name_path = './exp_key_and_name1801.pkl'
        key_name = None
        EXPERIMENT_KEY = None
        if os.path.exists(key_name_path):
            # print('>>>>>>>>file_exist')
            # try:
            key_name = pickle.load(open(key_name_path, 'rb'))
            if self.name_exp in key_name.keys():
                EXPERIMENT_KEY = key_name[str(self.name_exp)]
                print('EXP key_: ', EXPERIMENT_KEY)
            else:
                print('This is the first time to run for this experiment!')
        
        # First, let's see if we continue or start a fresh:
        CONTINUE_RUN = False
        if (EXPERIMENT_KEY is not None):
            # There is one, but the experiment might not exist yet:
            api = comet_ml.API(API_KEY) # Assumes API key is set in config/env
            try:
                api_experiment = api.get_experiment_by_key(EXPERIMENT_KEY)
            except Exception:
                api_experiment = None
                print('Fail to connect to CometML')
            if api_experiment is not None:
                CONTINUE_RUN = True
                # We can get the last details logged here, if logged:
                try:
                    print(api_experiment.get_parameters_summary())
                    step = int(api_experiment.get_parameters_summary("curr_step")["valueCurrent"])
                    # epoch = int(api_experiment.get_parameters_summary("curr_epoch")["valueCurrent"])
                except:
                    step = 0
        
        if CONTINUE_RUN:
            print('>>>> CONTINUE FROM A EXP <<<<')
            # 1. Recreate the state of ML system before creating experiment
            # otherwise it could try to log params, graph, etc. again
            PATH = './checkpoints/' + self.name_exp
            
            if os.path.exists(PATH):
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max_epoch-epoch_i-1)
                checkpoint = torch.load(PATH)
                self.net.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                epoch_i = checkpoint['epoch']
                scheduler.load_state_dict(checkpoint['scheduler'])
                print('Load checkpoint done!!!')
            else: 
                print('>>>> STARTING A FRESH EXP <<<< (No checkpoint available!!!)')
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max_epoch-epoch_i-1)
            
            # 2. Setup the existing experiment to carry on:
            experiment = comet_ml.ExistingExperiment(
                api_key=API_KEY,
                project_name=PROJ_NAME, workspace=WORKSPACE, log_code=True,
                previous_experiment=EXPERIMENT_KEY,
                log_env_details=True, # to continue env logging
                log_env_gpu=True,     # to continue GPU logging
                # log_env_cpu=True,     # to continue CPU logging
            )
            experiment.set_name(self.name_exp)
            temp = self.name_exp.split('_')
            exp, arch, reg, cri = temp[0], temp[1], temp[2], temp[3]
            experiment.add_tag(exp)
            experiment.add_tag(arch)
            experiment.add_tag(reg)
            experiment.add_tag(cri)
            experiment.add_tag('Change data')
            if 'hidden' in self.name_exp:
                hidden_size = temp[-2].split('hiddensize')[-1]
                num_layers =  temp[-1].split('numlayers')[-1]
                ss = str(hidden_size)+ '_' + str(num_layers)
                experiment.add_tag(ss)
            if self.tunning:
                experiment.add_tag('Tunning')
            
            if self.L_model_thres + self.L_loss_thres == 0:
                experiment.add_tag('Ques_2')
            elif self.L_model_thres * self.L_loss_thres != 0:
                print('L_model thres and L_loss thres must not equal to 0!!!')
                return 0
            elif self.L_model_thres != 0:
                experiment.add_tag('Ques_3')
            elif self.L_loss_thres != 0:
                experiment.add_tag('Ques_5')
                
            # experiment.add_tag('Lipschitzz')
            experiment.log_parameter('name_exp', self.name_exp)
            experiment.log_parameter("batch_size", self.bs)
            experiment.log_parameter("learning_rate", self.lr)
            
            # Retrieved from above APIExperiment
            experiment.set_step(step)
            experiment.set_epoch(epoch_i)
            
            # 3. Restore log
            if os.path.exists('./log/' + self.name_exp + '.csv'):
                Log = pd.read_csv('./log/' + self.name_exp + '.csv')
                Train_Loss = Log['Train_Loss'].tolist()
                Train_Acc = Log['Train_Acc'].tolist()
                Test_Loss = Log['Test_Loss'].tolist()
                Test_Acc = Log['Test_Acc'].tolist()
                L_model = Log['L_model'].tolist()
                L_loss = Log['L_loss'].tolist()
                L_model_train = Log['L_model_train'].tolist()
                L_loss_train = Log['L_loss_train'].tolist()
                L_model_test = Log['L_model_test'].tolist()
                L_loss_test = Log['L_loss_test'].tolist()
                Time_L = Log['Time_L'].tolist()
                Train_Time = Log['Train_Time'].tolist()
                Test_Time = Log['Test_Time'].tolist()
                Epoch = Log['Epoch'].tolist()
                Acc_Gap = Log['Acc_Gap'].tolist()
                Loss_Gap = Log['Loss_Gap'].tolist()
            else:

                Train_Loss = []
                Train_Acc = []
                Test_Loss = []
                Test_Acc = []
                L_model = []
                L_loss = []
                L_model_train = []
                L_loss_train = []
                L_model_test = []
                L_loss_test = []
                Time_L = []
                Train_Time = []
                Test_Time = []
                Epoch = []
                Acc_Gap = []
                Loss_Gap = []
            
        else:
            print('>>>> STARTING A FRESH EXP <<<<')
            # 1. Create the experiment first
            #    This will use the COMET_EXPERIMENT_KEY if defined in env.
            #    Otherwise, you could manually set it here. If you don't
            #    set COMET_EXPERIMENT_KEY, the experiment will get a
            #    random key!
            experiment = Experiment(api_key=API_KEY,
                                project_name=PROJ_NAME, workspace=WORKSPACE, log_code=True)
        
            experiment.set_name(self.name_exp)

            temp = self.name_exp.split('_')
            exp, arch, reg, cri = temp[0], temp[1], temp[2], temp[3]
            experiment.add_tag(exp)
            experiment.add_tag(arch)
            experiment.add_tag(reg)
            experiment.add_tag(cri)
            experiment.add_tag('Change data')
            if 'hidden' in self.name_exp:
                hidden_size = temp[-2].split('hiddensize')[-1]
                num_layers =  temp[-1].split('numlayers')[-1]
                ss = str(hidden_size)+ '_' + str(num_layers)
                experiment.add_tag(ss)
                
            if self.tunning:
                experiment.add_tag('Tunning')
            
            if self.L_model_thres + self.L_loss_thres == 0:
                experiment.add_tag('Ques_2')
            elif self.L_model_thres * self.L_loss_thres != 0:
                print('L_model thres and L_loss thres must not equal to 0!!!')
                return 0
            elif self.L_model_thres != 0:
                experiment.add_tag('Ques_3')
            elif self.L_loss_thres != 0:
                experiment.add_tag('Ques_5')
                
            # experiment.add_tag('Lipschitzz')
            experiment.log_parameter('name_exp', self.name_exp)
            experiment.log_parameter("batch_size", self.bs)
            experiment.log_parameter("learning_rate", self.lr)
            
            Train_Loss = []
            Train_Acc = []
            Test_Loss = []
            Test_Acc = []
            L_model = []
            L_loss = []
            L_model_train = []
            L_loss_train = []
            L_model_test = []
            L_loss_test = []
            Time_L = []
            Train_Time = []
            Test_Time = []
            Epoch = []
            Acc_Gap = []
            Loss_Gap = []
            
            # 2. Setup the state of the ML system
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max_epoch-epoch_i-1)
        
        # Train or continue training
        
        #--- update key and name
        key_name_path = './exp_key_and_name1801.pkl'
        key_name = None
        if os.path.exists(key_name_path):
            key_name = pickle.load(open(key_name_path, 'rb'))
            if self.name_exp not in key_name.keys():
                key_name[str(self.name_exp)] = experiment.get_key() #Get exp key from exp name @@@@
        else:
            key_name = dict()
            key_name[str(self.name_exp)] = experiment.get_key()
        pickle.dump(key_name, open(key_name_path, 'wb'))
        
        #--- Setup and train
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max_epoch-epoch_i-1)
            
        for epoch in range(epoch_i+1, max_epoch):
                train_loss, train_acc, train_L_model, train_L_loss, train_time_L, train_time_all= self.train(self.net, epoch, self.train_loader, self.criterion, self.optimizer, self.name_exp, self.lamb, self.max_epoch)
                
                
                if epoch % self.interval == 0: 
                    time_L = train_time_L
                
                    ### Comet logging...
                    experiment.log_metric("Train_loss", train_loss, epoch=epoch)
                    experiment.log_metric("Train_acc", train_acc, epoch=epoch)
                    experiment.log_metric('L_model', train_L_model)
                    experiment.log_metric('L_loss', train_L_loss)
                    
                    experiment.log_metric('Train_time', train_time_all)
                
                    ### Pandas logging...
                    Train_Loss.append(train_loss)
                    Train_Acc.append(train_acc)
                    L_model.append(train_L_model)
                    L_loss.append(train_L_loss)
                    
                    Train_Time.append(train_time_all)
                    
                    
            
                    ###------------------------------------------------------------------------------
                        
                    test_loss, test_acc, L_model_train_, L_loss_train_, L_model_test_, L_loss_test_, test_time_L, test_time_all = self.test(self.net, epoch, self.test_loader, self.criterion, self.max_epoch)
                    
                    
                    ### Comet logging...
                    experiment.log_metric("Test_loss", test_loss, epoch=epoch)
                    experiment.log_metric("Test_acc", test_acc, epoch=epoch)
                    experiment.log_metric('Test_time', test_time_all)
                    experiment.log_metric('L_model_train', L_model_train_)
                    experiment.log_metric('L_loss_train', L_loss_train_)
                    experiment.log_metric('L_model_test', L_model_test_)
                    experiment.log_metric('L_loss_test', L_loss_test_)
                    experiment.log_metric('L_time', time_L + test_time_L)
                    acc_gap = abs(train_acc - test_acc)
                    loss_gap = abs(train_loss - test_loss)
                    experiment.log_metric('Acc_Gap', acc_gap)
                    experiment.log_metric('Loss_Gap', loss_gap)
                
                    ### Pandas logging...
                    Test_Loss.append(test_loss)
                    Test_Acc.append(test_acc)
                    Test_Time.append(test_time_all)
                    Epoch.append(epoch)
                    L_model_train.append(L_model_train_)
                    L_loss_train.append(L_loss_train_)
                    L_model_test.append(L_model_test_)
                    L_loss_test.append(L_loss_test_)
                    Time_L.append(time_L + test_time_L)
                    Acc_Gap.append(acc_gap)
                    Loss_Gap.append(loss_gap)
                    
                    
                    
                
                    ### Save checkpoint:
                    PATH = './checkpoints/' + self.name_exp
                    torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.net.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                    }, PATH)
                
                    ### Save log per epoch @@
                    Log = pd.DataFrame()
                    Log['Epoch'] = Epoch
                    Log['Train_Loss'] = Train_Loss
                    Log['Train_Acc'] = Train_Acc
                    Log['Test_Loss'] = Test_Loss
                    Log['Test_Acc'] = Test_Acc
                    Log['L_model'] = L_model
                    Log['L_loss'] = L_loss
                    Log['L_model_train'] = L_model_train
                    Log['L_loss_train'] = L_loss_train
                    Log['L_model_test'] = L_model_test
                    Log['L_loss_test'] = L_loss_test
                    Log['Time_L'] = Time_L
                    Log['Train_Time'] = Train_Time
                    Log['Test_Time'] = Test_Time
                    Log['Acc_Gap'] = Acc_Gap
                    Log['Loss_Gap'] = Loss_Gap
            
                    # if not os.path.exist('./log'):
                    #     os.makedirs('./log')
                    save_name = './log/' + self.name_exp + '.csv'
                    Log.to_csv(save_name, index=False)
                    
                    ### Stopppp
                    if self.L_model_thres != 0 and L_model_train_ > self.L_model_thres:
                        print('\n\n >>> Stop due to reach L_model threshold!!!', L_model_train_)
                        break
                    
                    if self.L_loss_thres != 0 and L_loss_train_ > self.L_loss_thres:
                        print('\n\n >>> Stop due to reach L_loss threshold!!!', L_loss_train_)
                        break 
                    
                if len(Train_Loss)>11 and abs(Train_Loss[-1] - Train_Loss[-10]) < 1e-5:
                        break
                    
                                           
                
                scheduler.step()
        
        experiment.end()
        
    

    def train(self, net, epoch, train_loader, criterion, optimizer, name_exp, lamb, num_epochs=200):
        
        time0 = time.time()
    
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        L_loss = 0
        L_model = 0
        time_L = 0
        loop = tqdm(enumerate(train_loader), leave=True,total=len(train_loader))
        for batch_idx, (inputs, targets) in loop:
            # print('input', type(inputs), 'target', type(targets))
            inputs, targets = inputs.to(device), targets.to(device)
            inputs.requires_grad = True # this is essential!
            outputs = net(inputs)
            # print(outputs[0].shape, outputs[0])
            # return 0 
            if self.is_mse:
                targets_onehot = self.target_2_onehot(targets).to(device)
                loss = criterion(outputs, targets_onehot)
                loss_t = loss.item()
            else:
                # if len(outputs.shape) == 0:
                #     exit()
                try:
                    loss = criterion(outputs, targets)
                except:
                    exit()
                loss_t = loss.item()
            
            if self.reg == 'l2':
                l2_reg = torch.tensor(0.).cuda()
                for param in net.parameters():
                      l2_reg += torch.sum(torch.square(param))
                      
                loss += lamb*l2_reg
                
            elif self.reg == 'l1':
                l1_reg = torch.tensor(0.).cuda()
                for param in net.parameters():
                    # print(torch.sum(torch.abs(param)).shape)
                    l1_reg += torch.sum(torch.abs(param))
                      
                loss += lamb*l1_reg
                
            elif self.reg == 'frobreg':
                reg = JacobianReg(n=-1)
                loss_jr = reg(inputs, outputs)
                loss += loss_jr
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if self.reg == 'lcc':
                self.max_norm(net, max_val=lamb, norm_type=2)
    
            train_loss += loss_t
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            time1 = time.time()
            
            temp_L_loss = estimate_L_loss(net, criterion, optimizer, (inputs, targets), self.is_mse)
            if temp_L_loss == 0:
                L_loss += 0
            else:
                L_loss += temp_L_loss.item()
            L_model += estimate_L_model(net, criterion, (inputs, targets)).item()
            time2 = time.time()
            time_L += (time2 - time1)
    
            #update progress_bar:
            loop.set_description(f'Epoch [{epoch}/{num_epochs}]')
            loop.set_postfix(Train_loss=train_loss/(batch_idx + 1), Train_acc=100.*correct/total, L_model=L_model/(batch_idx + 1), L_loss=L_loss/(batch_idx + 1))
    
        time3 = time.time()
        num_batch = batch_idx + 1
        
        return train_loss/num_batch, 100.*correct/total, L_model/num_batch, L_loss/num_batch, time_L, (time3 - time0)
    
    
    def test(self, net, epoch, test_loader, criterion, num_epochs=200):
        # global best_acc
        time0 = time.time()
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        loop = tqdm(enumerate(test_loader), leave=True,total=len(test_loader))
    
        with torch.no_grad():
            for batch_idx, (inputs, targets) in loop:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                
                if self.is_mse:
                    targets_onehot = self.target_2_onehot(targets).to(device)
                    loss = criterion(outputs, targets_onehot)
                else:
                    loss = criterion(outputs, targets)
                
    
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                #update progress_bar:
                loop.set_description(f'>> test [{epoch}/{num_epochs}]')
                loop.set_postfix(Test_loss=test_loss/(batch_idx + 1), Test_acc=100.*correct/total)
                
        time1 = time.time()
        
        time_t1 = time.time()
        # test_L_model = self.test_L_model(self.net, self.criterion, [self.train_loader, self.test_loader])
        L_model_train_, L_loss_train_, L_model_test_, L_loss_test_ = self.test_L(self.net, self.criterion, self.optimizer, [self.train_loader, self.test_loader], self.is_mse)
        time_t2 = time.time()
    
        return test_loss/(batch_idx + 1), 100.*correct/total, L_model_train_, L_loss_train_, L_model_test_, L_loss_test_, (time_t2 - time_t1), (time1 - time0)
        
    def target_2_onehot(self, targets):
        bs = targets.shape[0]
        onehot = torch.zeros(bs, 10)
        onehot = onehot + 1e-6
        for i, t in enumerate(targets):
            onehot[i][t] = 1
            
        return onehot
        
    def max_norm(self, model, max_val=1, norm_type=2):
        pre = 0
        post = 0
        for name, param in model.named_parameters():
            if 'bias' not in name:
                if norm_type == 2:
                    norm = param.norm(2) #2 - 8
                elif norm_type == 1:
                    norm = param.norm(1) #1 - 48
                elif norm_type == 'inf':
                    norm = param.norm(float('inf')) #inf - 36
                
                # pre += norm
                param_t = param * (1.0 / max(1.0, norm / max_val))
                param.data.copy_(param_t)
                # param = param * (1.0 / max(1.0, norm / max_val))
                
        # print('Pre: ', pre)
                
        # for name, param in model.named_parameters():
        #     if 'bias' not in name:
        #         if norm_type == 2:
        #             norm = param.norm(2) #2 - 8
        #         elif norm_type == 1:
        #             norm = param.norm(1) #1 - 48
        #         elif norm_type == 'inf':
        #             norm = param.norm(float('inf')) #inf - 36
                
        #         post += norm   
                
        # print('Post: ', post)
              
    
    def test_L_model(self, net, criterion, list_data_loader):
        L_model = 0
        for dataloader in list_data_loader:
            for inputt, targett in dataloader:
                inputt, targett = inputt.to(device), targett.to(device)
                temp_L = estimate_L_model(net, criterion, (inputt, targett)).item()
                if temp_L > L_model:
                    L_model = temp_L
                    
        return L_model
    
    def test_L(self, net, criterion, optimizer, list_data_loader, is_mse):
        
        def test_L_(net, criterion, optimizer, dataloader, is_mse):
            L_model, L_loss = 0, 0
            for inputt, targett in dataloader:
                inputt, targett = inputt.to(device), targett.to(device)
                
                #L_loss
                try:
                    temp_L = estimate_L_loss(net, criterion, optimizer, (inputt, targett), is_mse).item()
                except:
                    temp_L = estimate_L_loss(net, criterion, optimizer, (inputt, targett), is_mse)
                
                if temp_L > L_loss:
                    L_loss = temp_L
                
                #L_model
                temp_L2 = estimate_L_model(net, criterion, (inputt, targett)).item()
                if temp_L2 > L_model:
                    L_model = temp_L2
                    
            return L_model, L_loss
                    
        train_loader, test_loader = list_data_loader[0], list_data_loader[1]
        L_model_train_, L_loss_train_ = test_L_(net, criterion, optimizer, train_loader, is_mse)
        L_model_test_, L_loss_test_ = test_L_(net, criterion, optimizer, test_loader, is_mse)
        
        return L_model_train_, L_loss_train_, L_model_test_, L_loss_test_
        

class JacobianReg(torch.nn.Module):
    '''
    Loss criterion that computes the trace of the square of the Jacobian.
    Arguments:
        n (int, optional): determines the number of random projections.
            If n=-1, then it is set to the dimension of the output 
            space and projection is non-random and orthonormal, yielding 
            the exact result.  For any reasonable batch size, the default 
            (n=1) should be sufficient.
    '''
    def __init__(self, n=-1):
        assert n == -1 or n > 0
        self.n = n
        super(JacobianReg, self).__init__()

    def forward(self, x, y):
        '''
        computes (1/2) tr |dy/dx|^2
        '''
        B,C = y.shape
        if self.n == -1:
            num_proj = C
        else:
            num_proj = self.n
        J2 = 0
        for ii in range(num_proj):
            if self.n == -1:
                # orthonormal vector, sequentially spanned
                v=torch.zeros(B,C)
                v[:,ii]=1
            else:
                # random properly-normalized vector for each sample
                v = self._random_vector(C=C,B=B)
            if x.is_cuda:
                v = v.cuda()
            Jv = self._jacobian_vector_product(y, x, v, create_graph=True)
            J2 += C*torch.norm(Jv)**2 / (num_proj*B)
        R = (1/2)*J2
        return R

    def _random_vector(self, C, B):
        '''
        creates a random vector of dimension C with a norm of C^(1/2)
        (as needed for the projection formula to work)
        '''
        if C == 1: 
            return torch.ones(B)
        v=torch.randn(B,C)
        arxilirary_zero=torch.zeros(B,C)
        vnorm=torch.norm(v, 2, 1,True)
        v=torch.addcdiv(arxilirary_zero, 1.0, v, vnorm)
        return v
                                                                            
    def _jacobian_vector_product(self, y, x, v, create_graph=False): 
        '''
        Produce jacobian-vector product dy/dx dot v.
        Note that if you want to differentiate it,
        you need to make create_graph=True
        '''                                                            
        flat_y = y.reshape(-1)
        flat_v = v.reshape(-1)
        grad_x, = torch.autograd.grad(flat_y, x, flat_v, 
                                        retain_graph=True, 
                                        create_graph=create_graph)
        return grad_x