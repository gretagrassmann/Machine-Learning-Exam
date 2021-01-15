import time
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import DataLoader, DataListLoader
from torch_geometric.nn import DataParallel
from sklearn import metrics
import torch.nn.functional as F
from sklearn.metrics import precision_recall_curve
from utils import FocalLoss


class Trainer():
    def __init__(self, option, model, train_dataset, valid_dataset, test_dataset=None, weight=[[1.0, 1.0]],
                 tasks_num=17):
        # option contains the most important variable
        self.option = option
        self.device = torch.device("cuda:{}".format(option['gpu'][0]) if torch.cuda.is_available() else "cpu")
        """
        In my case cpu.The CPU is a powerful execution engine, that focuses its smaller number of cores on individual 
        tasks and on getting things done quickly. This makes it uniquely well equipped for jobs ranging from serial 
        computing to running databases. GPUs began as specialized ASICs developed to accelerate specific 3D rendering tasks.
        """
        self.model = DataParallel(model).to(self.device) if option['parallel'] else model.to(self.device)

        # Setting the train valid and test data loader
        if self.option['parallel']:
            self.train_dataloader = DataListLoader(train_dataset, batch_size=self.option['batch_size'], shuffle=True)
            # Data loader which merges data objects from a torch_geometric.data.dataset to a python list.
            self.valid_dataloader = DataListLoader(valid_dataset, batch_size=self.option['batch_size'])
            if test_dataset: self.test_dataloader = DataListLoader(test_dataset, batch_size=self.option['batch_size'])
        else:
            self.train_dataloader = DataLoader(train_dataset, batch_size=self.option['batch_size'], shuffle=True)
            # Data loader which merges data objects from a torch_geometric.data.dataset to a mini-batch.
            self.valid_dataloader = DataLoader(valid_dataset, batch_size=self.option['batch_size'])
            if test_dataset: self.test_dataloader = DataLoader(test_dataset, batch_size=self.option['batch_size'])
        self.save_path = self.option['exp_path']
        # Setting the Adam optimizer with hyper-param
        if option['focalloss']:
            self.log('Using FocalLoss')
            print("Focalloss")
            self.criterion = [FocalLoss(alpha=1 / w[0]) for w in weight]  # alpha 0.54
        else:
            self.criterion = [torch.nn.CrossEntropyLoss(torch.Tensor(w).to(self.device), reduction='mean') for w in
                              weight]
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.option['lr'],
                                          weight_decay=option['weight_decay'])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.7,
            patience=self.option['lr_scheduler_patience'], min_lr=1e-6
        )
        """
        Learning rate scheduling should be applied after optimizer’s update (->self.optimizer and then self.scheduler).
        torch.optim.lr_scheduler provides several methods to adjust the learning rate based on the number of epochs. 
        torch.optim.lr_scheduler.ReduceLROnPlateau allows dynamic learning rate reducing based on some validation 
        measurements.
        mode='min' : In min mode, lr will be reduced when the quantity monitored has stopped decreasing.
        factor=0.7 : Factor by which the learning rate will be reduced. new_lr = lr * factor.
        patience (int) : Number of epochs with no improvement after which learning rate will be reduced.
        min_lr (float or list) : A scalar or a list of scalars. A lower bound on the learning rate of all param 
                                groups or each group respectively. 
        """

        # other
        self.start = time.time()
        self.tasks_num = tasks_num
        # For 'bace' this is one

        self.records = {'trn_record': [], 'val_record': [], 'val_losses': [],
                        'best_ckpt': None, 'val_roc': [], 'val_prc': []}
        self.log(msgs=['\t{}:{}\n'.format(k, v) for k, v in self.option.items()], show=False)
        self.log('train set num:{}    valid set num:{}    test set num: {}'.format(
            len(train_dataset), len(valid_dataset), len(test_dataset)))
        self.log("total parameters:" + str(sum([p.nelement() for p in self.model.parameters()])))
        self.log(msgs=str(model).split('\n'), show=False)

    def train_iterations(self):
        self.model.train()
        losses = []
        for data in tqdm(self.train_dataloader): #10 repetitions: 1216 molecules divided in batch with size 128
            """
            Each data corresponds to (for example):
            Batch(batch=[4474], edge_attr=[9640, 10], edge_index=[2, 9640], x=[4474, 39], y=[128, 1])
            128 is the batch size.
            data is a batch with 128 examples of molecules
            """
            # makes your loops show a progress meter
            """
            IN THIS ORDER:
            zero_grad clears old gradients from the last step.
            loss.backward() computes the derivative of the loss w.r.t. the parameters using backpropagation.
            optimizer.step() causes the optimizer to take a step based on the gradients of the parameters.
            """
            self.optimizer.zero_grad()
            # zero_grad clears old gradients from the last step (otherwise you’d just accumulate them from all calls)
            data = data.to(self.device)
            output = self.model(data)
            loss = 0

            i = 0 # there is only one task, classification
            y_pred = output[:, i * 2:(i + 1) * 2] # Since there is only one task, y_pred=output [batch_size,2]
            y_label = data.y[:, i].squeeze()    # y_label has 128 values, one for each molecule [batch_size]

            validId = np.where((y_label.cpu().numpy() == 0) | (y_label.cpu().numpy() == 1))[0] #[128] values 0->127

            y_pred = y_pred[torch.tensor(validId).to(self.device)]
            y_label = y_label[torch.tensor(validId).to(self.device)]
            loss += self.criterion[i](y_pred, y_label)
            y_pred = F.softmax(y_pred.detach().cpu(), dim=-1)[:, 1].view(-1).numpy() # [batch_size,2]->(batch_size,)
            """
            Softmax is a mathematical function that converts a vector of numbers into a vector of probabilities, 
            where the probabilities of each value are proportional to the relative scale of each value in the vector.
            The most common use of the softmax function in applied machine learning is in its use as an activation 
            function in a neural network model. Specifically, the network is configured to output C values, one for 
            each class in the classification task, and the softmax function is used to normalize the outputs, 
            converting them from weighted sum values into probabilities that sum to one. Each value in the output 
            of the softmax function is interpreted as the probability of membership for each class.
            """

            loss.backward()

            self.optimizer.step()
            losses.append(loss.item())

        trn_roc = [metrics.roc_auc_score(y_label, y_pred)] #one value for each epoch
        trn_prc = [metrics.auc(precision_recall_curve(y_label, y_pred)[1],
                               precision_recall_curve(y_label, y_pred)[0])]    #one value for each epoch
        # for each epoch, one value of losses for each batch (for 1216 molecules and batch_size=128 there are
        # 10 batch -> 10 losses)
        trn_loss = np.array(losses).mean()
        return trn_loss, np.array(trn_roc).mean(), np.array(trn_prc).mean()

    def valid_iterations(self, mode='valid'):
        self.model.eval()
        """
        model.eval() changes the forward() behaviour of the module it is called upon. It is a kind of switch for
        some specific layers/parts of the model that behave differently during training and inference (evaluating) time.
        For example, Dropouts Layers, BatchNorm Layers etc. 
        """
        if mode == 'test' or mode == 'eval': dataloader = self.test_dataloader
        if mode == 'valid': dataloader = self.valid_dataloader
        losses = []
        with torch.no_grad():
            """
            Disables gradient calculation. This is useful for inference, when you are sure that you will not call 
            Tensor.backward(). It will reduce memory consumption for computations.
            """
            for data in tqdm(dataloader):
                data = data.to(self.device)
                output = self.model(data)

                i = 0 # only one task: classification
                y_pred = output[:, i * 2:(i + 1) * 2]
                y_label = data.y[:, i].squeeze()

                validId = np.where((y_label.cpu().numpy() == 0) | (y_label.cpu().numpy() == 1))[0]

                y_pred = y_pred[torch.tensor(validId).to(self.device)]
                y_label = y_label[torch.tensor(validId).to(self.device)]

                loss = self.criterion[i](y_pred, y_label)

                y_pred = F.softmax(y_pred.detach().cpu(), dim=-1)[:, 1].view(-1).numpy()

                losses.append(loss.item())

        val_roc = [metrics.roc_auc_score(y_label, y_pred)]
        val_prc = [metrics.auc(precision_recall_curve(y_label, y_pred)[1],
                               precision_recall_curve(y_label, y_pred)[0])]
        val_loss = np.array(losses).mean()
        if mode == 'eval':
            self.log('SEED {} DATASET {}  The best test_loss:{:.3f} test_roc:{:.3f} test_prc:{:.3f}.'
                     .format(self.option['seed'], self.option['dataset'], val_loss, np.array(val_roc).mean(),
                             np.array(val_prc).mean()))

        return val_loss, np.array(val_roc).mean(), np.array(val_prc).mean()

    def train(self):
        self.log('Training start...')
        early_stop_cnt = 0
        for epoch in range(self.option['epochs']):


            trn_loss, trn_roc, trn_prc = self.train_iterations()
            val_loss, val_roc, val_prc = self.valid_iterations()
            test_loss, test_roc, test_prc = self.valid_iterations(mode='test')

            self.scheduler.step(val_loss)
            # If you don’t call it, the learning rate won’t be changed and stays at the initial value.
            lr_cur = self.scheduler.optimizer.param_groups[0]['lr']
            """
            When you initialize the optimizer, pytorch creates one param_group. The learning rate is accessible via 
            param_group['lr'] and the list of parameters is accessible via param_group['params'].
            If you want different learning rates for different parameters, you can initialise the optimizer like this.
            optim.SGD([
                {'params': model.base.parameters()},
                {'params': model.classifier.parameters(), 'lr': 1e-3}
            ], lr=1e-2, momentum=0.9)
            This creates two parameter groups with different learning rates. That is the reason for having param_groups.
            """

            self.log('Epoch:{} {} trn_loss:{:.3f} trn_roc:{:.3f} trn_prc:{:.3f} lr_cur:{:.5f}'.
                     format(epoch, self.option['dataset'], trn_loss, trn_roc, trn_prc, lr_cur),
                     with_time=True)
            self.log('Epoch:{} {} val_loss:{:.3f} val_roc:{:.3f} val_prc:{:.3f} lr_cur:{:.5f}'.
                     format(epoch, self.option['dataset'], val_loss, val_roc, val_prc, lr_cur),
                     with_time=True)
            self.log('Epoch:{} {} test_loss:{:.3f} test_roc:{:.3f} test_prc:{:.3f} lr_cur:{:.5f}'.
                     format(epoch, self.option['dataset'], test_loss, test_roc, test_prc, lr_cur),
                     with_time=True)

            self.records['val_roc'].append(val_roc)
            self.records['val_prc'].append(val_prc)
            self.records['val_record'].append([epoch, val_loss, val_roc, val_prc, lr_cur])
            self.records['trn_record'].append([epoch, trn_loss, trn_roc, trn_prc, lr_cur])
            """
            Now I save the model only if val_roc or val_prc are bigger than the preceding model.
            """
            if val_roc == np.array(self.records['val_roc']).max() or val_prc == np.array(self.records['val_prc']).max():
                self.save_model_and_records(epoch)
                early_stop_cnt = 0
            else:
                early_stop_cnt += 1

        self.save_model_and_records(epoch, final_save=True)

    def save_model_and_records(self, epoch, final_save=False):
        if final_save:
            self.save_loss_records() # Microsoft Excel
            file_name = 'best_model.ckpt'
        else:
            file_name = 'best_model.ckpt'
            self.records['best_ckpt'] = file_name

        with open(os.path.join(self.save_path, file_name), 'wb') as f:
            torch.save({
                'option': self.option,
                'records': self.records,
                'model_state_dict': self.model.state_dict(),
            }, f)
        self.log('Model saved at epoch {}'.format(epoch))

    def save_loss_records(self):
        """
        Saves on Microsoft Excel -for training and validation
        """
        trn_record = pd.DataFrame(self.records['trn_record'],
                                  columns=['epoch', 'trn_loss', 'trn_auc', 'trn_acc', 'lr'])
        val_record = pd.DataFrame(self.records['val_record'],
                                  columns=['epoch', 'val_loss', 'val_auc', 'val_acc', 'lr'])
        ret = pd.DataFrame({
            'epoch': trn_record['epoch'],
            'trn_loss': trn_record['trn_loss'],
            'val_loss': val_record['val_loss'],
            'trn_auc': trn_record['trn_auc'],
            'val_auc': val_record['val_auc'],
            'trn_lr': trn_record['lr'],
            'val_lr': val_record['lr']
        })
        ret.to_csv(self.save_path + '/record.csv') # Microsoft excel file
        return ret

    def load_best_ckpt(self):
        #This is used in run.py:
        ckpt_path = self.save_path + '/' + self.records['best_ckpt']
        self.log('The best ckpt is {}'.format(ckpt_path))
        self.load_ckpt(ckpt_path)

    def load_ckpt(self, ckpt_path):
        self.log('Ckpt loading: {}'.format(ckpt_path))
        ckpt = torch.load(ckpt_path)
        self.option = ckpt['option']
        self.records = ckpt['records']
        self.model.load_state_dict(ckpt['model_state_dict'])

    def log(self, msg=None, msgs=None, with_time=False, show=True):
        if with_time: msg = msg + ' time elapsed {:.2f} hrs ({:.1f} mins)'.format(
            (time.time() - self.start) / 3600.,
            (time.time() - self.start) / 60.
        )
        with open(self.save_path + '/log.txt', 'a+') as f:
            if msgs:
                self.log('#' * 80)
                if '\n' not in msgs[0]: msgs = [m + '\n' for m in msgs]
                f.writelines(msgs)
                if show:
                    for x in msgs:
                        print(x, end='')
                self.log('#' * 80)
            if msg:
                f.write(msg + '\n')
                if show:
                    print(msg)
