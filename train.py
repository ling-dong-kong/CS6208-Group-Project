from __future__ import print_function
import os
import argparse

import numpy as np
import sklearn.metrics as metrics

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

from data import ModelNet40
from model import DGCNN_cls, DualDGC
from util import cal_loss, IOStream


def _init_():
    
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    if not os.path.exists('outputs/'+args.exp_name):
        os.makedirs('outputs/'+args.exp_name)
    if not os.path.exists('outputs/'+args.exp_name+'/'+'models'):
        os.makedirs('outputs/'+args.exp_name+'/'+'models')
        
    os.system('cp main_cls.py outputs'+ '/' + args.exp_name + '/' + 'main_cls.py.backup')
    os.system('cp model.py outputs'   + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp util.py outputs'    + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py outputs'    + '/' + args.exp_name + '/' + 'data.py.backup')


def train(args, io):
    
    train_loader = DataLoader(
        ModelNet40(
            data_root=args.data_root,
            partition='train',
            num_points=args.num_points,
            if_dual=True if args.model == 'dual_dgc' else False,
            ratio=args.ratio,
        ),
        num_workers=args.workers,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        ModelNet40(
            data_root=args.data_root,
            partition='test',
            num_points=args.num_points,
            if_dual=False,
            ratio=0.,
        ),
        num_workers=args.workers,
        batch_size=args.test_batch_size,
        shuffle=True,
        drop_last=False,
    )

    device = torch.device("cuda" if args.cuda else "cpu")

    if args.model == 'dgcnn':
        model = DGCNN_cls(args).to(device)
    elif args.model == 'dual_dgc':
        model = DualDGC(args).to(device)
    else:
        raise Exception("Not implemented")
    print(str(model))
    model = nn.DataParallel(model)
    
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    if args.scheduler == 'cos':
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=1e-3)
    elif args.scheduler == 'step':
        scheduler = StepLR(opt, step_size=20, gamma=0.7)
    
    criterion = cal_loss
    MSE = torch.nn.MSELoss(reduction='mean').to(device)

    best_test_acc = 0
    
    for epoch in range(args.epochs):
        
        # Train
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        
        for batch in train_loader:
            
            if not args.model == 'dual_dgc':
                data, label = batch
                data, label = data.to(device), label.to(device).squeeze()  # [bs, num_points, 3], [bs]
                data = data.permute(0, 2, 1)  # [bs, 3, num_points]
                batch_size = data.size()[0]
            
            else:
                data, label, data_b, label_b = batch
                data, label = data.to(device), label.to(device).squeeze()  # [bs, num_points, 3], [bs]
                data = data.permute(0, 2, 1)  # [bs, 3, num_points]
                data_b, label_b = data_b.to(device), label_b.to(device).squeeze()  # [bs, num_points, 3], [bs]
                data_b = data_b.permute(0, 2, 1)  # [bs, 3, num_points]
                batch_size = data.size()[0]
                
            opt.zero_grad()
            
            if not args.model == 'dual_dgc':
                logits = model(data)
                loss = criterion(logits, label)
                loss.backward()
                opt.step()
                preds = logits.max(dim=1)[1]
            
            else:
                logits_a, logits_b = model(data, data_b)
                loss = criterion(logits_a, label)
                
                softmax_a, softmax_b = F.softmax(logits_a, dim=1), F.softmax(logits_b, dim=1)
                loss_mse = MSE(softmax_a, softmax_b.detach())
                
                loss += loss_mse * 10
                loss.backward()
                opt.step()
                preds = logits_b.max(dim=1)[1]
            
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
            
        if args.scheduler == 'cos':
            scheduler.step()
            
        elif args.scheduler == 'step':
            if opt.param_groups[0]['lr'] > 1e-5:
                scheduler.step()
            if opt.param_groups[0]['lr'] < 1e-5:
                for param_group in opt.param_groups:
                    param_group['lr'] = 1e-5

        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (
            epoch, train_loss * 1.0 / count,
            metrics.accuracy_score(train_true, train_pred),
            metrics.balanced_accuracy_score(train_true, train_pred),
        )
        io.cprint(outstr)

        # Test
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        
        for data, label in test_loader:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            
            if not args.model == 'dual_dgc':
                logits = model(data)
            else:
                logits = model(data, data)
            
            loss = criterion(logits, label)
            preds = logits.max(dim=1)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
            
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (
            epoch, test_loss*1.0 / count, test_acc, avg_per_class_acc,
        )
        io.cprint(outstr)
        
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), 'outputs/%s/models/model.t7' % args.exp_name)


def test(args, io):
    
    test_loader = DataLoader(
        ModelNet40(partition='test', num_points=args.num_points),
        batch_size=args.test_batch_size,
        shuffle=True,
        drop_last=False,
    )

    device = torch.device("cuda" if args.cuda else "cpu")

    if args.model == 'dgcnn':
        model = DGCNN_cls(args).to(device)
    else:
        raise Exception("Not implemented")

    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_path))
    model = model.eval()
    test_acc = 0.0
    count = 0.0
    test_true = []
    test_pred = []
    
    for data, label in test_loader:

        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]
        logits = model(data)
        preds = logits.max(dim=1)[1]
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())
        
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    outstr = 'Test :: test acc: %.6f, test avg acc: %.6f'%(test_acc, avg_per_class_acc)
    io.cprint(outstr)


if __name__ == "__main__":
    
    def get_args():
        parser = argparse.ArgumentParser(description='Point Cloud Recognition')
        parser.add_argument(
            '--exp_name', type=str, default='exp', metavar='N',
            help='Name of the experiments')
        parser.add_argument(
            '--data_root', type=str,
            default='data/modelnet40_ply_hdf5_2048')
        parser.add_argument(
            '--model', type=str, default='dgcnn', metavar='N', choices=['dgcnn', 'dual_dgc'],
            help='Model to use, [dgcnn, dual_dgc]')
        parser.add_argument(
            '--if_attn', action='store_true',
            help='channel attention')
        parser.add_argument(
            '--ratio', type=float, default=0.1,
            help='ratio for dropping points in a point cloud')
        parser.add_argument(
            '--dataset', type=str, default='modelnet40', metavar='N', choices=['modelnet40'],
            help='Dataset to use, [modelnet40]')
        parser.add_argument(
            '--batch_size', type=int, default=32, metavar='batch_size',
            help='Size of batch)')
        parser.add_argument(
            '--test_batch_size', type=int, default=16, metavar='batch_size',
            help='Size of batch)')
        parser.add_argument(
            '--workers', type=int, default=8, metavar='num_workers',
            help='Number of workers)')
        parser.add_argument(
            '--epochs', type=int, default=250, metavar='N',
            help='number of episode to train ')
        parser.add_argument(
            '--use_sgd', type=bool, default=True,
            help='Use SGD')
        parser.add_argument(
            '--lr', type=float, default=0.001, metavar='LR',
            help='learning rate (default: 0.001, 0.1 if using sgd)')
        parser.add_argument(
            '--momentum', type=float, default=0.9, metavar='M',
            help='SGD momentum (default: 0.9)')
        parser.add_argument(
            '--scheduler', type=str, default='cos', metavar='N', choices=['cos', 'step'],
            help='Scheduler to use, [cos, step]')
        parser.add_argument(
            '--no_cuda', type=bool, default=False,
            help='enables CUDA training')
        parser.add_argument(
            '--seed', type=int, default=1205, metavar='S',
            help='random seed')
        parser.add_argument(
            '--eval', type=bool,  default=False,
            help='evaluate the model')
        parser.add_argument(
            '--num_points', type=int, default=1024,
            help='num of points to use')
        parser.add_argument(
            '--dropout', type=float, default=0.5,
            help='initial dropout rate')
        parser.add_argument(
            '--emb_dims', type=int, default=1024, metavar='N',
            help='Dimension of embeddings')
        parser.add_argument(
            '--k', type=int, default=20, metavar='N',
            help='Num of nearest neighbors to use')
        parser.add_argument(
            '--model_path', type=str, default='', metavar='N',
            help='Pretrained model path')
        
        return parser.parse_args()
        
    args = get_args()
    
    _init_()

    io = IOStream('outputs/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        test(args, io)
