import argparse
import os
import time
from utils import seed_set, Option
from model import TrimNet as Model
from trainer import Trainer
from dataset import load_dataset_scaffold


def train():
    parser = argparse.ArgumentParser()
    # For writing user-friendly command line interfaces. The program defines what arguments it requires, and argparse
    # figures out how to parse those out of sys.argv (list of command line arguments passed to the script) or takes the
    # default values.
    parser.add_argument("--data", type=str, default='../data/', help="all data dir")
    parser.add_argument("--dataset", type=str, default='bace', help="muv,tox21,toxcast,sider,clintox,hiv,bace,bbbp")
    parser.add_argument('--seed', default=68, type=int)
    parser.add_argument("--gpu", type=int, nargs='+', default=0, help="CUDA device ids")

    parser.add_argument("--hid", type=int, default=32, help="hidden size of transformer model")
    parser.add_argument('--heads', default=4, type=int)
    parser.add_argument('--depth', default=3, type=int)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--batch_size", type=int, default=128, help="number of batch_size") #DEFAULT 128
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs") #DEFAULT 200
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument('--lr_scheduler_patience', default=10, type=int)
    parser.add_argument('--early_stop_patience', default=-1, type=int)
    parser.add_argument('--lr_decay', default=0.98, type=float)
    parser.add_argument('--focalloss', default=False, action="store_true")

    parser.add_argument('--eval', default=False, action="store_true")
    parser.add_argument("--exps_dir", default='../test', type=str, help="out dir")
    parser.add_argument('--exp_name', default=None, type=str)

    d = vars(parser.parse_args())
    # Turns the argparse.Namespace into a keyword argument that you can send to a function
    args = Option(d)
    # Class instantiation defined in utils.py
    seed_set(args.seed)
    # Defined in utils.py

    args.parallel = True if args.gpu and len(args.gpu) > 1 else False
    args.parallel_devices = args.gpu # =0
    args.tag = time.strftime("%m-%d-%H-%M") if args.exp_name is None else args.exp_name
    args.exp_path = os.path.join(args.exps_dir, args.tag)
    """ Once I run run.py a folder (ex: 01-12-11-51) is opened in the folder test.
        It will contain log (notepad), records (microsoft excel) and best_model.ckpt."""

    if not os.path.exists(args.exp_path): #This creates that folder
        os.makedirs(args.exp_path)
    args.code_file_path = os.path.abspath(__file__)


    if args.dataset == 'bace':
        args.tasks = ['Class']
        train_dataset, valid_dataset, test_dataset = load_dataset_scaffold(args.data, args.dataset, args.seed,
                                                                           args.tasks)

        args.out_dim = 2 * len(args.tasks)
        # For 'bace' len(args.tasks)=1. It's just classification

    else:  # Unknown dataset error
        raise Exception('Unknown dataset, please enter the correct --dataset option')

    args.in_dim = train_dataset.num_node_features
    args.edge_in_dim = train_dataset.num_edge_features
    weight = train_dataset.weights
    option = args.__dict__
    # __dict__ contains all the attributes which describe the object in question


    model = Model(args.in_dim, args.edge_in_dim, hidden_dim=args.hid, depth=args.depth,
                  heads=args.heads, dropout=args.dropout, outdim=args.out_dim)
    trainer = Trainer(option, model, train_dataset, valid_dataset, test_dataset, weight=weight,
                      tasks_num=len(args.tasks))
    trainer.train()
    print('Testing...')
    trainer.load_best_ckpt()
    trainer.valid_iterations(mode='eval')




if __name__ == '__main__':
    train()
