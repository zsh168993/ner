
from sklearn.model_selection import train_test_split
import argparse
import warnings
warnings.filterwarnings('ignore')
import os
from data_pro import *
import torch
import torch.optim as optim
import json
import torch.nn as nn
import numpy as np
from dataset_utils import NERDataset
from trainer import *
from predicter import *
from model import *
parser = argparse.ArgumentParser()




# args for path
parser.add_argument('--path_train', default='data/train.json',
                    help="path_dataset")
parser.add_argument('--path_dev', default='data/dev.json',
                    help="path_dev")

parser.add_argument('--output_dir', default='data',
                    help='the output dir for model checkpoints')

parser.add_argument('--bert_dir', default='bert_model',
                    help='bert dir for ernie / roberta-wwm / uer')

parser.add_argument('--bert_type', default='roberta_wwm',
                    help='roberta_wwm / ernie_1 / uer_large')

parser.add_argument('--task_type', default='crf',
                    help='crf / span / mrc')

parser.add_argument('--loss_type', default='ls_ce',
                    help='loss type for span bert')

parser.add_argument('--use_type_embed', default=False, action='store_true',
                    help='weather to use soft label in span loss')

parser.add_argument('--use_fp16', default=False, action='store_true',
                    help='weather to use fp16 during training')

# other args
parser.add_argument('--seed', type=int, default=123, help='random seed')

parser.add_argument('--gpu_ids', type=str, default='0',
                    help='gpu ids to use, -1 for cpu, "0,1" for multi gpu')

parser.add_argument('--mode', type=str, default='train',
                    help='train / stack')

parser.add_argument('--max_seq_len', default=512, type=int)

parser.add_argument('--eval_batch_size', default=64, type=int)

parser.add_argument('--swa_start', default=3, type=int,
                    help='the epoch when swa start')

# train args
parser.add_argument('--train_epochs', default=10, type=int,
                    help='Max training epoch')

parser.add_argument('--dropout_prob', default=0.1, type=float,
                    help='drop out probability')

parser.add_argument('--lr', default=2e-5, type=float,
                    help='learning rate for the bert module')

parser.add_argument('--other_lr', default=2e-3, type=float,
                    help='learning rate for the module except bert')

parser.add_argument('--max_grad_norm', default=1.0, type=float,
                    help='max grad clip')

parser.add_argument('--warmup_proportion', default=0.1, type=float)

parser.add_argument('--weight_decay', default=0.00, type=float)

parser.add_argument('--adam_epsilon', default=1e-8, type=float)

parser.add_argument('--train_batch_size', default=24, type=int)

parser.add_argument('--eval_model', default=True, action='store_true',
                    help='whether to eval model after training')

parser.add_argument('--attack_train', default='', type=str,
                    help='fgm / pgd attack train when training')

# test args
parser.add_argument('--version', default='v0', type=str,
                    help='submit version')

parser.add_argument('--submit_dir', default='submit', type=str)

parser.add_argument('--ckpt_dir', default='', type=str)

def main():
    # Load the parameters from json file
    args = parser.parse_args()

    # train_datasets =get_examples(args.path_train)
    # train_features=convert_examples_to_features(train_datasets)
    # train_dataset = NERDataset(train_features)


    dev_datasets = get_examples(args.path_dev)
    dev_features = convert_examples_to_features(dev_datasets)
    dev_dataset = NERDataset(dev_features)


    model = CRFModel(args.bert_dir,num_tags=17)
    train(args, model, dev_dataset)

    model = CRFModel(args.bert_dir, num_tags=17)

    model.load_state_dict(torch.load("E:\命名实体识别\data\checkpoint-100000\model.pt", map_location=torch.device('cpu')))
    results=predict(args, model, dev_dataset)
    print(results)

    print(dev_features)




if __name__ == '__main__':
    main()

