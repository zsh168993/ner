import os
import copy
import torch
import logging
from torch.cuda.amp import autocast as ac
from torch.utils.data import DataLoader, RandomSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from utils.attack_train_utils import FGM, PGD
from utils.functions_utils import load_model_and_parallel, swa

logger = logging.getLogger()
logger.setLevel(logging.INFO)

if not logger.handlers:
    # Logging to a file
    file_handler = logging.FileHandler("train.log")
    file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logger.addHandler(file_handler)

    # Logging to console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(stream_handler)


def predict(opt, model, predict_dataset):

    predict_loader = DataLoader(dataset=predict_dataset,
                              batch_size=opt.train_batch_size,
                              sampler=None,
                              num_workers=0)
    results=[]
    real=[]
    model.eval()
    for step, batch_data in enumerate(predict_loader):
        for key in batch_data.keys():
            batch_data[key] = batch_data[key]#.to(device)
        batch_data["labels"]=None
        result = model(**batch_data)[0]
        results.append(result)


    logger.info('predict done')
    return results
