import os
import argparse
from model import *

from data_loader import DataLoader
from evaluator import Evaluator
from trainer import Trainer
import random
import numpy as np
import torch
import flair
from flair.embeddings import *
from torchvision.models import resnet152

import subprocess


device_id = 1
torch.cuda.set_device(device_id)
flair.device = torch.device('cuda:%d' % device_id)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Argument Parser for MNER')

    parser.add_argument("--lamb", dest="lamb", type=float, default=0.5)
    parser.add_argument("--conf_thresh", dest="conf_thresh", type=float, default=0.5)
    parser.add_argument("--nms_thresh", dest="nms_thresh", type=float, default=0.4)
    parser.add_argument("--split_file", dest="split_file", type=str,
                        default='datasets/Textual target labels')
    parser.add_argument("--image_obj_features_dir", dest="image_obj_features_dir", type=str,
                        default='datasets/images')
    parser.add_argument("--image_obj_boxes_dir", dest="image_obj_boxes_dir", type=str,
                        default="datasets/annotations")

    parser.add_argument("--yolov4conv137weight", dest="yolov4conv137weight", type=str, default='pretrained/yolo')


    MODEL_DIR = '/mnt/E/wjq/MSTD/models_addpre/'


    parser.add_argument("--batch_size", dest="batch_size", type=int, default=8)
    parser.add_argument("--lr", dest="lr", type=float, default=5e-5)
    parser.add_argument("--dropout", dest="dropout", type=float, default=0.5)
    parser.add_argument("--num_epochs", dest="num_epochs", type=int, default=40)
    parser.add_argument("--clip_value", dest="clip_value", type=float, default=1)
    parser.add_argument("--wdecay", dest="wdecay", type=float, default=0.0000001)
    parser.add_argument("--mode", dest="mode", type=int, default=1)
    parser.add_argument("--model_dir", dest="model_dir", type=str, default=MODEL_DIR)
    parser.add_argument("--model_file_name", dest="model_file_name", type=str, default="epoch48_f1_0.36438_ap50_0.49680.pth")
    parser.add_argument("--sent_maxlen", dest="sent_maxlen", type=int, default=35)
    parser.add_argument("--word_maxlen", dest="word_maxlen", type=int, default=41)
    args = parser.parse_args()

    return args


def main():
    params = parse_arguments()
    print(params)
    print("Constructing data loaders...")

    dl = DataLoader(params)
    evaluator = Evaluator(params, dl)
    print("Constructing data loaders...[OK]")

    if params.mode == 0:
        print("Training...")
        t = Trainer(params, dl, evaluator)
        t.train()
        print("Training...[OK]")
    elif params.mode == 1:
        print("Loading model...")
        embedding_types = [
            BertEmbeddings("bert-base-uncased"),
            # BertEmbeddings("bert-large-uncased"),
        ]

        embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

        model = MSTD(params, embeddings, None, num_of_tags=4)
        model_file_path = os.path.join(params.model_dir, params.model_file_name)
        model.load_state_dict(torch.load(model_file_path))
        if torch.cuda.is_available():
            model = model.cuda()
        print("Loading model...[OK]")

        print("Evaluating model on test set...")
        (acc, f1, prec, rec, EM), ap, ap50, ap75 = evaluator.get_accuracy(model, 'test')
        print("Accuracy : {}".format(acc))
        print("F1 : {}".format(f1))
        print("Precision : {}".format(prec))
        print("Recall : {}".format(rec))
        print("EM : {}".format(EM))
        print("AP : {}".format(ap))
        print("AP50 : {}".format(ap50))
        print("AP75 : {}".format(ap75))
        print("Evaluating model on test set...[OK]")


if __name__ == '__main__':
    setup_seed(1024)
    main()
