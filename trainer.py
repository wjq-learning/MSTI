import torch.utils.data
from torchcrf import CRF
from model import MSTD
from timeit import default_timer as timer
from util import *
from tqdm import tqdm
import numpy as np
from flair.embeddings import *
import flair
from models.yoloLoss import Yolo_loss


def init_xavier(m):
    """
    Sets all the linear layer weights as per xavier initialization
    :param m:
    :return: Nothing
    """
    if type(m) == torch.nn.Linear:
        fan_in = m.weight.size()[1]
        fan_out = m.weight.size()[0]
        std = np.sqrt(6.0 / (fan_in + fan_out))
        m.weight.data.normal_(0, std)
        if m.bias is not None:
            m.bias.data.zero_()


def burnin_schedule(i):
    if i < 20:
        factor = 1
    elif i < 40:
        factor = 0.1
    else:
        factor = 0.01
    return factor


# def burnin_schedule(i):
#     if i < 5:
#         factor = (1 / 5.0) ** 4
#     elif i < 30:
#         factor = 1
#     elif i < 50:
#         factor = 0.1
#     else:
#         factor = 0.01
#     return factor


class Trainer:
    def __init__(self, params, data_loader, evaluator, pre_model=None):
        self.params = params
        self.data_loader = data_loader
        self.evaluator = evaluator
        self.pre_model = pre_model

    def train(self):
        num_of_tags = len(self.data_loader.labelVoc)
        embedding_types = [
            BertEmbeddings("bert-base-uncased"),
            # BertEmbeddings("bert-large-uncased"),
        ]
        embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)
        model = MSTD(self.params, embeddings, self.pre_model, num_of_tags=4)
        crf_loss_function = CRF(num_of_tags)
        yolo_loss_function = Yolo_loss(device=flair.device, batch=self.params.batch_size, n_classes=1)
        if torch.cuda.is_available():
            model = model.cuda()
            crf_loss_function = crf_loss_function.cuda()
        paras = dict(model.named_parameters())
        paras_new = []
        for k, v in paras.items():
            # print(k)
            if 'list_embedding_0' in k:
                paras_new += [{'params': [v], 'lr': 1e-6}]
            elif 'yolov4' in k:
                paras_new += [{'params': [v], 'lr': 1e-4}]
            else:
                paras_new += [{'params': [v], 'lr': 1e-4}]
        optimizer = torch.optim.Adam(paras_new, weight_decay=self.params.wdecay)
        # optimizer = torch.optim.SGD(model.parameters(), lr=self.params.lr, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, burnin_schedule)
        # optimizer = torch.optim.Adam(model.parameters(), lr=self.params.lr)
        f1_best = 0
        ap_best = 0
        best_f1_epoch = 0
        best_ap_epoch = 0
        try:
            lamb = self.params.lamb
            for epoch in range(self.params.num_epochs):
                losses = []
                crf_losses = []
                yolo_losses = []
                start_time = timer()

                for (x, x_flair, y, mask, x_c, lens, img, target, size, bboxes) in tqdm(
                        self.data_loader.train_data_loader):
                    model.train()
                    # print(size)
                    emissions, img_output = model(to_variable(x), x_flair,
                                        lens, to_variable(mask),
                                        to_variable(x_c), to_variable(img))

                    tags = to_variable(y).transpose(0, 1).contiguous()  # seq_len * bs
                    mask = to_variable(mask).byte().transpose(0, 1)  # seq_len * bs
                    target = to_variable(target)
                    # print(target)

                    crf_loss = -crf_loss_function(emissions, tags, mask=mask)
                    # computing yolo loss

                    yolo_loss = yolo_loss_function(img_output, target)
                    # print(img_output[0].shape, img_output[1].shape, img_output[2].shape)
                    loss = lamb*crf_loss + (1-lamb)*yolo_loss
                    # loss = crf_loss
                    loss.backward()
                    losses.append(loss.data.cpu().numpy())
                    crf_losses.append(crf_loss.data.cpu().numpy())
                    yolo_losses.append(yolo_loss.data.cpu().numpy())

                    if self.params.clip_value > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.params.clip_value)
                    optimizer.step()

                scheduler.step()
                optim_state = optimizer.state_dict()

                torch.cuda.empty_cache()
                # Calculate accuracy and save best model
                if (epoch + 1) % self.params.validate_every == 0:
                    # acc_dev, f1_dev, p_dev, r_dev = self.evaluator.get_accuracy(model, 'val', crf_loss_function)
                    with torch.no_grad():
                        (acc_dev, f1_dev, p_dev, r_dev, EM_dev), ap, ap50, ap75 = self.evaluator.get_accuracy(model, 'test', crf_loss_function)

                        print(
                            "Epoch {} : Training Loss: {:.5f}, CRF Loss: {:.5f}, YOLO Loss: {:.5f}, Acc: {:.5f}, F1: {:.5f}, Prec: {:.5f}, Rec: {:.5f}, EM: {:.5f}, AP: {:.5f}, AP50: {:.5f}, AP75: {:.5f}, LR: {:.5f}"
                            "Time elapsed {:.2f} mins"
                                .format(epoch + 1, np.asscalar(np.mean(losses)), np.asscalar(np.mean(crf_losses)), np.asscalar(np.mean(yolo_losses)),
                                        acc_dev, f1_dev, p_dev, r_dev, EM_dev, ap, ap50, ap75,
                                        optim_state['param_groups'][0]['lr'],
                                        (timer() - start_time) / 60))
                        if f1_dev > f1_best:
                            print("f1-score increased....saving weights !!")
                            best_f1_epoch = epoch + 1
                            f1_best = f1_dev
                            f1_model_path = self.params.model_dir + "/epoch{}_f1_{:.5f}_ap50_{:.5f}.pth".format(epoch + 1, f1_dev, ap50)
                            torch.save(model.state_dict(), f1_model_path)
                            print("model save in " + f1_model_path)

                        elif ap50 > ap_best:
                            print("ap-score increased....saving weights !!")
                            best_ap_epoch = epoch + 1
                            ap_best = ap50
                            ap_model_path = self.params.model_dir + "/epoch{}_f1_{:.5f}_ap50_{:.5f}.pth".format(epoch + 1,
                                                                                                           f1_dev, ap50)
                            torch.save(model.state_dict(), ap_model_path)
                            print("model save in " + ap_model_path)
                else:
                    print("Epoch {} : Training Loss: {:.5f}".format(epoch + 1, np.asscalar(np.mean(losses))))
                torch.cuda.empty_cache()
                if epoch + 1 == self.params.num_epochs:
                    print("{} epoch get the best f1 {:.5f}".format(best_f1_epoch, f1_best))
                    print("the model is save in " + f1_model_path)

                    print("{} epoch get the best ap50 {:.5f}".format(best_ap_epoch, ap_best))
                    print("the model is save in " + ap_model_path)
        except KeyboardInterrupt:
            print("Interrupted.. saving model !!!")
            best_model_path = self.params.model_dir + "/epoch{}_f1_{:.5f}.pth".format(best_f1_epoch, f1_best)
            print("{} epoch get the best f1 {:.5f}".format(best_f1_epoch, f1_best))
            print("the model is save in " + f1_model_path)
            torch.save(model.state_dict(), self.params.model_dir + '/model_weights_interrupt.t7')
