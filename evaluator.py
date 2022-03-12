from util import *
from tqdm import tqdm
from torchcrf import CRF
import torch
import torch.nn.functional as F
from predict import decoder
import cv2
import numpy as np
from tool.tv_reference.utils import collate_fn as val_collate
from tool.tv_reference.coco_utils import convert_to_coco_api
from tool.tv_reference.coco_eval import CocoEvaluator
from tool.utils import post_processing
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

VOC_CLASSES = (    # always index 0
    'target')


def voc_ap(rec, prec, use_07_metric=False):
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.

    else:
        # correct ap caculation
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        i = np.where(mrec[1:] != mrec[:-1])[0]

        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap


def voc_eval(pred, target, threshold=0.5, use_07_metric=False, ):
    '''
    preds {'cat':[[image_id,confidence,x1,y1,x2,y2],...],'dog':[[],...]}
    target {(image_id,class):[[],]}
    '''
    # print(pred)
    if len(pred) == 0:
        return -1
    image_ids = [x[0] for x in pred]
    confidence = np.array([float(x[3]) for x in pred])
    BB = np.array([[x[1][0], x[1][1], x[2][0], x[2][1]] for x in pred])
    # print(image_ids)
    # print(BB)
    # print(confidence)
    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    # sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]
    # print(BB)
    image_ids = [image_ids[x] for x in sorted_ind]
    # print(image_ids)
    # go down dets and mark TPs and FPs
    npos = 0.
    for key1 in target:
        npos += len(target[key1])
    # print(npos)
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d, image_id in enumerate(image_ids):
        bb = BB[d]
        if image_id in target:
            temp = target[image_id]
            # print(temp)
            BBGT = [[item[0][0], item[0][1], item[1][0], item[1][1]] for item in temp]
            # print(BBGT)
            for bbgt in BBGT:
                # compute overlaps
                # intersection
                ixmin = np.maximum(bbgt[0], bb[0])
                iymin = np.maximum(bbgt[1], bb[1])
                ixmax = np.minimum(bbgt[2], bb[2])
                iymax = np.minimum(bbgt[3], bb[3])
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih

                union = (bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) + (bbgt[2] - bbgt[0] + 1.) * (
                        bbgt[3] - bbgt[1] + 1.) - inters
                if union == 0:
                    print(bb, bbgt)

                overlaps = inters / union
                if overlaps > threshold:
                    tp[d] = 1
                    BBGT.remove(bbgt)
                    if len(BBGT) == 0:
                        del target[image_id]
                    break
            fp[d] = 1 - tp[d]
        else:
            fp[d] = 1
    # print(tp)
    # print(fp)
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    # print(tp)
    # print(fp)
    rec = tp / float(npos)
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    # print(rec,prec)
    ap = voc_ap(rec, prec, use_07_metric)
    return ap


def draw(ori_img, box, color):
    left_up = box[0]
    right_bottom = box[1]
    # print(left_up)
    cv2.rectangle(ori_img, left_up, right_bottom, color, 2)

    return ori_img


class Evaluator:
    def __init__(self, params, data_loader):
        self.params = params
        self.data_loader = data_loader

    def get_accuracy(self, model, split, crf=None):
        if split == 'val':
            data_loader = self.data_loader.val_data_loader
        else:
            data_loader = self.data_loader.test_data_loader

        if crf == None:
            num_of_tags = len(self.data_loader.labelVoc)
            crf = CRF(num_of_tags)
            if torch.cuda.is_available():
                crf = crf.cuda()

        model.eval()
        labels_pred = []
        labels = []
        words = []
        sent_lens = []

        obj_preds = []
        obj_targets = {}
        for (i, (x, x_flair, y, mask, x_c, lens, img, target, size, bboxes)) in tqdm(enumerate(data_loader)):
            emissions, img_output = model(to_variable(x), x_flair,
                                          lens, to_variable(mask),
                                          to_variable(x_c), to_variable(img), mode='test')  # seq_len * bs * labels
            pre_test_label_index = crf.decode(emissions)  # bs * seq_len
            words.append(x)
            labels.append(y.cpu().numpy().squeeze(0))
            labels_pred.append(pre_test_label_index[0])
            sent_lens.append(lens.cpu().numpy()[0])

            h = size.squeeze()[0]
            w = size.squeeze()[1]

            targets_img = []
            for j in range(len(bboxes[0])):
                box = bboxes[0][j]
                x1 = int(box[0] / 608.0 * w)
                y1 = int(box[1] / 608.0 * h)
                x2 = int(box[2] / 608.0 * w)
                y2 = int(box[3] / 608.0 * h)
                targets_img.append([(x1, y1), (x2, y2), 1.0])
            # print(targets_img)
            if targets_img == [[(0, 0), (0, 0), 1.0]]:
                targets_img = []

            list_features_numpy = []
            for feature in img_output:
                list_features_numpy.append(feature.data.cpu().numpy())
            boxes = post_processing(
                img=img,
                conf_thresh=self.params.conf_thresh,
                n_classes=1,
                nms_thresh=self.params.nms_thresh,
                list_features_numpy=list_features_numpy
            )

            preds_img = []
            for j in range(len(boxes)):
                box = boxes[j]
                x1 = int((box[0] - box[2] / 2.0) * w)
                y1 = int((box[1] - box[3] / 2.0) * h)
                x2 = int((box[0] + box[2] / 2.0) * w)
                y2 = int((box[1] + box[3] / 2.0) * h)
                preds_img.append([(x1, y1), (x2, y2), box[4]])

            # print(i, preds_img)
            # print(i, targets_img)

            for item in preds_img:
                item.insert(0, i)
                obj_preds.append(item)

            for item in targets_img:
                if i not in obj_targets.keys():
                    obj_targets[i] = []
                obj_targets[i].append(item)

        thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        aps = list()
        for th in thresholds:
            a = obj_preds.copy()
            b = obj_targets.copy()
            aps.append(voc_eval(a, b, th))

        ap = np.mean(aps)
        ap50 = aps[0]
        ap75 = aps[5]
        print(aps)

        return self.evaluate(labels_pred, labels, words, sent_lens), ap, ap50, ap75
        # return self.evaluate(labels_pred, labels, words, sent_lens), 1

    def predict(self, model, split, crf=None):
        if split == 'val':
            data_loader = self.data_loader.val_data_loader
        else:
            data_loader = self.data_loader.test_data_loader

        if crf == None:
            num_of_tags = len(self.data_loader.labelVoc)
            crf = CRF(num_of_tags)
            if torch.cuda.is_available():
                crf = crf.cuda()

        model.eval()
        labels_pred = []
        labels = []
        words = []
        sent_lens = []

        obj_preds = []
        obj_targets = {}
        for (i, (x, x_flair, y, mask, x_c, lens, img, target, size, bboxes, ori_img, img_id)) in tqdm(enumerate(data_loader)):
            emissions, img_output = model(to_variable(x), x_flair,
                                          lens, to_variable(mask),
                                          to_variable(x_c), to_variable(img), mode='test')  # seq_len * bs * labels
            pre_test_label_index = crf.decode(emissions)  # bs * seq_len
            words.append(x)
            labels.append(y.cpu().numpy().squeeze(0))
            labels_pred.append(pre_test_label_index[0])
            sent_lens.append(lens.cpu().numpy()[0])

            h = size.squeeze()[0]
            w = size.squeeze()[1]

            targets_img = []
            for j in range(len(bboxes[0])):
                box = bboxes[0][j]
                x1 = int(box[0] / 608.0 * w)
                y1 = int(box[1] / 608.0 * h)
                x2 = int(box[2] / 608.0 * w)
                y2 = int(box[3] / 608.0 * h)
                targets_img.append([(x1, y1), (x2, y2), 1.0])
            # print(targets_img)
            if targets_img == [[(0, 0), (0, 0), 1.0]]:
                targets_img = []

            list_features_numpy = []
            for feature in img_output:
                list_features_numpy.append(feature.data.cpu().numpy())
            boxes = post_processing(
                img=img,
                conf_thresh=self.params.conf_thresh,
                n_classes=1,
                nms_thresh=self.params.nms_thresh,
                list_features_numpy=list_features_numpy
            )

            preds_img = []
            for j in range(len(boxes)):
                box = boxes[j]
                x1 = int((box[0] - box[2] / 2.0) * w)
                y1 = int((box[1] - box[3] / 2.0) * h)
                x2 = int((box[0] + box[2] / 2.0) * w)
                y2 = int((box[1] + box[3] / 2.0) * h)
                preds_img.append([(x1, y1), (x2, y2), box[4]])

            # if len(preds_img) == 0:
            #     preds_img.append([(0, 0), (0, 0), 0.0])
            ori_img = ori_img[0]
            flag = False

            # print(i, preds_img)
            # print(i, targets_img)

            for item in preds_img:
                ori_img = draw(ori_img, item, [0, 0, 255])
                item.insert(0, i)
                obj_preds.append(item)
                flag = True

            for item in targets_img:
                ori_img = draw(ori_img, item, [255, 0, 0])
                if i not in obj_targets.keys():
                    obj_targets[i] = []
                obj_targets[i].append(item)
                flag = True
            print(img_id)
            if flag:
                cv2.imwrite('result/%d.jpg' % img_id, ori_img)

        thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        aps = list()
        for th in thresholds:
            aps.append(voc_eval(obj_preds, obj_targets, th))

        ap = np.mean(aps)
        ap50 = aps[0]
        ap75 = aps[5]

        return self.evaluate(labels_pred, labels, words, sent_lens), ap, ap50, ap75

    def img_eval(self, target_boxes, target_cls_indexs, target_probs, pred_boxes, pred_cls_indexs, pred_probs, h, w):
        preds_img = []
        for i, box in enumerate(pred_boxes):
            x1 = int(box[0] * w)
            x2 = int(box[2] * w)
            y1 = int(box[1] * h)
            y2 = int(box[3] * h)
            prob = float(pred_probs[i])
            preds_img.append([(x1, y1), (x2, y2), prob])
        targets_img = []
        for i, box in enumerate(target_boxes):
            x1 = int(box[0] * w)
            x2 = int(box[2] * w)
            y1 = int(box[1] * h)
            y2 = int(box[3] * h)
            prob = float(target_probs[i])
            targets_img.append([(x1, y1), (x2, y2), prob])
        return preds_img, targets_img

    def evaluate(self, labels_pred, labels, words, sents_length):
        accs = []
        ems = []
        preds = []
        gts = []
        # correct_preds, total_correct, total_preds = 0., 0., 0.
        TP = 0
        TN = 0
        FP = 0
        FN = 0

        for lab, lab_pred, length, word_sent in zip(labels, labels_pred, sents_length, words):
            lab = lab[:length]
            lab_pred = lab_pred[:length]
            # exact_match = [a == b for (a, b) in zip(lab, lab_pred)]
            # accs += exact_match
            # lab_chunks = set(self.get_chunks(lab, self.data_loader.labelVoc))
            # lab_pred_chunks = set(self.get_chunks(lab_pred, self.data_loader.labelVoc))
            # correct_preds += len(lab_chunks & lab_pred_chunks)
            # total_preds += len(lab_pred_chunks)
            # total_correct += len(lab_chunks)
            pred = [a == 1 or a == 2 for a in lab_pred]
            gt = [a == 1 or a == 2 for a in lab]
            exact_match = [a == b for (a, b) in zip(pred, gt)]
            accs += exact_match
            preds += pred
            gts += gt
            ems.append(sum(exact_match) // len(exact_match))

        # p = correct_preds / total_preds if correct_preds > 0 else 0
        # r = correct_preds / total_correct if correct_preds > 0 else 0
        # f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
        # for a, b in zip(preds, gts):
        #     if a == 0 and b == 0:
        #         TN += 1
        #     if a == 1 and b == 1:
        #         TP += 1
        #     if a == 0 and b == 1:
        #         FP += 1
        #     if a == 1 and b == 0:
        #         FN += 1

        # p = float(TP / (TP + FP))
        # r = float(TP / (TP + FN))
        # f1 = float(2*TP)/(2*TP + FP + FN)
        # print(f1, p, r)
        p = precision_score(gts, preds, average='binary')
        r = recall_score(gts, preds, average='binary')
        f1 = f1_score(gts, preds, average='binary')
        acc = np.mean(accs)
        EM = np.mean(ems)
        return acc, f1, p, r, EM

    def get_chunks(self, seq, tags):
        """
        tags:dic{'per':1,....}
        Args:
            seq: [4, 4, 0, 0, ...] sequence of labels
            tags: dict["O"] = 4
        Returns:
            list of (chunk_type, chunk_start, chunk_end)
        Example:
            seq = [4, 5, 0, 3]
            tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
            result = [("PER", 0, 2), ("LOC", 3, 4)]
        """
        default = tags['O']
        idx_to_tag = {idx: tag for tag, idx in tags.items()}
        chunks = []
        chunk_type, chunk_start = None, None
        for i, tok in enumerate(seq):
            # End of a chunk 1
            if tok == default and chunk_type is not None:
                # Add a chunk.
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = None, None

            # End of a chunk + start of a chunk!
            elif tok != default:
                tok_chunk_class, tok_chunk_type = self.get_chunk_type(tok, idx_to_tag)
                if chunk_type is None:
                    chunk_type, chunk_start = tok_chunk_type, i
                elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                    chunk = (chunk_type, chunk_start, i)
                    chunks.append(chunk)
                    chunk_type, chunk_start = tok_chunk_type, i
            else:
                pass
        # end condition
        if chunk_type is not None:
            chunk = (chunk_type, chunk_start, len(seq))
            chunks.append(chunk)

        return chunks

    def get_chunk_type(self, tok, idx_to_tag):
        """
        Args:
            tok: id of token, such as 4
            idx_to_tag: dictionary {4: "B-PER", ...}
        Returns:
            tuple: "B", "PER"
        """
        tag_name = idx_to_tag[tok]
        tag_class = tag_name.split('-')[0]
        tag_type = tag_name.split('-')[-1]
        return tag_class, tag_type
