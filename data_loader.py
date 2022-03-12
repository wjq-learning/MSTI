import traceback
from numbers import Number
from pytorch_pretrained_bert import BertTokenizer
from util import *
import torch.utils.data
import os
from collections import Counter
import numpy as np
import gensim
from gensim.models import word2vec
from gensim.models import fasttext
import  torch
import random
from flair.data import Sentence
import torchvision.transforms as transforms
from PIL import Image
import cv2

transform = [transforms.ToTensor()]


class CustomDataSet(torch.utils.data.TensorDataset):
    def __init__(self, params, x, x_flair, x_c, y, img_id, s_idx, e_idx):
        self.params = params
        self.x = x
        # print(self.x)
        self.x_flair = x_flair

        self.x_c = x_c


        self.y = y
        # print(self.y)
        self.img_id = img_id
        self.s_idx = s_idx
        self.e_idx = e_idx
        self.grid_num = 7
        self.image_size = 608
        self.mean = (123,117,104)
        self.num_of_samples = e_idx - s_idx

    def __len__(self):
        return self.num_of_samples

    def __getitem__(self, idx):
        x = self.x[self.s_idx + idx]
        x_flair = self.x_flair[self.s_idx + idx]
        y = self.y[self.s_idx + idx]
        x_c = self.x_c[self.s_idx + idx]
        img_id = self.img_id[self.s_idx + idx]
        
        img_path = os.path.join(self.params.image_obj_features_dir, img_id + '.jpg')
        img = cv2.imread(img_path)
        ori_img = img
        h, w, _ = ori_img.shape
        size = [h, w]

        img = cv2.imread(img_path)
        img = self.BGR2RGB(img)  # because pytorch pretrained model use RGB
        img = self.subMean(img, self.mean)
        img = cv2.resize(img, (self.image_size, self.image_size))
        for t in transform:
            img = t(img)
        img = np.array(img)

        box_path = os.path.join(self.params.image_obj_boxes_dir, img_id + '.txt')
        box_f = open(box_path, 'r', encoding='utf-8')
        bboxes = []

        for line in box_f.readlines():
            splited = line.strip().split()
            # print(splited)
            num_boxes = len(splited) // 5
            # print(num_boxes)
            for i in range(num_boxes):
                x1 = float(splited[0 + 5 * i]) / w * self.image_size
                y1 = float(splited[1 + 5 * i]) / h * self.image_size
                x2 = float(splited[2 + 5 * i]) / w * self.image_size
                y2 = float(splited[3 + 5 * i]) / h * self.image_size
                c = int(splited[4 + 5 * i])
                # print(i, x1, y1, x2, y2, c)
                # if c != 0:
                #     print(x1, y1, x2, y2, c)
                bboxes.append([x1, y1, x2, y2, 0])

        out_bboxes = np.array(bboxes, dtype=np.float)
        out_bboxes1 = np.zeros([60, 5])
        out_bboxes1[:min(out_bboxes.shape[0], 60)] = out_bboxes[:min(out_bboxes.shape[0], 60)]
        # print(images.shape, out_bboxes1.shape)
        # print(ori_img.shape)
        # print
        # print(bboxes)
        return x, x_flair, y, x_c, img, out_bboxes1, size, bboxes

    def collate(self, batch):
        x = np.array([x[0] for x in batch])
        x_flair = [x[1] for x in batch]
        y = np.array([x[2] for x in batch])
        x_c = np.array([x[3] for x in batch])
        img = np.array([x[4] for x in batch])
        target = np.array([x[5] for x in batch])
        size = np.array([x[6] for x in batch])
        # print(np.array([z[7] for z in batch]).shape)
        # ori_img = np.array([x[7] for x in batch])
        bboxes = np.array([x[7] for x in batch])

        bool_mask = y == 0
        mask = 1 - bool_mask.astype(np.int)

        # index of first 0 in each row, if no zero then idx = -1
        zero_indices = np.where(bool_mask.any(1), bool_mask.argmax(1), -1).astype(np.int)
        # print(zero_indices)
        input_len = np.zeros(len(batch))
        for i in range(len(batch)):
            if zero_indices[i] == -1:
                input_len[i] = len(x[i])
            else:
                input_len[i] = zero_indices[i]
        sorted_input_arg = np.argsort(-input_len)

        x = x[sorted_input_arg]
        x_flair = sorted(x_flair, key=lambda i: len(i), reverse=True)

        y = y[sorted_input_arg]
        # print(y)
        mask = mask[sorted_input_arg]
        # mask_object = mask_object[sorted_input_arg]
        x_c = x_c[sorted_input_arg]
        img = img[sorted_input_arg]
        target = target[sorted_input_arg]
        size = size[sorted_input_arg]
        # ori_img = ori_img[sorted_input_arg]

        input_len = input_len[sorted_input_arg]
        # img_id = img_id[sorted_input_arg]

        max_seq_len = int(input_len[0])

        # trunc_x = np.zeros((len(batch), max_seq_len))
        trunc_x = np.zeros((len(batch), max_seq_len))
        trunc_x_flair = []

        trunc_y = np.zeros((len(batch), max_seq_len))
        trunc_x_c = np.zeros((len(batch), max_seq_len, self.params.word_maxlen))

        trunc_mask = np.zeros((len(batch), max_seq_len))
        # print(len(batch))
        for i in range(len(batch)):
            # print('max_seq_len:', max_seq_len)
            # print('x_len:', len(x[0]))
            # print('y:', y)
            trunc_x_flair.append(x_flair[i])
            trunc_x[i] = x[i, :max_seq_len]

            trunc_y[i] = y[i, :max_seq_len]
            trunc_mask[i] = mask[i, :max_seq_len]
            trunc_x_c[i] = x_c[i, :max_seq_len, :]

        return to_tensor(trunc_x).long(), trunc_x_flair, to_tensor(trunc_y).long(), to_tensor(trunc_mask).long(), \
               to_tensor(trunc_x_c).long(), to_tensor(input_len).int(), to_tensor(img), to_tensor(target), \
               to_tensor(size), bboxes

    def encoder(self, boxes, labels):
        '''
        boxes (tensor) [[x1,y1,x2,y2],[]]
        labels (tensor) [...]
        return 7x7x30
        '''
        target = torch.zeros((self.grid_num, self.grid_num, 11))
        cell_size = 1. / self.grid_num
        wh = boxes[:, 2:] - boxes[:, :2]
        cxcy = (boxes[:, 2:] + boxes[:, :2]) / 2
        for i in range(cxcy.size()[0]):
            cxcy_sample = cxcy[i]
            ij = (cxcy_sample / cell_size).ceil() - 1  #
            target[int(ij[1]), int(ij[0]), 4] = 1
            target[int(ij[1]), int(ij[0]), 9] = 1
            target[int(ij[1]), int(ij[0]), int(labels[i]) + 9] = 1
            xy = ij * cell_size
            delta_xy = (cxcy_sample - xy) / cell_size
            target[int(ij[1]), int(ij[0]), 2:4] = wh[i]
            target[int(ij[1]), int(ij[0]), :2] = delta_xy
            target[int(ij[1]), int(ij[0]), 7:9] = wh[i]
            target[int(ij[1]), int(ij[0]), 5:7] = delta_xy
        return target

    def BGR2RGB(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def BGR2HSV(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    def HSV2BGR(self, img):
        return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    def RandomBrightness(self, bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h, s, v = cv2.split(hsv)
            adjust = random.choice([0.5, 1.5])
            v = v * adjust
            v = np.clip(v, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def RandomSaturation(self, bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h, s, v = cv2.split(hsv)
            adjust = random.choice([0.5, 1.5])
            s = s * adjust
            s = np.clip(s, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def RandomHue(self, bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h, s, v = cv2.split(hsv)
            adjust = random.choice([0.5, 1.5])
            h = h * adjust
            h = np.clip(h, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def randomBlur(self, bgr):
        if random.random() < 0.5:
            bgr = cv2.blur(bgr, (5, 5))
        return bgr

    def randomShift(self, bgr, boxes, labels):

        center = (boxes[:, 2:] + boxes[:, :2]) / 2
        if random.random() < 0.5:
            height, width, c = bgr.shape
            after_shfit_image = np.zeros((height, width, c), dtype=bgr.dtype)
            after_shfit_image[:, :, :] = (104, 117, 123)  # bgr
            shift_x = random.uniform(-width * 0.2, width * 0.2)
            shift_y = random.uniform(-height * 0.2, height * 0.2)

            if shift_x >= 0 and shift_y >= 0:
                after_shfit_image[int(shift_y):, int(shift_x):, :] = bgr[:height - int(shift_y), :width - int(shift_x),
                                                                     :]
            elif shift_x >= 0 and shift_y < 0:
                after_shfit_image[:height + int(shift_y), int(shift_x):, :] = bgr[-int(shift_y):, :width - int(shift_x),
                                                                              :]
            elif shift_x < 0 and shift_y >= 0:
                after_shfit_image[int(shift_y):, :width + int(shift_x), :] = bgr[:height - int(shift_y), -int(shift_x):,
                                                                             :]
            elif shift_x < 0 and shift_y < 0:
                after_shfit_image[:height + int(shift_y), :width + int(shift_x), :] = bgr[-int(shift_y):,
                                                                                      -int(shift_x):, :]

            shift_xy = torch.FloatTensor([[int(shift_x), int(shift_y)]]).expand_as(center)
            center = center + shift_xy
            mask1 = (center[:, 0] > 0) & (center[:, 0] < width)
            mask2 = (center[:, 1] > 0) & (center[:, 1] < height)
            mask = (mask1 & mask2).view(-1, 1)
            boxes_in = boxes[mask.expand_as(boxes)].view(-1, 4)
            if len(boxes_in) == 0:
                return bgr, boxes, labels
            box_shift = torch.FloatTensor([[int(shift_x), int(shift_y), int(shift_x), int(shift_y)]]).expand_as(
                boxes_in)
            boxes_in = boxes_in + box_shift
            labels_in = labels[mask.view(-1)]
            return after_shfit_image, boxes_in, labels_in
        return bgr, boxes, labels

    def randomScale(self, bgr, boxes):

        if random.random() < 0.5:
            scale = random.uniform(0.8, 1.2)
            height, width, c = bgr.shape
            bgr = cv2.resize(bgr, (int(width * scale), height))
            scale_tensor = torch.FloatTensor([[scale, 1, scale, 1]]).expand_as(boxes)
            boxes = boxes * scale_tensor
            return bgr, boxes
        return bgr, boxes

    def randomCrop(self, bgr, boxes, labels):
        if random.random() < 0.5:
            center = (boxes[:, 2:] + boxes[:, :2]) / 2
            height, width, c = bgr.shape
            h = random.uniform(0.6 * height, height)
            w = random.uniform(0.6 * width, width)
            x = random.uniform(0, width - w)
            y = random.uniform(0, height - h)
            x, y, h, w = int(x), int(y), int(h), int(w)

            center = center - torch.FloatTensor([[x, y]]).expand_as(center)
            mask1 = (center[:, 0] > 0) & (center[:, 0] < w)
            mask2 = (center[:, 1] > 0) & (center[:, 1] < h)
            mask = (mask1 & mask2).view(-1, 1)

            boxes_in = boxes[mask.expand_as(boxes)].view(-1, 4)
            if (len(boxes_in) == 0):
                return bgr, boxes, labels
            box_shift = torch.FloatTensor([[x, y, x, y]]).expand_as(boxes_in)

            boxes_in = boxes_in - box_shift
            boxes_in[:, 0] = boxes_in[:, 0].clamp_(min=0, max=w)
            boxes_in[:, 2] = boxes_in[:, 2].clamp_(min=0, max=w)
            boxes_in[:, 1] = boxes_in[:, 1].clamp_(min=0, max=h)
            boxes_in[:, 3] = boxes_in[:, 3].clamp_(min=0, max=h)

            labels_in = labels[mask.view(-1)]
            img_croped = bgr[y:y + h, x:x + w, :]
            return img_croped, boxes_in, labels_in
        return bgr, boxes, labels

    def subMean(self, bgr, mean):
        mean = np.array(mean, dtype=np.float32)
        bgr = bgr - mean
        return bgr

    def random_flip(self, im, boxes):
        if random.random() < 0.5:
            im_lr = np.fliplr(im).copy()
            h, w, _ = im.shape
            xmin = w - boxes[:, 2]
            xmax = w - boxes[:, 0]
            boxes[:, 0] = xmin
            boxes[:, 2] = xmax
            return im_lr, boxes
        return im, boxes

    def random_bright(self, im, delta=16):
        alpha = random.random()
        if alpha > 0.3:
            im = im * alpha + random.randrange(-delta, delta)
            im = im.clip(min=0, max=255).astype(np.uint8)
        return im


class DataLoader:
    def __init__(self, params):
        '''
        self.x : sentence encoding with padding at word level
        self.x_c : sentence encoding with padding at character level
        self.x_img : image features corresponding to the sentences
        self.y : label corresponding to the words in the sentences
        :param params:
        '''
        self.params = params

        self.id_to_vocb, \
            self.sentences, self.datasplit, \
            self.x, self.x_flair, self.x_c, self.y, \
            self.num_sentence, self.vocb, \
            self.vocb_char, self.labelVoc ,self.img_id\
            = self.load_data()

        kwargs = {'num_workers': 8, 'pin_memory': True} if torch.cuda.is_available() else {}

        dataset_train = CustomDataSet(params, self.x,  self.x_flair, self.x_c, self.y, self.img_id, self.datasplit[0], self.datasplit[1])
        self.train_data_loader = torch.utils.data.DataLoader(dataset_train,
                                                             batch_size=self.params.batch_size,
                                                             collate_fn=dataset_train.collate,
                                                             shuffle=True, **kwargs)

        dataset_val = CustomDataSet(params, self.x, self.x_flair, self.x_c, self.y,  self.img_id, self.datasplit[1], self.datasplit[2])
        self.val_data_loader = torch.utils.data.DataLoader(dataset_val,
                                                           batch_size=1,
                                                           collate_fn=dataset_val.collate,
                                                           shuffle=False, **kwargs)
        dataset_test = CustomDataSet(params, self.x,  self.x_flair, self.x_c, self.y, self.img_id, self.datasplit[2], self.datasplit[3])
        self.test_data_loader = torch.utils.data.DataLoader(dataset_test,
                                                            batch_size=1,
                                                            collate_fn=dataset_test.collate,
                                                            shuffle=False, **kwargs)




    def load_data(self):
        print('calculating vocabulary...')

        datasplit, sentences, sent_maxlen, word_maxlen, num_sentence,  img_id = self.load_sentence(
            'IMGID', self.params.split_file, 'train', 'val', 'test')

        id_to_vocb, vocb, vocb_inv, vocb_char, vocb_inv_char, labelVoc, labelVoc_inv = self.vocab_bulid(sentences)

        # word_matrix = self.load_word_matrix(vocb, size=self.params.embedding_dimension)

        x, x_flair, x_c, y = self.pad_sequence(sentences, vocb, vocb_char, labelVoc,
                                               word_maxlen=self.params.word_maxlen, sent_maxlen=sent_maxlen)

        return [id_to_vocb, sentences, datasplit, x, x_flair, x_c, y, num_sentence,  vocb, vocb_char,
                labelVoc, img_id]

    def load_sentence(self, IMAGEID, tweet_data_dir, train_name, val_name, test_name):
        """
        read the word from doc, and build sentence. every line contain a word and it's tag
        every sentence is split with a empty line. every sentence begain with an "IMGID:num"

        """
        # IMAGEID='IMGID'
        img_id = []
        sentences = []
        sentence = []
        sent_maxlen = 0
        word_maxlen = 0
        obj_features = []
        # img_feature = []
        datasplit = []
        # mask_object = []

        for fname in (train_name, val_name, test_name):
            datasplit.append(len(img_id))
            with open(os.path.join(tweet_data_dir, fname), 'r', encoding='utf-8') as file:
                last_line = ''
                for line in file:
                    line = line.rstrip()
                    if line == '':
                        sent_maxlen = max(sent_maxlen, len(sentence))
                        sentences.append(sentence)
                        sentence = []
                    else:
                        if IMAGEID in line:
                            num = line[6:]
                            img_id.append(num)
                            if last_line != '':
                                print(num)
                        else:
                            if len(line.split()) == 1:
                                print(line)
                            sentence.append(line.split())
                            word_maxlen = max(word_maxlen, len(str(line.split()[0])))
                    last_line = line

        # sentences.append(sentence)
        datasplit.append(len(img_id))
        num_sentence = len(sentences)

        print("datasplit", datasplit)
        print(sentences[len(sentences) - 2])
        print(sentences[0])


        print('sent_maxlen', sent_maxlen)
        print('word_maxlen', word_maxlen)
        print('number sentence', len(sentences))
        print('number image', len(img_id))

        return [datasplit, sentences, sent_maxlen, word_maxlen, num_sentence, img_id]

    def vocab_bulid(self, sentences):
        """
        input:
            sentences list,
            the element of the list is (word, label) pair.
        output:
            some dictionaries.

        """
        words = []
        chars = []
        labels = []

        for sentence in sentences:
            # print(sentence)
            for word_label in sentence:
                # print(word_label)
                if word_label[1] != 'O' and word_label[1] != 'B-S' and word_label[1] != 'I-S':
                    print(sentence)
                words.append(word_label[0])
                labels.append(word_label[1])
                for char in word_label[0]:
                    chars.append(char)
        word_counts = Counter(words)
        vocb_inv = [x[0] for x in word_counts.most_common()]
        vocb = {x: i + 1 for i, x in enumerate(vocb_inv)}
        vocb['PAD'] = 0
        id_to_vocb = {i: x for x, i in vocb.items()}

        char_counts = Counter(chars)
        vocb_inv_char = [x[0] for x in char_counts.most_common()]
        vocb_char = {x: i + 1 for i, x in enumerate(vocb_inv_char)}

        labels_counts = Counter(labels)
        print('labels_counts', len(labels_counts))
        print(labels_counts)
        labelVoc_inv, labelVoc = self.label_index(labels_counts)
        print('labelVoc', labelVoc)

        return [id_to_vocb, vocb, vocb_inv, vocb_char, vocb_inv_char, labelVoc, labelVoc_inv]

    @staticmethod
    def label_index(labels_counts):
        """
           the input is the output of Counter. This function defines the (label, index) pair,
           and it cast our datasets label to the definition (label, index) pair.
        """

        num_labels = len(labels_counts)
        labelVoc_inv = [x[0] for x in labels_counts.most_common()]

        labelVoc = {'0': 0,
                    'B-S': 1, 'I-S': 2,
                    'O': 3}
        if len(labelVoc) < num_labels:
            for key, value in labels_counts.items():
                if not labelVoc.has_key(key):
                    labelVoc.setdefault(key, len(labelVoc))
        return labelVoc_inv, labelVoc

    @staticmethod
    def pad_sequences(y, sent_maxlen):
        padded = np.zeros((len(y), sent_maxlen))
        for i, each in enumerate(y):
            trunc_len = min(sent_maxlen, len(each))
            padded[i, :trunc_len] = each[:trunc_len]
        return padded.astype(np.int32)

    def pad_sequence(self, sentences, vocabulary, vocabulary_char, labelVoc, word_maxlen=30,
                     sent_maxlen=35):
        """
            This function is used to pad the word into the same length, the word length is set to 30.
            Moreover, it also pad each sentence into the same length, the length is set to 35.

        """
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
        print(tokenizer)

        x = []
        x_flair = []
        y = []
        for sentence in sentences:
            w_id = []
            y_id = []
            st = Sentence()
            for idx, word_label in enumerate(sentence):
                try:
                    w_id.append(tokenizer.vocab[word_label[0].lower()])
                except Exception as e:
                    w_id.append(tokenizer.vocab['[MASK]'])
                st.add_token(word_label[0])
                y_id.append(labelVoc[word_label[1]])

            x.append(w_id)
            x_flair.append(st)
            y.append(y_id)

        y = self.pad_sequences(y, sent_maxlen)
        x = self.pad_sequences(x, sent_maxlen)

        x_c = []
        for sentence in sentences:
            s_pad = np.zeros([sent_maxlen, word_maxlen], dtype=np.int32)
            s_c_pad = []
            for word_label in sentence:
                w_c = []
                char_pad = np.zeros([word_maxlen], dtype=np.int32)
                for char in word_label[0]:
                    try:
                        w_c.append(vocabulary_char[char])
                    except:
                        w_c.append(0)
                if len(w_c) <= word_maxlen:
                    char_pad[:len(w_c)] = w_c
                else:
                    char_pad = w_c[:word_maxlen]

                s_c_pad.append(char_pad)

            for i in range(len(s_c_pad)):
                # Post truncating
                if i < sent_maxlen:
                    s_pad[i, :len(s_c_pad[i])] = s_c_pad[i]
            x_c.append(s_pad)

        x_c = np.asarray(x_c)
        # x = np.asarray(x)
        y = np.asarray(y)
        # mask_object = np.asarray(mask_object)
        # print(x)
        # print(y)
        return [x, x_flair, x_c, y]
