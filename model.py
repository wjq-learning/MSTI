import torch
import torch.nn as nn
# from models.resnet_model import resnet34
import math
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from pytorch_pretrained_bert import BertTokenizer
from models.resnet_model import resnet152
from models.yolov4 import Yolov4, Conv_Bn_Activation, Upsample, ConvTranspose_Bn_Activation
from models.transformerEncoder import BertAttention
from config import Config
import torch.nn.functional as F


class Conv_Transformer1(torch.nn.Module):
    def __init__(self, hidden_size):
        super(Conv_Transformer1, self).__init__()
        self.hidden_size = hidden_size
        self.conv1 = Conv_Bn_Activation(1024, self.hidden_size, 3, 2, 'relu')
        self.upsample = Upsample()
        self.conv1transpose = ConvTranspose_Bn_Activation(self.hidden_size, 1024, 3, 2, 'relu', 0)

    def c2t(self, x):
        output = F.max_pool2d(self.conv1(x), 2)
        return output

    def t2c(self, x):
        output = self.upsample(x, (8, self.hidden_size, 10, 10))
        output = self.conv1transpose(output)
        return output


class Conv_Transformer2(torch.nn.Module):
    def __init__(self, hidden_size):
        super(Conv_Transformer2, self).__init__()
        self.hidden_size = hidden_size
        self.conv1 = Conv_Bn_Activation(512, self.hidden_size, 3, 2, 'relu')
        self.conv2 = Conv_Bn_Activation(self.hidden_size, self.hidden_size, 3, 2, 'relu')
        self.conv1transpose1 = ConvTranspose_Bn_Activation(self.hidden_size, self.hidden_size, 3, 2, 'relu', 0)
        self.conv1transpose2 = ConvTranspose_Bn_Activation(self.hidden_size, 512, 3, 2, 'relu', 1)
        self.upsample = Upsample()

    def c2t(self, x):
        x = self.conv1(x)
        output = F.max_pool2d(self.conv2(x), 2)
        return output

    def t2c(self, x):
        output = self.upsample(x, (8, self.hidden_size, 10, 10))
        output = self.conv1transpose1(output)
        # print(output.shape)
        output = self.conv1transpose2(output)
        return output


class Conv_Transformer3(torch.nn.Module):
    def __init__(self, hidden_size):
        super(Conv_Transformer3, self).__init__()
        self.hidden_size = hidden_size
        self.conv1 = Conv_Bn_Activation(256, self.hidden_size, 5, 4, 'relu')
        self.conv2 = Conv_Bn_Activation(self.hidden_size, self.hidden_size, 3, 2, 'relu')
        self.conv1transpose1 = ConvTranspose_Bn_Activation(self.hidden_size, self.hidden_size, 3, 2, 'relu', 0)
        self.conv1transpose2 = ConvTranspose_Bn_Activation(self.hidden_size, 256, 5, 4, 'relu', 3)
        self.upsample = Upsample()

    def c2t(self, x):
        x = self.conv1(x)
        output = F.max_pool2d(self.conv2(x), 2)
        return output

    def t2c(self, x):
        output = self.upsample(x, (8, self.hidden_size, 10, 10))
        output = self.conv1transpose1(output)
        output = self.conv1transpose2(output)
        return output


class MSTD(torch.nn.Module):
    def __init__(self, params,  embeddings, pretrain_model, pretrained_weight=None, num_of_tags=4):
        super(MSTD, self).__init__()
        self.params = params

        self.embeddings = embeddings
        self.layer_indexes = [-1]
        self.pooling_operation = "first"

        self.input_embeddding_size = embeddings.embedding_length

        lstm_input_size = self.input_embeddding_size
        # if self.params.pretrain_load == 1:
        #     self.pretrain_model = pretrain_model

        self.yolov4 = Yolov4(yolov4conv137weight=self.params.yolov4conv137weight, n_classes=1)

        self.conv_transformer1 = Conv_Transformer1(self.input_embeddding_size)
        self.conv_transformer2 = Conv_Transformer2(self.input_embeddding_size)
        self.conv_transformer3 = Conv_Transformer3(self.input_embeddding_size)

        self.dropout = nn.Dropout(params.dropout)

        self.self_att = BertAttention(config=Config)

        self.projection = nn.Linear(in_features=self.input_embeddding_size*2, out_features=num_of_tags)
        self.conv2target = nn.Conv2d(768, 11, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_end = nn.BatchNorm2d(11)
        self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=lstm_input_size,
                            num_layers=1, bidirectional=True)

    def forward(self, sentences, x_flair, sentence_lens, mask, chars, img,
                mode="train"):  # !!! word_seq  char_seq
        # print(images.shape)
        batch_size = img.shape[0]
        if mode == 'test':
            batch_size = 1
        attention_mask = torch.zeros((batch_size, 75+mask.shape[1]), dtype=torch.float)
        img_mask = torch.ones((batch_size, 75), dtype=torch.float)
        attention_mask[:, :75] = img_mask
        attention_mask[:, 75:] = mask
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype, device=next(self.parameters()).device)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        d5, d4, d3 = self.yolov4.encoder(img)  # 1024*19*19 512*38*38  256*76*76
        t5 = self.conv_transformer1.c2t(d5)
        t4 = self.conv_transformer2.c2t(d4)
        t3 = self.conv_transformer3.c2t(d3)


        t5 = t5.view(batch_size, self.input_embeddding_size, 5 * 5).transpose(2, 1)
        t4 = t4.view(batch_size, self.input_embeddding_size, 5 * 5).transpose(2, 1)
        t3 = t3.view(batch_size, self.input_embeddding_size, 5 * 5).transpose(2, 1)
        t0 = torch.cat((t5, t4, t3), dim=1)

        self.embeddings.embed(x_flair)

        lengths = [len(sentence.tokens) for sentence in x_flair]
        longest_token_sequence_in_batch: int = max(lengths)

        pre_allocated_zero_tensor = torch.zeros(
            self.input_embeddding_size * longest_token_sequence_in_batch,
            dtype=torch.float,
        )


        all_embs = list()
        for sentence in x_flair:
            all_embs += [
                emb for token in sentence for emb in token.get_each_embedding()
            ]
            nb_padding_tokens = longest_token_sequence_in_batch - len(sentence)


            if nb_padding_tokens > 0:
                t = pre_allocated_zero_tensor[
                    : self.input_embeddding_size * nb_padding_tokens
                    ].cuda()
                all_embs.append(t)

        embed_flair = torch.cat(all_embs).view(
            [
                len(x_flair),
                longest_token_sequence_in_batch,
                self.input_embeddding_size,
            ]
        )

        embeds = self.dropout(embed_flair)
        # print(t0.shape, embeds.shape)
        embeds = torch.cat((t0, embeds), dim=1)
        embeds = self.self_att(embeds, extended_attention_mask)
        c = embeds[:, :75, :]
        txt_feature = embeds[:, 75:, :]
        c5 = c[:, :25, :]
        c4 = c[:, 25:50, :]
        c3 = c[:, 50:, :]
        c5 = c5.transpose(2, 1).view(batch_size, self.input_embeddding_size, 5, 5)
        c4 = c4.transpose(2, 1).view(batch_size, self.input_embeddding_size, 5, 5)
        c3 = c3.transpose(2, 1).view(batch_size, self.input_embeddding_size, 5, 5)

        t5 = self.conv_transformer1.t2c(c5)
        t4 = self.conv_transformer2.t2c(c4)
        t3 = self.conv_transformer3.t2c(c3)

        img_output = self.yolov4.decoder(t5, t4, t3)

        txt_feature = txt_feature.permute(1, 0, 2)  # se bs hi+embedding_h+c
        # sentence_lens += 1
        packed_input = pack_padded_sequence(txt_feature, sentence_lens.numpy())
        packed_outputs, _ = self.lstm(packed_input)
        txt_feature, _ = pad_packed_sequence(packed_outputs)

        txt_feature = txt_feature.permute(1, 0, 2)  # batch_size * seq_len * hidden_dimension*2
        txt_output = self.projection(txt_feature)

        return txt_output.permute(1, 0, 2), img_output  # len*bs*4

