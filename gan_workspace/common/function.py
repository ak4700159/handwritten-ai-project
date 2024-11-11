# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset


def batch_norm(c_out, momentum=0.1):
    return nn.BatchNorm2d(c_out, momentum=momentum)


# LeakyReLU 활성화 함수는 relu함수와 다르게 임계치가 0보다 작으면 0.01을 곱한다.
# 이를 통해 relu함수에서 발생한 knock out 문제를 거의 해결한다. 하지만 속도면에서 relu가 더 좋다

# BatchNorm2d : 입력값이 4차원(높이 * 가로 * 세로 * 개수)인 경우
def conv2d(c_in, c_out, k_size=3, stride=2, pad=1, dilation=1, bn=True, lrelu=True, leak=0.2):
    layers = []
    if lrelu:
        layers.append(nn.LeakyReLU(leak))
    layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)


def deconv2d(c_in, c_out, k_size=3, stride=1, pad=1, dilation=1, bn=True, dropout=False, p=0.5):
    layers = []
    layers.append(nn.LeakyReLU(0.2))
    layers.append(nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    if dropout:
        layers.append(nn.Dropout(p))
    return nn.Sequential(*layers)


def lrelu(leak=0.2):
    return nn.LeakyReLU(leak)


def dropout(p=0.2):
    return nn.Dropout(p)


def fc(input_size, output_size):
    return nn.Linear(input_size, output_size)
    
    # stddev = 표준 편차
    # 임베딩 파일이 없으면 초기값 넣어 생성한다?
def init_embedding(embedding_num, embedding_dim, stddev=0.01):
    # 정규분포를 생성하는 난수 함수, 정규분포에서 무작위 난수를 생성하여 텐서를 반환
    # embedding_num * embedding_dim 차원의 텐서가 생성
    embedding = torch.randn(embedding_num, embedding_dim) * stddev
    
    embedding = embedding.reshape((embedding_num, 1, 1, embedding_dim))
    return embedding

#카테고리 백터 가공?
def embedding_lookup(embeddings, embedding_ids, GPU=False):
    batch_size = len(embedding_ids)
    embedding_dim = embeddings.shape[3]
    local_embeddings = []
    for id_ in embedding_ids:
        if GPU:
            local_embeddings.append(embeddings[id_].cpu().numpy())
        else:
            local_embeddings.append(embeddings[id_].data.numpy()) 
    local_embeddings = torch.from_numpy(np.array(local_embeddings))
    if GPU:
        local_embeddings = local_embeddings.cuda()
    local_embeddings = local_embeddings.reshape(batch_size, embedding_dim, 1, 1)
    return local_embeddings


def interpolated_embedding_lookup(embeddings, interpolated_embedding_ids, grid):
    batch_size = len(interpolated_embedding_ids)
    interpolated_embeddings = []
    embedding_dim = embeddings.shape[3]

    for id_ in interpolated_embedding_ids:
        interpolated_embeddings.append((embeddings[id_[0]] * (1 - grid) + embeddings[id_[1]] * grid).cpu().numpy())
    interpolated_embeddings = torch.from_numpy(np.array(interpolated_embeddings)).cuda()
    interpolated_embeddings = interpolated_embeddings.reshape(batch_size, embedding_dim, 1, 1)
    return interpolated_embeddings