# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import

import argparse
import glob
import os
import pickle as pickle
import random


# from_dir = './font_dataset
# train_path = "../dataset/train"
# val_path = "../dataset/val"
# train_val_split = 트레인 데이터와 평가 데이터의 비율을 의미
# with_charid = 제목에 char_id가 표시되어 있는지
def pickle_examples(from_dir, train_path, val_path, train_val_split=0.2, with_charid=False):
    """
    Compile a list of examples into pickled format, so during
    the training, all io will happen in memory
    """
    paths = glob.glob(os.path.join(from_dir, "*.png"))
    with open(train_path, 'wb') as ft:
        with open(val_path, 'wb') as fv:
            print('all data num:', len(paths))
            c = 1
            val_count = 0
            train_count = 0
            if with_charid:
                print('pickle with charid')
                # paths 경로 안에는 font_dataset 안의 파일 이름 모두 담기게 된다
                for p in paths:
                    c += 1
                    label = int(os.path.basename(p).split("_")[0])
                    charid = int(os.path.basename(p).split("_")[1].split(".")[0])
                    # .png 파일 하나를 열고 이미지를 읽고 
                    # label(어떤 폰트 사용했는지), charid(문자 식별 번호), img_bytes(이미지 데이터)
                    with open(p, 'rb') as f:
                        img_bytes = f.read()
                        example = (label, charid, img_bytes)
                        r = random.random()
                        # 설정한 비율데로 train val 파일에 저장된다.
                        if r < train_val_split:
                            pickle.dump(example, fv)
                            val_count += 1
                            if val_count % 10000 == 0:
                                print("%d imgs saved in val.obj" % val_count)
                        else:
                            pickle.dump(example, ft)
                            train_count += 1
                            if train_count % 10000 == 0:
                                print("%d imgs saved in train.obj" % train_count)
                print("%d imgs saved in val.obj, end" % val_count)
                print("%d imgs saved in train.obj, end" % train_count)
            else:
                for p in paths:
                    c += 1
                    label = int(os.path.basename(p).split("_")[0])
                    with open(p, 'rb') as f:
                        img_bytes = f.read()
                        example = (label, img_bytes)
                        r = random.random()
                        if r < train_val_split:
                            pickle.dump(example, fv)
                            val_count += 1
                            if val_count % 10000 == 0:
                                print("%d imgs saved in val.obj" % val_count)
                        else:
                            pickle.dump(example, ft)
                            train_count += 1
                            if train_count % 10000 == 0:
                                print("%d imgs saved in train.obj" % train_count)
                print("%d imgs saved in val.obj, end" % val_count)
                print("%d imgs saved in train.obj, end" % train_count)
            return
        
# 특정 폰트와 특정 글자에 대해서만 학습시키기 위해 데이터를 추출한 것
def pickle_interpolation_data(from_dir, save_path, char_ids, font_filter):
    paths = glob.glob(os.path.join(from_dir, "*.png"))
    with open(save_path, 'wb') as ft:
        c = 0
        for p in paths:
            charid = int(p.split('/')[-1].split('.')[0].split('_')[1])
            label = int(os.path.basename(p).split("_")[0])
            if (charid in char_ids) and (label in font_filter):
                c += 1
                with open(p, 'rb') as f:
                    img_bytes = f.read()
                    example = (label, charid, img_bytes)
                    pickle.dump(example, ft)
        print('data num:', c)
        return