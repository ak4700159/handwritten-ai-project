# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
import pickle as pickle
import numpy as np
import random
import os
import torch
from .utils import pad_seq, bytes_to_file, read_split_image, round_function
from .utils import shift_and_resize_image, normalize_image, centering_image

# 이 코드는 주로 이미지 데이터를 로드하고 전처리하는 기능을 담당하며, 딥러닝 모델 훈련을 위한 데이터 준비 과정을 다루고 있습니다.
def get_batch_iter(examples, batch_size, augment, with_charid=False):
    # the transpose ops requires deterministic
    # batch size, thus comes the padding
    padded = pad_seq(examples, batch_size)

    def process(img):
        img = bytes_to_file(img)
        try:
            img_A, img_B = read_split_image(img)
            if augment:
                # augment the image by:
                # 1) enlarge the image
                # 2) random crop the image back to its original size
                # NOTE: image A and B needs to be in sync as how much
                # to be shifted
                w, h = img_A.shape
                multiplier = random.uniform(1.00, 1.20)
                # add an eps to prevent cropping issue
                nw = int(multiplier * w) + 1
                nh = int(multiplier * h) + 1
                shift_x = int(np.ceil(np.random.uniform(0.01, nw - w)))
                shift_y = int(np.ceil(np.random.uniform(0.01, nh - h)))
                img_A = shift_and_resize_image(img_A, shift_x, shift_y, nw, nh)
                img_B = shift_and_resize_image(img_B, shift_x, shift_y, nw, nh)
            img_A = normalize_image(img_A)
            img_A = img_A.reshape(1, len(img_A), len(img_A[0]))
            img_B = normalize_image(img_B)
            img_B = img_B.reshape(1, len(img_B), len(img_B[0]))
            return np.concatenate([img_A, img_B], axis=0)
        finally:
            img.close()
            
    def batch_iter(with_charid=with_charid):
        for i in range(0, len(padded), batch_size):
            batch = padded[i: i + batch_size]
            labels = [e[0] for e in batch]
            if with_charid:
                charid = [e[1] for e in batch]
                image = [process(e[2]) for e in batch]
                image = np.array(image).astype(np.float32)
                image = torch.from_numpy(image)
                # stack into tensor
                yield [labels, charid, image]
            else:
                image = [process(e[1]) for e in batch]
                image = np.array(image).astype(np.float32)
                image = torch.from_numpy(image)
                # stack into tensor
                yield [labels, image]

    return batch_iter(with_charid=with_charid)

# PickledImageProvider:
# 피클된 이미지 데이터를 로드하는 클래스입니다.
class PickledImageProvider(object):
    def __init__(self, obj_path, verbose):
        self.obj_path = obj_path
        self.verbose = verbose
        self.examples = self.load_pickled_examples()

    def load_pickled_examples(self):
        with open(self.obj_path, "rb") as of:
            examples = list()
            while True:
                try:
                    # 저장된 객체를 불러온다.
                    e = pickle.load(of)
                    examples.append(e)
                except EOFError:
                    break
                except Exception:
                    pass
            if self.verbose:
                print("unpickled total %d examples" % len(examples))
            return examples

# TrainDataProvider:
# 훈련 및 검증 데이터를 관리하는 클래스입니다.
# 데이터 필터링, 배치 생성, 라벨 관리 등의 기능을 제공합니다.
# 즉, 75,000개의 글자 데이터를 생성 후 가공하는 역할. -> save_fixed_sample를 통해 fixed_label, fixed_label, fixed_label 생성
class TrainDataProvider(object):
    def __init__(self, data_dir, train_name="train.pkl", val_name="val.pkl", \
                 filter_by_font=None, filter_by_charid=None, verbose=True, val=True):
        self.data_dir = data_dir
        self.filter_by_font = filter_by_font
        self.filter_by_charid = filter_by_charid
        # pickle_examples 함수를 통해 생성된 train val 둘 다 같은 경로에 저장되어 있어야됨
        self.train_path = os.path.join(self.data_dir, train_name)
        self.val_path = os.path.join(self.data_dir, val_name)

        # 기본적으로 train.obj를 불러온다.
        self.train = PickledImageProvider(self.train_path, verbose)
        if val:
            self.val = PickledImageProvider(self.val_path, verbose)

        if self.filter_by_font:
            if verbose:
                print("filter by label ->", filter_by_font)
            self.train.examples = [e for e in self.train.examples if e[0] in self.filter_by_font]
            if val:
                self.val.examples = [e for e in self.val.examples if e[0] in self.filter_by_font]

        if self.filter_by_charid:
            if verbose:
                print("filter by char ->", filter_by_charid)
            self.train.examples = [e for e in self.train.examples if e[1] in filter_by_charid]
            if val:
                self.val.examples = [e for e in self.val.examples if e[1] in filter_by_charid]

        if verbose:
            if val:
                print("train examples -> %d, val examples -> %d" % (len(self.train.examples), len(self.val.examples)))
            else:
                print("train examples -> %d" % (len(self.train.examples)))

                
    def get_train_iter(self, batch_size, shuffle=True, with_charid=False):
        training_examples = self.train.examples[:]
        if shuffle:
            np.random.shuffle(training_examples)
           
        if with_charid:
            return get_batch_iter(training_examples, batch_size, augment=True, with_charid=True)
        else:
            return get_batch_iter(training_examples, batch_size, augment=True)

        
    def get_val_iter(self, batch_size, shuffle=True, with_charid=False):
        """
        Validation iterator runs forever
        """
        val_examples = self.val.examples[:]
        if shuffle:
            np.random.shuffle(val_examples)
        if with_charid:
            return get_batch_iter(val_examples, batch_size, augment=True, with_charid=True)
        else:
            return get_batch_iter(val_examples, batch_size, augment=True)

        
    def compute_total_batch_num(self, batch_size):
        """Total padded batch num"""
        return int(np.ceil(len(self.train.examples) / float(batch_size)))

    
    def get_all_labels(self):
        """Get all training labels"""
        return list({e[0] for e in self.train.examples})

    
    def get_train_val_path(self):
        return self.train_path, self.val_path
    

# 해당 함수는 고정된 샘플 데이터를 저장하는 함수입니다.
# 이미지를 중앙에 위치시키고 크기를 조정하는 작업을 수행합니다.
# sample_size = 배치사이즈 = 16
# img_size = 이미지 크기 = 126
# data_dir = train.pkl, val.pkl 데이터 경로
# save_dir = fixed_source.pkl, fixed_target.pkl, fixed_label.pkl = 저장될 경로
def save_fixed_sample(sample_size, img_size, data_dir, save_dir, \
                      val=False, verbose=True, with_charid=True, resize_fix=90):
    # 
    data_provider = TrainDataProvider(data_dir, verbose=verbose, val=val)
    if not val:
        train_batch_iter = data_provider.get_train_iter(sample_size, with_charid=with_charid)
    else:
        train_batch_iter = data_provider.get_val_iter(sample_size, with_charid=with_charid)
        
    for batch in train_batch_iter:
        if with_charid:
            font_ids, _, batch_images = batch
        else:
            font_ids, batch_images = batch
        fixed_batch = batch_images.cuda()
        fixed_source = fixed_batch[:, 1, :, :].reshape(sample_size, 1, img_size, img_size)
        fixed_target = fixed_batch[:, 0, :, :].reshape(sample_size, 1, img_size, img_size)

        # centering
        # zip() 함수는 여러 개의 순회 가능한 객체를 인자로 받고 각 객체가 담고 있는 원소를 튜플의 형태로 차례로
        # 접근할 수 있는 반복자를 반환한다.
        for idx, (image_S, image_T) in enumerate(zip(fixed_source, fixed_target)):
            image_S = image_S.cpu().detach().numpy().reshape(img_size, img_size)
            image_S = np.array(list(map(round_function, image_S.flatten()))).reshape(128, 128)
            image_S = centering_image(image_S, resize_fix=90)
            fixed_source[idx] = torch.tensor(image_S).view([1, img_size, img_size])
            image_T = image_T.cpu().detach().numpy().reshape(img_size, img_size)
            image_T = np.array(list(map(round_function, image_T.flatten()))).reshape(128, 128)
            image_T = centering_image(image_T, resize_fix=resize_fix)
            fixed_target[idx] = torch.tensor(image_T).view([1, img_size, img_size])

        # fond_ids  = label 을 의미
        fixed_label = np.array(font_ids)
        source_with_label = [(label, image_S.cpu().detach().numpy()) \
                             for label, image_S in zip(fixed_label, fixed_source)]
        source_with_label = sorted(source_with_label, key=lambda i: i[0])
        target_with_label = [(label, image_T.cpu().detach().numpy()) \
                             for label, image_T in zip(fixed_label, fixed_target)]
        target_with_label = sorted(target_with_label, key=lambda i: i[0])
        fixed_source = torch.tensor(np.array([i[1] for i in source_with_label])).cuda()
        fixed_target = torch.tensor(np.array([i[1] for i in target_with_label])).cuda()
        fixed_label = sorted(fixed_label)

        # [라벨 : 원본 고딕체 글자 : 다양한 폰트의 글자] 25 * 2000 = 50,000
        torch.save(fixed_source, os.path.join(save_dir, 'fixed_source.pkl'))
        torch.save(fixed_target, os.path.join(save_dir, 'fixed_target.pkl'))
        torch.save(fixed_label, os.path.join(save_dir, 'fixed_label.pkl'))
        return