# https://github.com/kaonashi-tyc/zi2zi

# -*- coding: utf-8 -*-

import argparse
from functools import cache
import utils
import sys
import glob
import numpy as np
import io, os
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import collections

# SRC = 폰트 고딕체
SRC_PATH = './fonts/source/'
# TRG = 총 56가지의 폰트
TRG_PATH = './fonts/target/'

# 출력되는 이미지는 스타일글자|고딕체글자 의 쌍으로 이미지가 저장.
OUTPUT_PATH = './dataset/'
# canvas_size = 128
# 최종 만들어질 이미지(example_img)는 128 * 256 이미지



# 단일 문자를 지정된 폰트로 그리는 함수.
def draw_single_char(ch, font, canvas_size):
    # L = 흑백이미지를 의미 
    # 단순 128 * 128 사이즈의 흑백 이미지 생성
    image = Image.new('L', (canvas_size, canvas_size), color=255)
    # 흑백 이미지
    drawing = ImageDraw.Draw(image)

    _, _, w, h = font.getbbox(ch)
    drawing.text(
        ((canvas_size-w)/2, (canvas_size-h)/2),
        ch,
        fill=(0),
        font=font
    )
    flag = np.sum(np.array(image))
    
    # 해당 font에 글자가 없으면 return None
    # 즉 이때 흰색을 의미함.
    if flag == 255 * 128 * 128:
        return None
    
    return image

def draw_example(ch, src_font, dst_font, canvas_size):
    # 특정 스타일로 만들어낸 단일 글자 생성(target image)
    dst_img = draw_single_char(ch, dst_font, canvas_size)
    if not dst_img:
        return None
    # 열과 행을 슬라이싱 후 crop -> resize -> padding 과정을 거치게 된다.
    dst_img = utils.centering_image(np.array(dst_img), pad_value = 255)
    dst_img = Image.fromarray(dst_img.astype(np.uint8))

    # 해당 font에 글자가 없으면 return None
    
    # 고딕체 스타일로 만들어낸 단일 글자 생성(source image)
    src_img = draw_single_char(ch, src_font, canvas_size)
    src_img = utils.centering_image(np.array(src_img))
    src_img = Image.fromarray(src_img.astype(np.uint8))
    example_img = Image.new("RGB", (canvas_size * 2, canvas_size), (255, 255, 255)).convert('L')
    # 왼쪽엔 특정 스타일로 만들어낸 단일 글자, 오른쪽엔 고딕체로 만들어낸 단일 글자
    example_img.paste(dst_img, (0, 0))
    example_img.paste(src_img, (canvas_size, 0))   
    
    return example_img

# ch = 그릴 문자
# src_font = 고딕체
# canvas_size = 128
# dst_path = 이미 생성된 내 손글씨 데이터를 의미 하는 걸로 보임.
def draw_handwriting(ch, src_font, canvas_size, dst_folder, label, count):
    dst_path = dst_folder + "%d_%04d" % (label, count) + ".png"
    dst_img = Image.open(dst_path)
    src_img = draw_single_char(ch, src_font, canvas_size)
    example_img = Image.new("RGB", (canvas_size * 2, canvas_size), (255, 255, 255)).convert('L')
    example_img.paste(dst_img, (0, 0))
    example_img.paste(src_img, (canvas_size, 0))
    return example_img

