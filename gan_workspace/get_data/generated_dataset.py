import font2img as ft
import package
from PIL import ImageFont
import os
import random
import sys
import utils

FONT_DATASET_PATH = "./font_dataset"
MAX_FONT_COUNT = 5
MAX_RAMDOM_SELECTED_WORD = 10

# count 수 만큼 글자를 생성
def generate_random_hangul_and_ascii():
    # 랜덤한 한글 문자 1개와 그 문자의 아스키 코드를 생성합니다.
    # 한글 유니코드 범위
    start = 0xAC00  # 가
    end = 0xD7A3  # 힣
    
    # 랜덤한 한글 유니코드 선택
    char_code = random.randint(start, end)
    
    # 유니코드를 문자로 변환
    hangul_char = chr(char_code)
    
    # 문자를 아스키 코드로 변환
    # utf8_bytes = list(hangul_char.encode('utf-8'))
    
    return hangul_char


if not os.path.exists(FONT_DATASET_PATH):
    os.mkdir(FONT_DATASET_PATH)

# 폰트별 MAX_RAMDOM_SELECTED_WORD 만큼 이미지를 생성한다
src_path = f"{ft.SRC_PATH}/source_font.ttf"
for _ in range(MAX_FONT_COUNT):
    random_font_idx = random.randint(1, 46)
    count = 0
    while True:
        if(count >= MAX_RAMDOM_SELECTED_WORD) : break
        ch = generate_random_hangul_and_ascii()
        if random_font_idx < 10 :
            trg_path = ft.TRG_PATH + "0" + str(random_font_idx) + ".ttf"
        else :
            trg_path = f"{ft.TRG_PATH}{random_font_idx}.ttf"
        # 두번재 파라미터는 글자크기를 의미
        trg_font = ImageFont.truetype(trg_path, 90)
        src_font = ImageFont.truetype(src_path, 90)

        example_img = ft.draw_example(ch, src_font, trg_font, 128)
        if example_img == None: continue

        # 이미지가 저장될 때 사용된 폰트 번호 _ 식별할 수 있는 문자값
        if os.path.exists(f"{FONT_DATASET_PATH}/{random_font_idx}_{ord(ch)}.png") : continue
        example_img.save(f"{FONT_DATASET_PATH}/{random_font_idx}_{ord(ch)}.png", 'png', optimize=True)
        count += 1

package.pickle_examples('./font_dataset', '../dataset/train.pkl', '../dataset/val.pkl', with_charid=True)