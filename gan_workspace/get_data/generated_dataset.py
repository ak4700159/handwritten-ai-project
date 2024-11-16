from time import sleep
import font2img as ft
from PIL import ImageFont
import os
import random
import package
import gc
from functools import cache, lru_cache
import threading



FONT_DATASET_PATH = "./font_dataset"
MAX_FONT_COUNT = 25
MAX_RAMDOM_SELECTED_WORD = 2000

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


# 폰트별 MAX_RAMDOM_SELECTED_WORD 만큼 이미지를 생성한다
src_path = f"{ft.SRC_PATH}/source_font.ttf"
def generated_dataset(font_id):
    if not os.path.exists(FONT_DATASET_PATH):
        os.mkdir(FONT_DATASET_PATH)

    # 사용된 문자가 저장된다. 폰트가 바뀔때마다 리셋
    ch_list = set()
    count = 0
    while True:
        if(count >= MAX_RAMDOM_SELECTED_WORD) : break
        ch = generate_random_hangul_and_ascii()
        if ch in ch_list : continue

        if font_id < 10 :
            trg_path = ft.TRG_PATH + "0" + str(font_id) + ".ttf"
        else :
            trg_path = f"{ft.TRG_PATH}{font_id}.ttf"
        
        # 두번재 파라미터는 글자크기를 의미
        trg_font = ImageFont.truetype(trg_path, 90)
        src_font = ImageFont.truetype(src_path, 90)

        example_img = ft.draw_example(ch, src_font, trg_font, 128)
        if example_img == None: continue

        # 이미지가 저장될 때 사용된 폰트 번호 _ 식별할 수 있는 문자값
        # 동일한 파일 존재시 처음부터.
        example_img.save(f"{FONT_DATASET_PATH}/{font_id}_{ord(ch)}.png", 'png', optimize=True)
        ch_list.add(ch)
        count += 1
    print(f'폰트 {font_id} 생성')


def main():
    threads = [] 
    for font_idx in range(1, MAX_FONT_COUNT + 1):
        t = threading.Thread(target=generated_dataset, args=(font_idx,))
        t.start()
        threads.append(t)

    for font_idx in range(MAX_FONT_COUNT):
        threads[font_idx].join()   
        
if __name__ == "__main__":
    main()
    # 생성한 학습용 데이터를 train / value 데이터로 나눈다 (4:1) --> .pkl 파일에 저장
    package.pickle_examples('./font_dataset', '../dataset/train.pkl', '../dataset/val.pkl', with_charid=True)

