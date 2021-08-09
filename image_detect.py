import cv2
import numpy as np
from matplotlib import pyplot as plt
import argparse
import os
import sys
import time

def MaxMinNormalization(x):
    Min = np.min(x)
    Max = np.max(x)
    x = ((x - Min) / (Max - Min)*255).astype(np.uint8)
    return x

def Z_ScoreNormalization(x):
    mean = np.mean(x) 
    sigma = np.std(x)
    x = (x - mean) / sigma;
    return x

def single_scale_search_fun(imgS, imgT, MAX_LEN = 640, window_step = 20, window_len = 100):
    print(f'call {sys._getframe().f_code.co_name}')

    starter = time.time()

    hS, wS, _ = imgS.shape
    if hS > wS:
        max_len, max_axi = hS, 0
    else:
        max_len, max_axi = wS, 1
    if max_len > MAX_LEN:
        if max_axi:
            new_w = MAX_LEN
            new_h = int(hS/wS*MAX_LEN)
        else:
            new_h = MAX_LEN
            new_w = int(wS/hS*MAX_LEN)
        imgS = cv2.resize(imgS, (new_w, new_h))
        imgT = cv2.resize(imgT, (new_w, new_h))    

    grayS = cv2.cvtColor(imgS, cv2.COLOR_BGR2GRAY)
    grayT = cv2.cvtColor(imgT, cv2.COLOR_BGR2GRAY)

    grayS = Z_ScoreNormalization(grayS)
    grayT = Z_ScoreNormalization(grayT)

    grayS = MaxMinNormalization(grayS)
    grayT = MaxMinNormalization(grayT)

    
    print(f'Resize and Normalization time: {time.time() - starter}')
    # window = 100
        # Resize and Normalization time: 0.012996435165405273
        # search time: 1.9573242664337158
    starter = time.time()
    # plt.imshow(grayB)
    # plt.show()
    # 获取图片A的大小
    height, width = grayS.shape

    # 取局部图像，寻找匹配位置
    result_window = np.zeros((height, width), dtype=imgS.dtype)
    step = window_step
    LEN = window_len
    # LEN = 50
    for start_y in range(0, height-LEN, step):
        for start_x in range(0, width-LEN, step):
            # window = grayA[start_y:start_y+LEN, start_x:start_x+LEN]
            # match = cv2.matchTemplate(grayB, window, cv2.TM_CCOEFF_NORMED)
            # match = cv2.matchTemplate(grayB, window, cv2.TM_CCORR_NORMED)
            # match = cv2.matchTemplate(grayB, window, cv2.TM_CCORR)

            window = grayS[start_y:start_y+LEN, start_x:start_x+LEN]
            match = cv2.matchTemplate(grayT, window, cv2.TM_CCORR_NORMED)
            _, _, _, max_loc = cv2.minMaxLoc(match)
            matched_window = grayT[max_loc[1]:max_loc[1]+LEN, max_loc[0]:max_loc[0]+LEN]
            result = cv2.absdiff(window, matched_window)
            result_window[start_y:start_y+LEN, start_x:start_x+LEN] = result
    
    print(f'search time: {time.time() - starter}')
    return result_window, imgS, imgT

def fast_single_scale_search_fun(imgS, imgT, MAX_LEN = 640, window_step = 20, window_len = 100, window_margin = 50):
    print(f'call {sys._getframe().f_code.co_name}')

    starter = time.time()

    hS, wS, _ = imgS.shape
    if hS > wS:
        max_len, max_axi = hS, 0
    else:
        max_len, max_axi = wS, 1
    if max_len > MAX_LEN:
        if max_axi:
            new_w = MAX_LEN
            new_h = int(hS/wS*MAX_LEN)
        else:
            new_h = MAX_LEN
            new_w = int(wS/hS*MAX_LEN)
        imgS = cv2.resize(imgS, (new_w, new_h))
        imgT = cv2.resize(imgT, (new_w, new_h))    

    grayS = cv2.cvtColor(imgS, cv2.COLOR_BGR2GRAY)
    grayT = cv2.cvtColor(imgT, cv2.COLOR_BGR2GRAY)

    grayS = Z_ScoreNormalization(grayS)
    grayT = Z_ScoreNormalization(grayT)

    grayS = MaxMinNormalization(grayS)
    grayT = MaxMinNormalization(grayT)

    
    print(f'Resize and Normalization time: {time.time() - starter}')
    # window = 100
        # Resize and Normalization time: 0.012996435165405273
        # search time: 1.9573242664337158
    starter = time.time()
    # plt.imshow(grayB)
    # plt.show()
    # 获取图片A的大小
    height, width = grayS.shape

    # 取局部图像，寻找匹配位置
    result_window = np.zeros((height, width), dtype=imgS.dtype)
    step = window_step
    # LEN = 50
    for start_y in range(0, height-window_len, step):
        for start_x in range(0, width-window_len, step):
            # window = grayA[start_y:start_y+LEN, start_x:start_x+LEwindow_lenN]
            # match = cv2.matchTemplate(grayB, window, cv2.TM_CCOEFF_NORMED)
            # match = cv2.matchTemplate(grayB, window, cv2.TM_CCORR_NORMED)
            # match = cv2.matchTemplate(grayB, window, cv2.TM_CCORR)

            window = grayS[start_y:start_y+window_len, start_x:start_x+window_len]
            Tstart_y = start_y - window_margin if start_y - window_margin > 0 else 0
            Tend_y   = start_y + window_margin + window_len if start_y + window_margin + window_len < height else height
            
            Tstart_x = start_x - window_margin if start_x - window_margin > 0 else 0
            Tend_x   = start_x + window_margin + window_len if start_x + window_margin + window_len < width else width
            grayT_block = grayT[Tstart_y:Tend_y, Tstart_x:Tend_x]
            match = cv2.matchTemplate(grayT_block, window, cv2.TM_CCORR_NORMED)
            _, _, _, max_loc = cv2.minMaxLoc(match)
            matched_window = grayT_block[max_loc[1]:max_loc[1]+window_len, max_loc[0]:max_loc[0]+window_len]
            result = cv2.absdiff(window, matched_window)
            result_window[start_y:start_y+window_len, start_x:start_x+window_len] = result
    
    print(f'fast search time: {time.time() - starter}')
    return result_window, imgS, imgT


def multi_scale_search_fun(imgS, imgT, MAX_LEN = 640, window_step = 20, window_len = 100, scope=0.3, num=5):
    print(f'call {sys._getframe().f_code.co_name}')

    hS, wS, _ = imgS.shape
    if hS > wS:
        max_len, max_axi = hS, 0
    else:
        max_len, max_axi = wS, 1
    if max_len > MAX_LEN:
        if max_axi:
            new_w = MAX_LEN
            new_h = int(hS/wS*MAX_LEN)
        else:
            new_h = MAX_LEN
            new_w = int(wS/hS*MAX_LEN)
        imgS = cv2.resize(imgS, (new_w, new_h))
        imgT = cv2.resize(imgT, (new_w, new_h))    

    grayS = cv2.cvtColor(imgS, cv2.COLOR_BGR2GRAY)
    grayT = cv2.cvtColor(imgT, cv2.COLOR_BGR2GRAY)

    grayS = Z_ScoreNormalization(grayS)
    grayT = Z_ScoreNormalization(grayT)

    grayS = MaxMinNormalization(grayS)
    grayT = MaxMinNormalization(grayT)

    # plt.imshow(grayB)
    # plt.show()
    # 获取图片A的大小
    height, width = grayS.shape

    # 取局部图像，寻找匹配位置
    result_window = np.zeros((height, width), dtype=imgS.dtype)
    step = window_step
    # LEN = 50
    for start_y in range(0, height-window_len, step):
        for start_x in range(0, width-window_len, step):
            window = grayT[start_y:start_y+window_len, start_x:start_x+window_len]
            # window = grayA[start_y:start_y+LEN, start_x:start_x+LEN]
            # match = cv2.matchTemplate(grayB, window, cv2.TM_CCOEFF_NORMED)
            # match = cv2.matchTemplate(grayB, window, cv2.TM_CCORR_NORMED)
            # match = cv2.matchTemplate(grayB, window, cv2.TM_CCORR)

            H, W = grayS.shape[:2]
            h, w = window.shape[:2]
            found = (0, 0, np.zeros((h, w), dtype=imgS.dtype), 0, 0)
            for scale in np.linspace(1-scope, 1+scope, num)[::-1]:
                resized = cv2.resize(grayS, (int(W * scale), int(H * scale)))
                ratio = W / float(resized.shape[1])
                if resized.shape[0] < h or resized.shape[1] < w:
                    break
                res = cv2.matchTemplate(resized, window, cv2.TM_CCORR_NORMED)

                loc = np.where(res >= res.max())
                pos_h, pos_w = list(zip(*loc))[0]

                if found is None or res.max() > found[-1]:
                    found = (pos_h, pos_w, resized.copy(), ratio, res.max())

            # if found is None: return (0,0,0,0,0)

            pos_h, pos_w, resized, ratio, score = found
            # start_h, start_w = int(pos_h * ratio), int(pos_w * ratio)
            # end_h, end_w = int((pos_h + h) * ratio), int((pos_w + w) * ratio)
            # matched_window = grayS[start_h:end_h, start_w:end_w]
            # print(pos_h, h, pos_w, w)
            matched_window = resized[pos_h:pos_h + h, pos_w:pos_w + w]
            # print('window.shape', window.shape)
            # print('matched_window.shape', matched_window.shape)
            result = cv2.absdiff(window, matched_window)
            result_window[start_y:start_y+window_len, start_x:start_x+window_len] = result

    return result_window, imgS, imgT

def diff_by_minus_fun(imgS, imgT, MAX_LEN = 640):
    print(f'call {sys._getframe().f_code.co_name}')
    if not imgS.shape == imgT.shape:
        print('imgS.shape', imgS.shape)
        print('imgT.shape', imgT.shape)
        imgT = cv2.resize(imgT, (imgS.shape[1], imgS.shape[0]))

    hS, wS, _ = imgS.shape
    if hS > wS:
        max_len, max_axi = hS, 0
    else:
        max_len, max_axi = wS, 1
    if max_len > MAX_LEN:
        if max_axi:
            new_w = MAX_LEN
            new_h = int(hS/wS*MAX_LEN)
        else:
            new_h = MAX_LEN
            new_w = int(wS/hS*MAX_LEN)
        imgS = cv2.resize(imgS, (new_w, new_h))
        imgT = cv2.resize(imgT, (new_w, new_h))    

    grayS = cv2.cvtColor(imgS, cv2.COLOR_BGR2GRAY)
    grayT = cv2.cvtColor(imgT, cv2.COLOR_BGR2GRAY)

    grayS = Z_ScoreNormalization(grayS)
    grayT = Z_ScoreNormalization(grayT)

    grayS = MaxMinNormalization(grayS)
    grayT = MaxMinNormalization(grayT)

    result_window = cv2.absdiff(grayS, grayT)
    return result_window, imgS, imgT

def mark_diff_areas_fun(imgS, imgT, result_window, thresh = 80):
    print(f'call {sys._getframe().f_code.co_name}')
    assert imgS.shape[:-1] == result_window.shape
    # 用四边形圈出不同部分
    _, result_window_bin = cv2.threshold(result_window, thresh, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(result_window_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print('contours.shape', contours)
    imgC = imgS.copy()
    for contour in contours:
        if contour.shape[0] == 1:
            continue
        # print('contour.shape', contour.shape)
        # print('contour', contour)
        min_value = np.nanmin(contour, 0)
        max_value = np.nanmax(contour, 0)
        loc1 = (min_value[0][0], min_value[0][1])
        loc2 = (max_value[0][0], max_value[0][1])
        rec_h = loc2[1] - loc1[1]
        rec_w = loc2[0] - loc1[0]
        # print(f'loc1: {loc1}, loc2: {loc2}')
        if rec_h <= 2 or  rec_w <= 2 or rec_w*rec_h <= 32:
            continue
        cv2.rectangle(imgC, loc1, loc2, 255, 2)
        # print()
    return imgC
    

def plot_all_fun(imgS, imgT, result_window, imgC, savefig_path = None, result_minus = None):
    print(f'call {sys._getframe().f_code.co_name}')

    plt.subplot(2, 3, 1), plt.imshow(cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)), plt.title('Source'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 3, 3), plt.imshow(cv2.cvtColor(imgT, cv2.COLOR_BGR2RGB)), plt.title('Target'), plt.xticks([]), plt.yticks([])
    
    plt.subplot(2, 3, 4), plt.imshow(cv2.cvtColor(imgC, cv2.COLOR_BGR2RGB)), plt.title('Mark of CCORR Diff'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 3, 5), plt.imshow(result_window), plt.title('Heatmap of Diff by CCORR'), plt.xticks([]), plt.yticks([])
    if result_minus.any():
        plt.subplot(2, 3, 6), plt.imshow(result_minus), plt.title('Heatmap of Diff by Minus'), plt.xticks([]), plt.yticks([])
    # plt.subplots_adjust(top=-1,bottom=-2,left=0,right=1,hspace=0.2,wspace=0)
    plt.subplots_adjust(wspace=0)
    if savefig_path:
        print(f'saving {savefig_path}')
        figure = plt.gcf()  
        figure.set_size_inches(16, 9)
        plt.savefig(savefig_path, dpi=1200, bbox_inches='tight')
        # plt.savefig(savefig_path)
    plt.show()
    # plt.close()

def matchAB(args):
    S_filepath      = args.source_image
    T_filepath      = args.target_image
    in_img_filepath = args.in_img_filepath
    in_img_dir      = args.in_img_dir
    out_img_dir     = args.out_img_dir

    if out_img_dir and not os.path.exists(out_img_dir):
        os.makedirs(out_img_dir) 
    
    MAX_LEN = 640
    window_step = 20
    window_len = 200
    thresh = 128

    if S_filepath and T_filepath:
        # 读取图像数据
        imgS = cv2.imread(S_filepath)
        imgT = cv2.imread(T_filepath)
        comp_name = S_filepath.split('\\')[-1].split('.')[0] + '__' + T_filepath.split('\\')[-1].split('.')[0] +'.jpg'

    elif in_img_filepath and os.path.exists(in_img_filepath):
        img = cv2.imread(in_img_filepath)
        H, W, _ = img.shape
        imgS = img[:H//2, :W//2]
        imgT = img[:H//2, W//2:]
        comp_name = in_img_filepath.split('\\')[-1].split('.')[0] + '_compared.jpg'

    elif in_img_dir and os.path.exists(in_img_dir):
        pass
    
    else:
        pass

    savefig_path = None
    if out_img_dir:
        savefig_path = os.path.join(out_img_dir, comp_name)

    result_search, imgS, imgT = single_scale_search_fun(imgS, imgT, MAX_LEN = MAX_LEN, window_step = window_step, window_len = window_len)
    # result_window, imgS, imgT = multi_scale_search_fun(imgS, imgT, MAX_LEN = MAX_LEN, window_step = window_step, window_len = window_len)
    result_search, imgS, imgT = fast_single_scale_search_fun(imgS, imgT, MAX_LEN = MAX_LEN, window_step = window_step, window_len = window_len)
    result_minus, _, _ = diff_by_minus_fun(imgS, imgT, MAX_LEN = MAX_LEN)
    imgC = mark_diff_areas_fun(imgS, imgT, result_search, thresh = thresh)
    plot_all_fun(imgS, imgT, result_search, imgC, savefig_path = savefig_path, result_minus = result_minus)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source_image', type=str, default=None)
    parser.add_argument('-t', '--target_image', type=str, default=None)

    parser.add_argument('-if', '--in_img_filepath',  type=str, default=None)

    parser.add_argument('-im', '--in_img_dir',  type=str, default=None)
    parser.add_argument('-om', '--out_img_dir', type=str, default=None)

    args = parser.parse_args()

    # match_aligned_images(args)
    matchAB(args)