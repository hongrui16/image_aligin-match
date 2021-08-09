from __future__ import print_function
import cv2
import numpy as np
import argparse
import os
import sys
from matplotlib import pyplot as plt


MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15

def sift_fun(image):    
    print(f'call {sys._getframe().f_code.co_name}')

    gray_image  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)    # 颜色空间转换
    sift        = cv2.SIFT_create()
    kp          = sift.detect(gray_image,None)
    img         = cv2.drawKeypoints(gray_image,kp,None)
    kp, dst     = sift.compute(gray_image, kp)
    # plt.imshow(img)
    # plt.show()
    # print(f'finish calling {sys._getframe().f_code.co_name}')
    return img, kp, dst

def sift_kp(image):    
    print(f'call {sys._getframe().f_code.co_name}')

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)    # 颜色空间转换
    sift = cv2.xfeatures2d_SIFT.create()
    kps, des = sift.detectAndCompute(image, None)
    kp_image = cv2.drawKeypoints(gray_image, kps, None)     # 绘制关键点的函数
    return kp_image,kps,des

def get_good_match(des1, des2):
    print(f'call {sys._getframe().f_code.co_name}')

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    return good

def siftImageAlignment(img1, img2):
    
    _, kp1, des1 = sift_fun(img1)
    _, kp2, des2 = sift_fun(img2)
    # _, kp1, des1 = sift_kp(img1)
    # _, kp2, des2 = sift_kp(img2)
    goodMatch = get_good_match(des1, des2)
    if len(goodMatch) > 4:
        ptsA= np.float32([kp1[m.queryIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
        ptsB = np.float32([kp2[m.trainIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
        ransacReprojThreshold = 2
        H, status =cv2.findHomography(ptsA, ptsB, cv2.RANSAC, ransacReprojThreshold)
        # H, status =cv2.findHomography(ptsA, ptsB, cv2.RANSAC)

        imgOut = cv2.warpPerspective(img2, H, (img1.shape[1],img1.shape[0]),flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        # Draw top matches
        # imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
        # print('goodMatch', goodMatch)
        imMatches = cv2.drawMatches(img1, kp1, img2, kp2, goodMatch, None)
        # imMatches = cv2.drawMatches(img2, kp2, img1, kp1, goodMatch, None)

    return imgOut, imMatches, H


def siftResult(ref_img_filepath, tar_img_filepath, args):
    img_ref = cv2.imread(ref_img_filepath)
    img_tar = cv2.imread(tar_img_filepath)

    if args.target_width*args.target_height:
        img_ref = cv2.resize(img_ref, (args.target_width, args.target_height))
        img_tar = cv2.resize(img_tar, (args.target_width, args.target_height))

    result, imMatches, H = siftImageAlignment(img_ref, img_tar)

    img_res = np.concatenate((img_ref, result), axis=1)
    img_res = np.concatenate((img_res, imMatches), axis=0)

    if args.output_dir and os.path.exists(args.output_dir):
        # Write aligned image to disk.
        img_name = tar_img_filepath.split('\\')[-1] if '\\' in tar_img_filepath else tar_img_filepath
        new_img_name = f'siftAligned_{img_name}'
        out_img_filepath = os.path.join(args.output_dir, new_img_name)
        print("Saving aligned image : ", out_img_filepath);
        cv2.imwrite(out_img_filepath, img_res)

    plt.imshow(cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB))
    plt.show()
    # cv2.imwrite('result.jpg', result)
    # cv2.waitKey(0)



def compare(args):
    # Read reference image
    in_img_dir = args.input_dir
    out_img_dir = args.output_dir

    ref_img_filepath = args.source_image
    tar_img_filepath = args.target_image

    resize_flag = True


    if out_img_dir and not os.path.exists(out_img_dir):
        os.makedirs(out_img_dir)

    if ref_img_filepath and tar_img_filepath:
        if os.path.exists(ref_img_filepath) and os.path.exists(tar_img_filepath):
            siftResult(ref_img_filepath, tar_img_filepath, args)

    else:
        pass


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='aligment')
    parser.add_argument('-s', '--source_image', type=str, default=None)
    parser.add_argument('-t', '--target_image', type=str, default=None)

    parser.add_argument('-im', '--input_dir', type=str, default=None)
    parser.add_argument('-om', '--output_dir', type=str, default=None)
    parser.add_argument('-th', '--target_height', type=int, default=1440)
    parser.add_argument('-tw', '--target_width', type=int, default=1920)

    args = parser.parse_args()
    compare(args)
    # image = cv2.imread(args.source_image)
    # sift_fun(image)