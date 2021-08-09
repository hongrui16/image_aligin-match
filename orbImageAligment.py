from __future__ import print_function
import cv2
import numpy as np
import argparse
import os
import sys
from matplotlib import pyplot as plt


MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15

def alignImages(im1, im2):

    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    # imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    print('matches', matches)
    imMatches = cv2.drawMatches(im2, keypoints2, im1, keypoints1, matches, None)

    # cv2.imwrite("matches.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))

    return im1Reg, imMatches, h

def match_fun(imReference, tar_img_filepath, resize_flag, img_name, args):
    print(f'call {sys._getframe().f_code.co_name}')

    im = cv2.imread(tar_img_filepath, cv2.IMREAD_COLOR)
    if resize_flag and args.target_width * args.target_height:
        im = cv2.resize(im, (args.target_width, args.target_height))
        imReference = cv2.resize(imReference, (args.target_width, args.target_height))

    # Registered image will be resotred in imReg.
    # The estimated homography will be stored in h.
    imReg, imMatches, h = alignImages(im, imReference)
    img_res = np.concatenate((imReference, imReg), axis=1)
    img_res = np.concatenate((img_res, imMatches), axis=0)


    if args.output_dir and os.path.exists(args.output_dir):
        # Write aligned image to disk.
        out_img_name = img_name.replace('target_', 'aligned_')
        out_img_filepath = os.path.join(args.output_dir, out_img_name)
        print("Saving aligned image : ", out_img_filepath);
        cv2.imwrite(out_img_filepath, img_res)
    
    plt.imshow(cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB))
    plt.show()
    

def compare(args):
    # Read reference image
    in_img_dir = args.input_dir
    out_img_dir = args.output_dir

    sou_img_filepath = args.source_image
    tar_img_filepath = args.target_image

    resize_flag = True

    if out_img_dir and not os.path.exists(out_img_dir):
        os.makedirs(out_img_dir)

    if in_img_dir and os.path.exists(in_img_dir):
        img_names = os.listdir(in_img_dir)
        target_imgs = []
        for img_name in img_names:
            if 'ref_' in img_name:
                sou_img_filepath = os.path.join(in_img_dir, img_name)
            elif 'target_' in img_name:
                target_imgs.append(img_name)
            else:
                pass
        print("Reading reference image : ", sou_img_filepath)
        imReference = cv2.imread(sou_img_filepath, cv2.IMREAD_COLOR)
        for img_name in target_imgs:
            tar_img_filepath = os.path.join(in_img_dir, img_name)
            print(f"Reading {tar_img_filepath} to align");
            match_fun(imReference, tar_img_filepath, resize_flag, img_name, args)
            # Print estimated homography

    elif sou_img_filepath and tar_img_filepath:
        if os.path.exists(sou_img_filepath) and os.path.exists(tar_img_filepath):
            img_name = tar_img_filepath.split('\\')[-1] if '\\' in tar_img_filepath else tar_img_filepath
            img_name = f'aligned_{img_name}'
            imReference = cv2.imread(sou_img_filepath, cv2.IMREAD_COLOR)
            match_fun(imReference, tar_img_filepath, resize_flag, img_name, args)

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
