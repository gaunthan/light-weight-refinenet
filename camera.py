#!/usr/bin/env python3

from models.resnet import rf_lw50, rf_lw101, rf_lw152
from utils.helpers import prepare_img
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import sys

cmap = np.load('./utils/cmap.npy')
has_cuda = torch.cuda.is_available()
n_classes = 60

mnet = rf_lw152(n_classes, pretrained=True).eval()
if has_cuda:
    mnet = mnet.cuda()

if len(sys.argv) <= 1:
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture(sys.argv[1])

if not cap.isOpened():
    print("Could not open camera")
    sys.exit(-1)

rval, frame = cap.read()
orig_size = frame.shape[:2][::-1]
while rval:
    img_inp = torch.tensor(prepare_img(frame).transpose(2, 0, 1)[None]).float()
    if has_cuda:
        img_inp = img_inp.cuda()
    segm = mnet(img_inp)[0].data.cpu().numpy().transpose(1, 2, 0)
    segm = cv2.resize(segm, orig_size, interpolation=cv2.INTER_CUBIC)
    segm = cmap[segm.argmax(axis=2).astype(np.uint8)]    
    result = np.hstack((frame, segm))
    
    cv2.imshow("Raw & Segmentation", result)
    key = cv2.waitKey(1) & 0xff
    if key == ord('q'):
        break

    rval, frame = cap.read()

cap.release()