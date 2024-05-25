import cv2
import os
import numpy as np

def visualize(input, name, save_dir):
    # input shape: [batch, h, w]
    # name: list, len(list) = b
    b = input.shape[0]
    for idx in range(b):
        pic_name = name[idx]
        save_img = np.round((input * 255).numpy()[idx])
        pic_path = os.path.join(save_dir, pic_name + '.png')
        cv2.imwrite(pic_path, save_img)
