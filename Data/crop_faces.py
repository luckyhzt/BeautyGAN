import cv2
import os
import numpy as np


def main():
    this_dir = os.path.dirname(__file__)
    img_path = 'D:\ProgramData\SCUT-FBP5500_v2\Images'
    save_path = 'D:\ProgramData\SCUT-FBP5500_v2\Pad_Square'

    num_images = 2000

    max_size = 1000

    for i in range(1, num_images+1):
        img_name = 'AF' + str(i) + '.jpg'
        img = cv2.imread(os.path.join(img_path, img_name))
        cv2.imwrite(os.path.join(save_path, str(i) + '.jpg'), img)
        print(i, '/', num_images)

    '''for i in range(1, num_images + 1):
        img_name = 'SCUT-FBP-' + str(i) + '.jpg'
        img = cv2.imread(os.path.join(img_path, img_name))
        img = pad_square(img)
        if img.shape[0] >= max_size:
            img = cv2.resize(img, (max_size, max_size), interpolation=cv2.INTER_AREA)

        cv2.imwrite(os.path.join(save_path, str(i) + '.jpg'), img)
        print(i, '/', num_images)'''
        


def remove_frame(img):
    rows, cols, channels = img.shape

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    start_flag = False
    end_flag = False
    start = 0
    end = cols

    sensitivity = 5
    low = np.array([0,0,255-sensitivity])
    up = np.array([255,sensitivity,255])

    for c in range(cols):
        white = True
        for r in range(rows):
            value = hsv[r, c, :]
            if value[1] in range(low[1], up[1]):
                a = 1
            else:
                print(value)
                white = False

        if white:
            print(1)
            start_flag = True
            if end_flag:
                end = c
                break
        else:
            print(0)
            if not start_flag:
                break
            else:
                if not end_flag:
                    start = c
                end_flag = True
    
    return img[:, start:end, :]
        


def pad_square(img, value=255):
    h, w, c = img.shape
    s = max(h, w)
    pad_value = [value] * c

    pad_w = int((s - w) / 2)
    pad_n = int((s - h) / 2)
    pad_e = s - (w + pad_w)
    pad_s = s - (h + pad_n)

    pad_img = cv2.copyMakeBorder(img, pad_n, pad_s, pad_w, pad_e, borderType=cv2.BORDER_CONSTANT, value=pad_value)

    return pad_img



if __name__ == '__main__':
    main()