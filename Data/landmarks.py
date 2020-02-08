import numpy as np
import cv2
import os
import codecs
import re
import struct
import shutil


def main():
    path = 'D:/ProgramData/SCUT-FBP5500_v2/facial landmark'
    img_path = 'D:/ProgramData/SCUT-FBP5500_v2/Pad_Square'

    '''landmarks = []
    for i in range(2000):
        # Load landmarks
        pts = os.path.join(path, 'AF'+str(i+1)+'.pts')
        data = open(pts, 'rb').read()
        points = struct.unpack('i172f', data)
        points = points[1:]
        num_points = int(len(points)/2)
        arr = np.zeros([num_points, 2])
        for p in range(num_points):
            arr[p, 0] = points[2*p]
            arr[p, 1] = points[2*p+1]
        landmarks.append(arr)
    
    landmarks = np.array(landmarks)
    print(landmarks.shape)
    np.save(os.path.join(img_path, 'landmarks.npy'), landmarks)'''


    landmarks = np.load(os.path.join(img_path, 'landmarks.npy'))
    num_images, num_points, _ = landmarks.shape
    for i in range(10):
        marks = landmarks[i, :, :].astype(int)
        # Load image
        imgfile = os.path.join(img_path, str(i+1)+'.jpg')
        img = cv2.imread(imgfile)
        rows, cols, _ = img.shape
        # Cut half face
        side = np.random.choice(['left', 'right'])
        half = cut_face(img, marks, side)
        # Face box
        x0, y0 = np.min(marks, axis=0)
        x1, y1 = np.max(marks, axis=0)
        cv2.rectangle(img, (x0, y0), (x1, y1), (0,0,255), 2)
        # Draw line
        div_line = np.zeros([9, 2])
        div_line[0, :] = [cols/2, 0]
        div_line[1, :] =  marks[0,:]
        div_line[2, :] = (marks[22,:] + marks[36,:]) / 2
        div_line[3, :] = (marks[60,:] + marks[72,:]) / 2
        div_line[4, :] = marks[66, :]
        div_line[5, :] = marks[76, :]
        div_line[6, :] = (marks[82,:] + marks[83,:]) / 2
        div_line[7, :] = marks[11, :]
        div_line[8, :] = [cols/2, rows-1]
        div_line = div_line.astype(np.int32)
        for l in range(div_line.shape[0]-1):
            cv2.line(img, tuple(div_line[l,:]), tuple(div_line[l+1,:]), (255,0,0), 2)
        # Draw landmarks
        for p in range(num_points):
            cv2.circle(img, tuple(marks[p,:]), 2, (0,255,0), 2)

        cv2.imshow(str(i+1), img)
        cv2.imshow(str(i+1) + '-half', half)
        cv2.waitKey(0)



def cut_face(img, landmark, side):
    # Make a copy to be cut
    face = img.copy()
    # Size
    rows, cols, _ = face.shape

    # Difine cut line according to landmark
    div_line = np.zeros([9, 2])
    div_line[0, :] = [cols/2, 0]
    div_line[1, :] = landmark[0,:]
    div_line[2, :] = (landmark[22,:] + landmark[36,:]) / 2
    div_line[3, :] = (landmark[60,:] + landmark[72,:]) / 2
    div_line[4, :] = landmark[66, :]
    div_line[5, :] = landmark[76, :]
    div_line[6, :] = (landmark[82,:] + landmark[83,:]) / 2
    div_line[7, :] = landmark[11, :]
    div_line[8, :] = [cols/2, rows-1]
    div_line = div_line.astype(np.int32)

    # Define the line in each row
    cut_line = []
    for i in range(div_line.shape[0]-1):
        x0 = div_line[i, 0]
        y0 = div_line[i, 1]
        x1 = div_line[i+1, 0]
        y1 = div_line[i+1, 1]
        if x0 == x1:
            for y in range(y0, y1):
                cut_line.append(x0)
        else:
            a = (y1 - y0) / (x1 - x0)
            b = y0 - a*x0
            for y in range(y0, y1):
                cut_line.append( (y-b)/a )
    cut_line.append(x1)
    cut_line = np.array(cut_line, dtype=np.int32)

    # Cut half of the face
    for r in range(rows):
        if side == 'right':
            face[r, 0:cut_line[r], :] = 255
        if side == 'left':
            face[r, cut_line[r]:cols, :] = 255
    
    return face


if __name__ == '__main__':
    main()
