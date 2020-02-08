import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import pickle



def extract():
    root_dir = 'D:/ProgramData/CelebA'

    num = 202599

    identity_txt = open(os.path.join(root_dir, 'identity_CelebA.txt'), 'r')
    bbox_txt = open(os.path.join(root_dir, 'list_bbox_celeba.txt'), 'r')
    attr_txt = open(os.path.join(root_dir, 'list_attr_celeba.txt'), 'r')
    landmarks_txt = open(os.path.join(root_dir, 'list_landmarks_celeba.txt'), 'r')
    # Skip first two lines
    bbox_txt.readline(), bbox_txt.readline()
    landmarks_txt.readline(), landmarks_txt.readline()
    attr_txt.readline(), 
    attr_names = attr_txt.readline().split()

    annotation = []

    for i in range(num):
        # Identity
        img_file, identity = identity_txt.readline().split()
        # Face bounding box
        bbox = bbox_txt.readline().split()[1:]
        bbox = np.array([int(b) for b in bbox])
        # Landmarks
        landmarks = landmarks_txt.readline().split()[1:]
        landmarks = np.array([int(l) for l in landmarks])
        # Attributes
        attr = dict()
        attr_value = attr_txt.readline().split()[1:]
        for a, n in enumerate(attr_names):
            attr[n] = bool(int(attr_value[a]) + 1)
        # Create Annotation
        this_anno = {
                'path': os.path.join('Images', 'img_celeba', img_file),
                'identity': identity,
                'bbox': bbox,
                'landmarks': landmarks,
                'attributes': attr,
            }
        annotation.append(this_anno)
    
    print(len(annotation))
    print(annotation[200]['path'])

    pk_file = os.path.join(root_dir, 'Annotation.pkl')
    with open(pk_file, 'wb') as outFile:
        pickle.dump(annotation, outFile)



def test():
    root_dir = 'D:/ProgramData/CelebA'
    
    pkl_file = os.path.join(root_dir, 'Annotation.pkl')
    with open(pkl_file, 'rb') as f:
        annotation = pickle.load(f)

    data = annotation[51022]

    img = cv2.imread(os.path.join(root_dir, data['path']))
    landmarks = data['landmarks'].astype(np.int32)
    attr = data['attributes']
    identity = data['identity']
    bbox = data['bbox']

    print(identity)
    print(attr)

    for i in range(int(landmarks.shape[0]/2)):
        cv2.circle(img, (landmarks[2*i], landmarks[2*i + 1]), 2, (255,0,0), 2) 
    
    start = (bbox[0], bbox[1])
    end = (bbox[0]+bbox[2], bbox[1]+bbox[3])
    cv2.rectangle(img, start, end, (0,0,255), 2) 
    
    cv2.imshow('1', img)
    cv2.waitKey(0)



if __name__ == '__main__':
    test()