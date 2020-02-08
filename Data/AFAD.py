import numpy as np
import os
import pickle
import cv2


def main():
    root = 'D:/ThesisData/AFAD_Lite'
    txt_file = os.path.join(root, 'AFAD-Lite.txt')
    with open(txt_file, 'r') as f:
        content = [line.rstrip()[2:] for line in f]

    annotation = []
    for info in content:
        age, gender, _ = info.split('/')
        if gender == '111':
            gender = 'M'
        elif gender =='112':
            gender = 'F'
        path = 'Images/' + info
        this_anno = {
            'path': path,
            'gender': gender,
            'age': int(age),
        }
        annotation.append(this_anno)
    
    pk_file = os.path.join(root, 'Annotation.pkl')
    with open(pk_file, 'wb') as f:
        pickle.dump(annotation, f)



def test():
    root = 'D:/ThesisData/AFAD_Lite'
    pk_file = os.path.join(root, 'Annotation.pkl')
    with open(pk_file, 'rb') as f:
        annotation = pickle.load(f)
    
    anno = annotation[9]
    img = cv2.imread( os.path.join(root, anno['path']) )
    print(anno['gender'])
    print(anno['age'])

    cv2.imshow('1', img)
    cv2.waitKey(0)



def choose():
    root = 'D:/ThesisData/AFAD_Lite'
    pk_file = os.path.join(root, 'Annotation.pkl')
    with open(pk_file, 'rb') as f:
        annotation = pickle.load(f)

    index = []
    for i, anno in enumerate(annotation):
        if anno['gender'] == 'F':
            index.append(i)
    
    index = np.array(index)

    np.save(os.path.join(root, 'female_index.npy'), index)



if __name__ == '__main__':
    choose()


