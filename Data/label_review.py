import numpy as np
import os
import pickle



def main():
    '''thisDir = os.path.dirname(__file__)
    label_path = 'D:/ThesisData/SCUT-FBP/Label/label.npy'''

    path = 'D:/ThesisData/SCUT-FBP5500_v2/annotation.pkl'
    annotation = pickle.load( open(path, 'rb') )
    
    label = []
    for i in range(1800):
        ratings = annotation[i]['all_ratings']
        score = np.sum(ratings) / np.count_nonzero(ratings)
        label.append(score)
    
    label = np.array(label).reshape(-1, 1)

    label = label_level(label, 4)
    dist = np.sum(label, axis=0)
    print(dist)





def label_distribution(label):
    label = label.astype(int)
    images = label.shape[0]
    dist = np.zeros((images, 5))

    for i in range(0, images):
        occur = np.bincount(label[i, :].reshape((-1)), minlength=6)
        occur = occur[1:6]
        dist[i, :] = occur / np.sum(occur)

    return dist



def average_score(label):
    avg = np.sum(label, axis=1, keepdims=True)
    num = np.count_nonzero(label, axis=1).reshape([-1, 1])
    avg = avg / num

    return avg


def label_level(label, num_level):
    level_gap = 4.0 / num_level
    level = np.floor( (label - 1) / level_gap)
    level = one_hot(level, num_class=int(num_level))

    return level


def one_hot(label, num_class):
    label = label.astype(int)
    n = label.shape[0]
    onehot = np.zeros((n, num_class), np.float32)
    
    for i in range(n):
        onehot[i, label[i]] = 1.0

    return onehot




if __name__ == '__main__':
    main()
    
