import numpy as np
import os



def main():
    thisDir = os.path.dirname(__file__)
    label_path = 'D:/ThesisData/SCUT-FBP/Label/label.npy'
    label = np.load(label_path)
    label = average_score(label)
    label = label_level(label, 4)
    dist = np.sum(label, axis=0)
    print(dist/np.sum(dist))





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
    
