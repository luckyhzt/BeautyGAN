import os
import numpy as np
from PIL import Image

import pickle
import cv2
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils import data as Data

import models.ops as ops
import utils


class FBP_dataset(Data.Dataset):
    '''
    A customized dataset for SCUT-FBP-500
    '''
    def __init__(self, mode, image_path, image_index, label_path, image_size, crop_size):
        # Parameters
        self.mode = mode
        self.len = image_index.shape[0]
        self.img_size = image_size
        self.crop_size = crop_size
        self.img_index = image_index
        self.img_path = image_path
        self.scale = 1.1

        # Load labels
        if self.mode in ('train', 'test'):
            label = np.load(label_path)
            self.label_avg = utils.average_score(label)

        # Load landmarks
        pkl_file = os.path.join(image_path, 'landmarks.pkl')
        with open(pkl_file, 'rb') as readFile:
            landmarks = pickle.load(readFile)
            face_boxes = np.array(landmarks['boxes'])
            self.face_boxes = face_boxes[:, 0:4]

        # Define image transforms
        if self.mode == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.RandomCrop(self.crop_size),
                torchvision.transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                ops.Img_to_zero_center(),
            ])
        if self.mode == 'test':
            self.transform = transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.CenterCrop(self.crop_size),
                transforms.ToTensor(),
                ops.Img_to_zero_center(),
            ])


    def __getitem__(self, index):
        # Get image
        index = self.img_index[index]
        img_path = os.path.join(self.img_path, str(index+1) + '.jpg')
        cv_image = cv2.imread(img_path)
        cv_face = self.crop_face(cv_image, self.face_boxes[index], self.scale)
        image = Image.fromarray(cv2.cvtColor(cv_face, cv2.COLOR_BGR2RGB))
        # Apply Transform
        x = self.transform(image)

        y = self.label_avg[index, :]
        return x, np.float32(y)


    def crop_face(self, img, box, scale):
        h, w, _ = img.shape
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        center = ( (x0 + x1) / 2, (y0 + y1) / 2 )
        size = int( scale * max(x1 - x0, y1 - y0) )

        x0 = int(center[0] - size/2)
        x1 = x0 + size
        y0 = int(center[1] - size/2)
        y1 = y0 + size

        face = img[max(y0,0):min(y1,h), max(x0,0):min(x1,w), :]

        if x0 < 0: pad_l = 0 - x0
        else: pad_l = 0
        if y0 < 0: pad_u = 0 - y0
        else: pad_u = 0
        if x1 > w: pad_r = x1 - w
        else: pad_r = 0
        if y1 > h: pad_d = y1 - h
        else: pad_d = 0

        face = cv2.copyMakeBorder(face, pad_u, pad_d, pad_l, pad_r, borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255))

        return face


    def __len__(self):
        return self.len




class FBP_dataset_V2(Data.Dataset):
    '''
    A customized dataset for SCUT-FBP-5500-V2
    '''
    def __init__(self, mode, dataset_path, image_index, image_size, crop_size):
        # Parameters
        self.mode = mode
        self.dataset_path = dataset_path
        self.len = image_index.shape[0]
        self.img_index = image_index
        self.img_size = image_size
        self.crop_size = crop_size
        self.scale = 1.1

        # Load annotation
        pkl_file = os.path.join(self.dataset_path, 'Annotation.pkl')
        with open(pkl_file, 'rb') as f:
            self.annotation = pickle.load(f)

        # Define image transforms
        if self.mode == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.RandomCrop(self.crop_size),
                torchvision.transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                ops.Img_to_zero_center(),
            ])
        if self.mode == 'test':
            self.transform = transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.CenterCrop(self.crop_size),
                transforms.ToTensor(),
                ops.Img_to_zero_center(),
            ])


    def __getitem__(self, index):
        # Get image
        data = self.annotation[ self.img_index[index] ]
        face = self.get_face(data)
        # Apply Transform
        x = self.transform(face)
        ratings = data['all_ratings']
        y = np.sum(ratings, keepdims=True) / np.count_nonzero(ratings)

        return x, np.float32(y)


    def get_face(self, data):
        # Load image in opencv
        img_path = os.path.join(self.dataset_path, data['path'])
        face = cv2.imread(img_path)
        # Get landmark
        landmark = data['landmarks']
        
        # Get face box
        h, w, _ = face.shape
        x0, y0 = np.min(landmark, axis=0)
        x1, y1 = np.max(landmark, axis=0)
        # Center and Size of extended face box
        center = [ (x0 + x1) / 2, (y0 + y1) / 2 ]
        size = self.scale * max(x1 - x0, y1 - y0)
        # Coordinate of extended face box
        x0 = int(center[0] - size/2)
        x1 = int(x0 + size)
        y0 = int(center[1] - size/2)
        y1 = int(y0 + size)
        # Extract the face
        face = face[max(y0,0):min(y1,h), max(x0,0):min(x1,w), :]

        # Pad the face to square
        if x0 < 0: pad_l = 0 - x0
        else: pad_l = 0
        if y0 < 0: pad_u = 0 - y0
        else: pad_u = 0
        if x1 > w: pad_r = x1 - w
        else: pad_r = 0
        if y1 > h: pad_d = y1 - h
        else: pad_d = 0
        face = cv2.copyMakeBorder(face, pad_u, pad_d, pad_l, pad_r, borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255))

        # Convert to PIL RGB image
        face = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))

        return face

    
    def cut_face(self, img, landmark, side):
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
                face[r, 0:cut_line[r], :] = 0
            if side == 'left':
                face[r, cut_line[r]:cols, :] = 0
        
        return face


    def __len__(self):
        return self.len




class AFAD(Data.Dataset):
    '''
    A customized dataset for AFAD
    '''
    def __init__(self, mode, dataset_path, image_index, image_size, crop_size):
        # Parameters
        self.mode = mode
        self.dataset_path = dataset_path
        self.len = image_index.shape[0]
        self.img_index = image_index
        self.img_size = image_size
        self.crop_size = crop_size
        self.scale = 1.1

        # Load annotation
        pkl_file = os.path.join(self.dataset_path, 'Annotation.pkl')
        with open(pkl_file, 'rb') as f:
            self.annotation = pickle.load(f)

        # Define image transforms
        if self.mode == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.RandomCrop(self.crop_size),
                torchvision.transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                ops.Img_to_zero_center(),
            ])
        if self.mode == 'test':
            self.transform = transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.CenterCrop(self.crop_size),
                transforms.ToTensor(),
                ops.Img_to_zero_center(),
            ])
    

    def __getitem__(self, index):
        data = self.annotation[ self.img_index[index] ]
        face = self.get_face(data)
        # Apply transform
        x = self.transform(face)

        return x
    

    def get_face(self, data):
        # Load image in opencv
        img_path = os.path.join(self.dataset_path, data['path'])
        face = cv2.imread(img_path)
        
        # Get face box
        h, w, _ = face.shape
        pad_h = int((h * self.scale - h) / 2)
        pad_w = int((w * self.scale - w) / 2)

        face = cv2.copyMakeBorder(face, pad_h, pad_h, pad_w, pad_w, borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255))

        # Convert to PIL RGB image
        face = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))

        return face


    def __len__(self):
        return self.len




class Label_Sampler():
    def __init__(self, sample_range):
        self.sample_range = sample_range
        self.prob = [91/1800, 856/1800, 650/1800, 203/1800]
    

    def sample(self, batch_size):
        #minScore = self.sample_range[0]
        #maxScore = self.sample_range[1]

        #sampled = np.random.uniform(low=minScore, high=maxScore, size=[self.batch_size,1])

        #sampled = 2.5 * np.random.randn(self.batch_size, 1) + 2.5

        sampled = np.random.rand(batch_size, 1) + np.random.choice(len(self.prob), (batch_size, 1), p=self.prob) + 1

        return torch.FloatTensor(sampled)
    




'''# test
index = np.arange(400)
trainset = FBP_dataset_V2('train', 'D:/ThesisData/SCUT-FBP5500_v2', np.arange(1000), 236, 224)
train_loader = Data.DataLoader(trainset, batch_size=8, shuffle=True)
train_iter = iter(train_loader)
x, y = train_iter.next()
print(x.size())
print(y)'''

'''sampler = Label_Sampler(10000, None)
s = sampler.sample()

one = np.where(np.logical_and(s>=1, s<2))[0].shape[0]
two = np.where(np.logical_and(s>=2, s<3))[0].shape[0]
three = np.where(np.logical_and(s>=3, s<4))[0].shape[0]
four = np.where(np.logical_and(s>=4, s<5))[0].shape[0]
print(one, two, three, four)'''