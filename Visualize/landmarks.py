import numpy as np
import cv2
import os
import pickle


def main():
    index = 12
    scale = 1.1
    
    dataset_path = 'D:/ThesisData/SCUT-FBP5500_v2'
    pkl_file = os.path.join(dataset_path, 'Annotation.pkl')
    with open(pkl_file, 'rb') as f:
        annotation = pickle.load(f)
    data = annotation[index]

    # Load image in opencv
    face = cv2.imread(os.path.join(dataset_path, data['path']))
    # Get landmark
    landmarks = data['landmarks']
    # Get face box
    h, w, _ = face.shape
    x0, y0 = np.min(landmarks, axis=0)
    x1, y1 = np.max(landmarks, axis=0)
    # Copies of face
    face1 = face.copy()
    
    # Draw landmarks
    for i in range(landmarks.shape[0]):
        cv2.circle(face, tuple(np.int32(landmarks[i,:])), 2, (0,0,255), 2)
    # Draw bounding box
    #cv2.rectangle(face, (int(x0),int(y0)), (int(x1),int(y1)), (255,0,0), 2)

    # Draw final box
    # Center and Size of extended face box
    center = [ (x0 + x1) / 2, (y0 + y1) / 2 ]
    size = scale * max(x1 - x0, y1 - y0)
    # Coordinate of extended face box
    x0 = int(center[0] - size/2)
    x1 = int(x0 + size)
    y0 = int(center[1] - size/2)
    y1 = int(y0 + size)
    # Draw bounding box
    #cv2.rectangle(face, (int(x0),int(y0)), (int(x1),int(y1)), (0,255,0), 2)

    # Extract face
    face1 = face[max(y0,0):min(y1,h), max(x0,0):min(x1,w), :]
    face1 = cv2.resize(face1, (236,236), interpolation = cv2.INTER_AREA)
    face1 = face1[6:229, 6:229, :]
    #cv2.rectangle(face1, (0,0), (224,224), (0,255,0), 2)

    #cv2.imwrite('Visualize/Images/face.png', face)
    cv2.imwrite('Visualize/Images/face_landmark.png', face1)
    cv2.imshow('original', face)
    cv2.imshow('crop', face1)
    cv2.waitKey(0)



if __name__ == '__main__':
    main()