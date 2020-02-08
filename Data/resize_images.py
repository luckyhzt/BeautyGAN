import cv2

for i in range(2000):
    path = 'D:/ProgramData/SCUT-FBP5500_v2/Images/AF' + str(i+1) + '.jpg'
    img = cv2.imread(path)
    resized_img = cv2.resize(img, (256, 256))
    save_path = 'D:/Program/BeautyEval/Data/Unlabeled_data/pad_256/' + str(i+1) + '.jpg'
    cv2.imwrite(save_path, resized_img)
    print(i+1)