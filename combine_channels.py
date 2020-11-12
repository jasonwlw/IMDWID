import os
import numpy as np
import cv2
from scipy.io import loadmat

direc = 'rgbd-scenes'
mat_files = []
rgb_files = {}
depth_files = {}
train_or_test = {}
for root, dirs, files in os.walk(direc):
    for fold in dirs:
        path = os.path.join(root, fold)
        for root2, dirs2, files2 in os.walk(path):
            for fil2 in files2:
                if fil2.endswith('.mat'):
                    mat_files.append(os.path.join(root2, fil2))

            ### COntinue here to get numpy image arrays for RGBD
            for fold2 in dirs2:
                path2 = os.path.join(root2, fold2)
                for fil3 in os.listdir(path2):
                    if 'depth' in fil3:
                        rgb = fil3.replace('_depth','')
                        rgb = cv2.imread(os.path.join(path2, rgb))
                        depth = cv2.imread(os.path.join(path2, fil3), -1)
                        
                        arr = np.zeros((rgb.shape[0], rgb.shape[1], rgb.shape[2]+1))
                        arr[:,:,:3] = rgb
                        arr[:,:,3] = depth
                        num = np.random.randint(1,101)
                        #print(fil3.replace('_depth','_combined').replace('.png', ''))
                        if num > 85:
                            root_path = '/home/witryjw/data/val/'
                            train_or_test[fil3.replace('_depth', '_combined').replace('.png','')] = 'val'
                        elif num > 70:
                            root_path = '/home/witryjw/data/test/'
                            train_or_test[fil3.replace('_depth', '_combined').replace('.png','')] = 'test'
                        else:
                            root_path = '/home/witryjw/data/train/'
                            train_or_test[fil3.replace('_depth', '_combined').replace('.png','')] = 'train'
                        save_path = os.path.join(root_path, fil3.replace('_depth', '_combined').replace('.png',''))
                        np.save(save_path, arr)
                        """
                        fil_save = os.path.split(mat_files[0])[1].split('.')[0]
                        annot = {}
                        loadmat(mat_files[0], mdict = annot)
                        for j,frame in enumerate(annot['bboxes'][0]):
                            print(fil_save+'_'+str(j)+'_combined')
                            assert 1 == 0
                        """
                        
                        

annot = {}
width = 640
height = 480
clses = []
for i,fil in enumerate(mat_files):
    fil_save = os.path.split(fil)[1].split('.')[0]
    loadmat(fil, mdict = annot)
    for j,frame in enumerate(annot['bboxes'][0]):
        split = train_or_test[fil_save+'_'+str(j+1)+'_combined']
        if split == 'val':
            impath = os.path.join('./data/val/', fil_save+'_'+str(j+1)+'_combined.npy')
        elif split == 'train':
            impath = os.path.join('./data/train/', fil_save+'_'+str(j+1)+'_combined.npy')
        elif split == 'test':
            impath = os.path.join('./data/test/', fil_save+'_'+str(j+1)+'_combined.npy')
        else:
            print("UHHHH")


        mask = np.zeros((width, height, len(frame[0])))
        im_classes = []
        for k,annotation in enumerate(frame[0]):
            #print(impath)
            cls = annotation[0][0]
            cls_index = annotation[1][0][0]
            if cls in clses:
                pass
            else:
                clses.append(cls)
            # top bottom left right

            top = annotation[2][0][0]
            bottom = annotation[3][0][0]
            left = annotation[4][0][0]
            right = annotation[5][0][0]
            im_classes.append(cls)
            mask[left:right, bottom:top, k] = 1
        save_path = impath.replace('combined', 'masks')
        np.save(save_path, mask)
        save_path = save_path.replace('masks', 'classes').replace('.npy','.txt')
        with open(save_path, 'w+') as f0:
            for cls in im_classes:
                f0.write(cls+'\n')


"""
with open('./classes.txt', 'w+') as f0:
    for cls in clses:
        f0.write(cls+'\n')
   """             



