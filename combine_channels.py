import os
import numpy as np
import cv2
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

direc = '../rgbd-scenes'
print(direc)
mat_files = []
rgb_files = {}
depth_files = {}
create_rgbd = {}
num = 0
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
                        dep = os.path.join(path2, fil3)
                        rgb = dep.replace('_depth','')
                        #rgb = cv2.imread(os.path.join(path2, rgb))
                        #depth = cv2.imread(os.path.join(path2, fil3), -1)
                        
                        #arr = np.zeros((rgb.shape[0], rgb.shape[1], rgb.shape[2]+1))
                        #arr[:,:,:3] = rgb
                        #arr[:,:,3] = depth
                        #print(fil3.replace('_depth','_combined').replace('.png', ''))

                        #save_path = os.path.join(root_path, fil3.replace('_depth', '_combined').replace('.png',''))
                        save_path = os.path.join(os.path.abspath('./'), 'data', fil3.replace('_depth', '_combined').replace('.png',''))
                        
                        create_rgbd[save_path] = [rgb,dep]
                        num += 1
                        #np.save(save_path, arr)
                        """
                        fil_save = os.path.split(mat_files[0])[1].split('.')[0]
                        annot = {}
                        loadmat(mat_files[0], mdict = annot)
                        for j,frame in enumerate(annot['bboxes'][0]):
                            print(fil_save+'_'+str(j)+'_combined')
                            assert 1 == 0
                        """
ims = np.fromiter(create_rgbd.keys(), dtype = 'S128', count = num)
print(type(ims))
print(ims)
print(ims[0])
print(type(ims[0]))
train_ims, val_ims = train_test_split(ims, test_size = 0.2, random_state = 42)
for key in create_rgbd:
    print(key)
    save_path = key.split(os.sep)
    print(save_path)
    if key in train_ims:
        save_path = save_path.insert(save_path.index('data') + 1, 'train')
    else:
        save_path = save_path.insert(save_path.index('data') + 1, 'val')
    save_path = os.path.join(*save_path)
    rgb = cv2.imread(create_rgbd[key][0])
    dep = cv2.imread(create_rgbd[key][1], -1)
    arr = np.zeros((rgb.shape[0], rgb.shape[1], rgbd.shape[2]+1))
    arr[:,:,:3] = rgb
    arr[:,:,3] = dep
    np.save(save_path, arr)

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



