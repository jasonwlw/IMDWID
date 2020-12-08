import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from skimage.transform import resize
from csv_dataset import CsvDataset
import sys
ROOT_DIR = os.path.abspath('../Mask_RCNN-master/')

sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, "samples/rgbd/"))

from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
#from mrcnn import utils
from mrcnn.model import log
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

from rgbd import RGBDConfig
from rgbd import RGBDDataset

GpuIndex = 0
os.environ['CUDA_VISIBLE_DEVICES'] = str(GpuIndex)

class DRISE():
    def __init__(self, input_shape, num_masks, init_mask_res, mask_prob, model_weights, class_list_path, dataset, config):
        self.input_shape = input_shape
        self.num_masks = num_masks
        self.s = init_mask_res
        self.p1 = mask_prob
        self.config = config
        self.get_model(model_weights)
        self.get_class_list(class_list_path)
        self.create_encoder()
        self.predictions_assigned = False
        self.dataset = dataset

    def get_dataset(self, gt_csv_path):
        return CsvDataset(gt_csv_path)

    def map_4d_to_RGB(self, impath):
        path_root, path_branch = os.path.split(impath)
        path_branch = path_branch.replace('_combined','')

        path_root = os.path.join(path_root, os.pardir, os.pardir)
        path_root = os.path.join(path_root, 'rgbd-scenes')
        print(path_branch)
        path_branch = path_branch.split('_')
        path_leaf = path_branch[-1].split('.')[0]
        path_branch2 = path_branch[-2]
        path_branch1 = '_'.join(path_branch[:-2])
        print(path_branch1)

        path_branch2 = '_'.join([path_branch1, path_branch2])
        path_leaf = '_'.join([path_branch2, path_leaf])

        return os.path.join(path_root, path_branch1, path_branch2, path_leaf+'.png')

        


    def get_model(self, weights):
        self.model = modellib.MaskRCNN(mode='inference', config = self.config, model_dir = ROOT_DIR)
        self.model.load_weights(weights, by_name=True)


    def generate_mask(self):
        cell_size = np.ceil(np.array(self.input_shape) / self.s)
        up_size = (self.s + 1) * cell_size

        grid = np.random.rand(self.s,self.s) < self.p1
        grid = grid.astype('float32')

        x = np.random.randint(0, cell_size[0])
        y = np.random.randint(0, cell_size[1])
        #Could generator be faster here?
        mask = resize(grid, up_size, order = 1, mode = 'reflect', anti_aliasing = False)[x:x+self.input_shape[0], y:y+self.input_shape[1]]
        mask = np.expand_dims(mask, 2)
        return mask


    def get_class_list(self, class_list_path = None):
        self.class_list = []
        if class_list_path is not None:
            with open(class_list_path, 'r') as f0:
                for line in f0.readlines():
                    self.class_list.append(line.strip())
        else:
            self.class_list = sorted(os.listdir(os.path.join(ROOT_DIR,'../rgbd-dataset/')))
        return 0

    def set_class_list(self, class_list):
        #Override basic class list function
        self.class_list = class_list
        return 0

    def create_encoder(self):
        self.ohe = OneHotEncoder(categories=[self.class_list], sparse = False)
        return 0

    def get_str_classes(self, class_ids):
        class_ids = class_ids.astype(int)
        print(self.class_list)
        return np.asarray(self.class_list)[class_ids-1]

    def transform_gt_classes(self, classes):
        print(classes)
        print(self.class_list)
        return self.ohe.fit_transform(list(classes.reshape(-1,1)))

    def read_image(self, impath, image_id):
        #im = np.load(impath)

        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(self.dataset, inference_config,
                               image_id, use_mini_mask=False)
        return image

    def get_predictions(self, image):
        #im = self.read_image(impath)
        mask = self.generate_mask()
        results = self.model.detect([image*mask]) 
        return results[0], mask

    def get_predictions_no_mask(self, image):
        #im = self.read_image(impath, image_id)

        #print(impath)
        results = self.model.detect([image])
        return results[0]

    def assign_predictions(self, gt_rois, gt_classes, results):
        # assign predictions to their respective targets
        iou_thres = 0.2
        iou = self.np_iou(gt_rois, results['rois'])
        print("GT", gt_rois)
        print("Pred", results['rois'])
        print("IOU", iou)
        iou = np.where(iou > iou_thres, 1, 0)
        confs = iou * results['scores']
        print("confs", confs)
        mask = np.any(confs, axis = 1)
        print("Any confs", mask)
        confs = np.argmax(confs, axis = 1)
        print("max confs", confs)
        confs = np.where(mask == True, confs, None)
        print("where confs", confs)
        print(mask)
        print(confs[mask])
        print("confs", confs)
        self.inds = confs[mask].astype(int)
        #for getting correct predictions
        self.maxes = dict(zip(np.arange(len(gt_classes)), confs))
        count = 0
        for i, conf in enumerate(confs):
            if conf is not None:
                confs[i] = count + len(gt_classes)
                count += 1
            else:
                pass
        # For getting correct saliency maps
        self.assignments = dict(zip(np.arange(len(gt_classes)), confs))
        print(confs)
        #assert is_unique(maxes) any duplicates?
        # Check that rois are in x1, y1, x2, y2 order 
        self.predicted_boxes = results['rois']
        #self.predicted_classes = self.get_str_classes(results['class_ids'])
        self.predicted_classes = results['class_ids']
        if self.inds.size == 0:
            #There are no predictions that meet the score+iou threshold
            self.predictions_assigned = False
        else:
            self.predictions_assigned = True


    def compute_saliency(self,image, gt_rois, gt_classes):
        if self.predictions_assigned:
            print(self.inds)
            print(type(self.inds),type(self.inds[0]))
            print(self.inds[0])
            gt_rois = np.append(gt_rois, self.predicted_boxes[self.inds,:], axis = 0)
            print("GT_CLASSES1", gt_classes)
            gt_classes = np.append(gt_classes, self.predicted_classes[self.inds])
            print("GT_CLASSES2", gt_classes)


        sal = np.zeros((len(gt_classes), *self.input_shape))
        print("GT CLASSES", gt_classes) 
        gt_classes = self.get_str_classes(gt_classes)
        gt_classes = self.transform_gt_classes(gt_classes)
        for i in tqdm(range(self.num_masks), desc="Detecting Masked Images"):
            # mask shape = (1, h, w)
            results, mask = self.get_predictions(image)
            if results['rois'].shape[0] == 0:
                #we have no detections; mask would be zeroed anyway, just don't add anything
                continue

            pred_rois = self.process_rois(results['rois'])
            pred_logits = self.process_logits(results['logits'])
            class_ids = results['class_ids']

            sL = self.np_iou(gt_rois, pred_rois)
            sP = self.cosine_similarity(gt_classes, pred_logits)
            sensitivity = sL * sP
            sensitivity = np.max(sensitivity, axis = 1)
           
            sal += (mask*sensitivity).transpose(2,0,1)

        return sal



    def process_logits(self, logits):
        # Add any processing to logits here
        logits = logits[:,1:]
        return logits

    def process_rois(self, rois):
        #rois = rois*np.array([480./640., 1., 480./640., 1.], dtype = np.float32)
        # Add any processing to rois here
        return rois

    def cosine_similarity(self, DP1, DP2):
        norms = 1./np.linalg.norm(DP2, axis = 1)
        #Insert norms for predicted boxes as well
        dots = np.tensordot(DP1, DP2.T, axes = 1)
        return dots * norms
    
    def np_iou(self, tb1, tb2):
        y11, x11, y12, x12 = np.split(tb1, 4, axis = 1)
        # MRCNN still return in this form?
        y21, x21, y22, x22 = np.split(tb2, 4, axis = 1)

        xA = np.maximum(x11, np.transpose(x21))
        yA = np.maximum(y11, np.transpose(y21))
        xB = np.minimum(x12, np.transpose(x22))
        yB = np.minimum(y12, np.transpose(y22))

        interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)

        boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
        boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)

        iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)

        return iou

    def get_detection_vectors(self, dataset, ID):
        gt_rois, gt_classes = dataset.get_rois_and_classes(ID)
        return gt_rois, gt_classes


    def explain_one(self, gt_csv_path=None, impath = None, imID = None, prediction_saliencies = True):
        if gt_csv_path is not None:
            dataset = self.get_dataset(gt_csv_path)
        if impath is not None:
            image_id = dataset.get_image_id(impath)
        elif imID is not None:
            impath = dataset.get_image_path(imID)
            print(impath)
            image_id = imID
        else:
            print("Please select either one image or an image ID")

        
         
        image, image_meta, gt_classes, gt_rois, gt_mask =\
        modellib.load_image_gt(self.dataset, self.config,
                               image_id, use_mini_mask=False)

        #gt_rois, gt_classes = dataset.get_rois_and_classes(ID)

        results_noMask = self.get_predictions_no_mask(image)
        #results_noMask['rois'] = results_noMask['rois']*np.array([480./640., 1., 480./640., 1.], dtype = np.float32)
        self.predictions_assigned = False
        self.predicted_boxes = []
        if prediction_saliencies and bool(len(results_noMask['rois'])) and bool(len(results_noMask['class_ids'])):
            assignments = self.assign_predictions(gt_rois, gt_classes, results_noMask)
        else:
            assignments = None
        #AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask, results_noMask["rois"], results_noMask["class_ids"], results_noMask["scores"], results_noMask['masks'])
        #gt_rois are y1, x2, y2, x2
        sal = self.compute_saliency(image, gt_rois, gt_classes)
        gt_classes = self.get_str_classes(gt_classes)
        if self.predictions_assigned:
            self.predicted_classes = self.get_str_classes(self.predicted_classes)
        for i, im in enumerate(sal[:len(gt_classes),:,:]):
            full_aug_mask = np.zeros((sal.shape[1], sal.shape[2]))
            found = True
            this_gt_box = gt_rois[i % len(gt_classes)]
            if self.predictions_assigned:
                key = self.maxes[i]
                if key is not None:
                    this_pred_box = self.predicted_boxes[key]
                else:
                    found = False
                    #missed detection
            #rgb_impath = self.map_4d_to_RGB(impath)
            #image = cv2.imread(self.dataset.get_image_path(image_id))
            #image = cv2.rectangle(image, (this_gt_box[1], this_gt_box[0]), (this_gt_box[3], this_gt_box[2]), (0,255,0), 2)
            print(this_gt_box)
            if found and self.predictions_assigned:
                #image = cv2.rectangle(image, (this_pred_box[1], this_pred_box[0]), (this_pred_box[3], this_pred_box[2]), (255,0,0), 2)
                image_pred = np.copy(image)

            
            plt.imshow(image)
            plt.imshow(im, cmap='jet', alpha=0.5)
            plt.colorbar()
            plt.savefig('./saliency_map_'+str(i)+'_'+gt_classes[i]+str(image_id)+'.png')
            plt.clf()
            if found and self.predictions_assigned:
                plt.imshow(image_pred)
                plt.imshow(sal[self.assignments[i]], cmap = 'jet', alpha = 0.5)
                plt.colorbar()
                plt.savefig('./saliency_map_'+str(i)+'_'+self.predicted_classes[i]+'_predictions'+str(image_id)+'.png')
                plt.clf()

                #plt.imshow(image_pred)
                #plt.imshow(sal[self.maxes[i]], cmap = 'jet', alpha = 0.5)
                #plt.colorbar()
                #plt.savefig('./saliency_map_'+str(i)+'_'+self.predicted_classes[i]+'_predictions')
                #plt.clf()

                plt.imshow(image_pred)
                plt.imshow(im - sal[self.assignments[i]], cmap = 'jet', alpha = 0.5)
                plt.colorbar()
                plt.savefig('./saliency_map_'+str(i)+'_'+self.predicted_classes[i]+'_predictions_im-targ')
                plt.clf()
                
                aug_mask = im - sal[self.assignments[i]]
                #new_range = [0.0,1.3]
                #aug_mask = im - sal[self.assignments[i]]
                #aug_mask = (aug_mask - np.min(aug_mask))/(np.max(aug_mask) - np.min(aug_mask))
                #aug_mask = aug_mask * (new_range[1] - new_range[0]) + new_range[0]
                full_aug_mask += aug_mask
                #aug_mask = np.expand_dims(aug_mask,2)

            else:
                full_aug_mask += im



                #np.save('./saliency_map_'+str(i)+'_gt_'+gt_classes[i]+'_pred_'+self.predicted_classes[i]+'im-targ', im_targ)
                #np.save('./saliency_map_'+str(i)+'_gt_'+gt_classes[i]+'_pred_'+self.predicted_classes[i]+'targ-im', targ_im)


        np.save(os.path.join('masks', str(image_id)), full_aug_mask)


    def explain_many(self, csv_path, prediction_saliencies = True, impaths = None, IDs = None):
        if impaths is None:
            for ID in IDs:
                self.explain_one(gt_csv_path=csv_path, imID = ID, prediction_saliencies = prediction_saliencies)
        elif IDs is None:
            for impath in impaths:
                self.explain_one(gt_csv_path=csv_path, impath = impath, prediction_saliencies = prediction_saliencies)

        else:
            print("SELECT EITHER A LIST OF IM IDS OR IMPATHS")

        return 0 



#self, input_size, num_masks, init_mask_res, mask_prob, model_weights, class_list_path
#weights = '../Mask_RCNN-master/logs/rgbd_10e_NoAug/mask_rcnn_rgbd_0009.h5'
weights = os.path.join('../Mask_RCNN-master/', 'logs', 'DRISE_40e_reduced_200TS', 'mask_rcnn_rgbd_0004.h5')
classes = '../classes.txt'
classes = None
csv_path = '../rgbd-dataset.csv'
class InferenceConfig(RGBDConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DEPTH_NULL = False
    #RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
config = InferenceConfig()

print(ROOT_DIR)

dataset_train = RGBDDataset()
dataset_train.load_images(os.path.join(ROOT_DIR, 'train.txt'), config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_train.prepare()

num_masks = 1000

drise = DRISE([640,640], num_masks, 8, 0.5, weights,classes, dataset_train, config)
#print(dataset_train.get_image_path(0))
#drise.explain_one(csv_path, imID = 35, prediction_saliencies = True)
all_ids = dataset_train.image_ids

all_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 34, 35, 36, 37, 38, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 58, 59, 61, 62, 63, 64, 65, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 85, 86, 87, 89, 90, 91, 92, 93, 94, 95, 96, 98, 99, 100, 101, 102, 104, 105, 106, 107, 108, 109, 111, 112, 114, 115, 116, 117, 119, 120, 122, 123, 125, 126, 130, 131, 133, 134, 137, 138, 140, 141, 143, 145, 146, 147, 148, 149, 150, 152, 154, 155, 156, 157, 158, 161, 163, 164, 165, 166, 167, 168, 171, 173, 176, 178, 179, 180, 181, 182, 185, 186, 187, 188, 191, 192, 194, 196, 197, 201, 202, 203, 204, 205, 208, 209, 210, 211, 212, 213, 215, 216, 217, 218, 219, 220, 221, 223, 224, 225, 226, 227, 228, 229, 230, 234, 235, 236, 237, 238]

"""
clses = []
insts = []
insts_dict = {}
ids_dict = {}
for ids in all_ids[:240]:
    impath = dataset_train.get_image_path(ids)
    cls = impath.split('/')[5]
    inst = impath.split('/')[6]
    clses.append(cls)
    insts.append(inst)
    print(ids, impath)
    ids_dict[impath] = ids
    if inst not in insts_dict:
        insts_dict[inst] = []
    insts_dict[inst].append(impath)
print(np.unique(clses, return_counts=True))
clses_here  = np.unique(clses)
insts_here = np.unique(insts)
to_use = []
for cls in clses_here:
    done = []
    for inst in insts_here:
        if cls in inst:
            num = inst.split('_')[-1]
            if num in done:
                continue
            else:
                to_use.append(insts_dict[inst][0])
                done.append(num)

print(to_use)
print(np.unique(to_use, return_counts=True))
"""

all_ids = np.random.choice(all_ids, size = int(0.25*len(all_ids)), replace = False)

drise.explain_many(csv_path, IDs = all_ids, prediction_saliencies = True)
"""
with open('./new_train.txt', 'w+') as f0:

    for line in to_use:
        f0.write(line+'\n')
"""

#np.save('path_to_id.npy', ids_dict)
