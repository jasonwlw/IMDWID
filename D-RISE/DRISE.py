import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from skimage.transform import resize
from csv_dataset import CsvDataset
import sys
ROOT_DIR = os.path.abspath('../Mask_RCNN-master/')

sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, "samples/shapes/"))

from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
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


    def get_class_list(self, class_list_path):
        self.class_list = []
        with open(class_list_path, 'r') as f0:
            for line in f0.readlines():
                self.class_list.append(line.strip())
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
        iou_thres = 0.4
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
        self.inds = confs[mask].astype(int)
        self.maxes = dict(zip(np.arange(len(gt_classes)), confs))
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
            gt_classes = np.append(gt_classes, self.predicted_classes[self.inds])


        sal = np.zeros((len(gt_classes), *self.input_shape))
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
        # Add any processing to rois here
        return rois

    def cosine_similarity(self, DP1, DP2):
        norms = 1./np.linalg.norm(DP2, axis = 1)
        dots = np.tensordot(DP1, DP2.T, axes = 1)
        return dots * norms
    
    def np_iou(self, tb1, tb2):
        x11, y11, x12, y12 = np.split(tb1, 4, axis = 1)
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


    def explain_one(self, gt_csv_path, impath = None, imID = None, prediction_saliencies = True):

        dataset = self.get_dataset(gt_csv_path)
        if impath is not None:
            image_id = dataset.get_image_id(impath)
        elif imID is not None:
            impath = dataset.get_image_path(imID)
            image_id = imID
        else:
            print("Please select either one image or an image ID")

        
         
        image, image_meta, gt_classes, gt_rois, gt_mask =\
        modellib.load_image_gt(self.dataset, self.config,
                               image_id, use_mini_mask=False)

        #gt_rois, gt_classes = dataset.get_rois_and_classes(ID)

        results_noMask = self.get_predictions_no_mask(image)

        if prediction_saliencies:
            assignments = self.assign_predictions(gt_rois, gt_classes, results_noMask)
        else:
            assignments = None

        sal = self.compute_saliency(image, gt_rois, gt_classes)
        gt_classes = self.get_str_classes(gt_classes)
        self.predicted_classes = self.get_str_classes(self.predicted_classes)
        for i, im in enumerate(sal[:len(gt_classes),:,:]):
            found = True
            this_gt_box = gt_rois[i % len(gt_classes)]
            if self.predictions_assigned:
                key = self.maxes[i]
                if key is not None:
                    this_pred_box = self.predicted_boxes[key]
                else:
                    found = False
                    #missed detection
            rgb_impath = self.map_4d_to_RGB(impath)
            image = cv2.imread(rgb_impath)
            image = cv2.rectangle(image, (this_gt_box[0], this_gt_box[1]), (this_gt_box[2], this_gt_box[3]), (0,255,0), 2)
            if found and self.predictions_assigned:
                image = cv2.rectangle(image, (this_pred_box[0], this_pred_box[1]), (this_pred_box[2], this_pred_box[3]), (255,0,0), 2)
                image_pred = np.copy(image)

            
            plt.imshow(image)
            plt.imshow(im, cmap='jet', alpha=0.5)
            plt.colorbar()
            plt.savefig('./saliency_map_'+str(i)+'_'+gt_classes[i])
            plt.clf()
            if found and self.predictions_assigned:
                plt.imshow(image_pred)
                plt.imshow(sal[self.maxes[i]], cmap = 'jet', alpha = 0.5)
                plt.colorbar()
                plt.savefig('./saliency_map_'+str(i)+'_'+self.predicted_classes[i]+'_predictions')
                plt.clf()

                targ_im = im - sal[self.maxes[i]]
                im_targ = sal[self.maxes[i]] - im

                #np.save('./saliency_map_'+str(i)+'_gt_'+gt_classes[i]+'_pred_'+self.predicted_classes[i]+'im-targ', im_targ)
                #np.save('./saliency_map_'+str(i)+'_gt_'+gt_classes[i]+'_pred_'+self.predicted_classes[i]+'targ-im', targ_im)

                plt.imshow(targ_im)
                plt.savefig('./saliency_map_'+'im'+str(ID)+'_map'+str(i)+'_gt-'+gt_classes[i]+'_pred-'+self.predicted_classes[i]+'targ-im')

                plt.clf()

                plt.imshow(im_targ)
                plt.savefig('./saliency_map_'+'im'+str(ID)+'_map'+str(i)+'_gt-'+gt_classes[i]+'_pred-'+self.predicted_classes[i]+'im-targ')

                plt.clf()


    def explain_many(self, csv_path, prediction_saliencies = True, impaths = None, IDs = None):
        if impaths is None:
            for ID in IDs:
                self.explain_one(csv_path, imID = ID, prediction_saliencies = prediction_saliencies)
        elif IDs is None:
            for impath in impaths:
                self.explain_one(csv_path, impath = impath, prediction_saliencies = prediction_saliencies)

        else:
            print("SELECT EITHER A LIST OF IM IDS OR IMPATHS")

        return 0 



#self, input_size, num_masks, init_mask_res, mask_prob, model_weights, class_list_path
weights = '../Mask_RCNN-master/logs/rgbd_10e_NoAug/mask_rcnn_rgbd_0009.h5'
classes = '../classes.txt'
csv_path = '../rgbd-dataset.csv'
class InferenceConfig(RGBDConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DEPTH_NULL = False
    #RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
config = InferenceConfig()

print(ROOT_DIR)

dataset_val = RGBDDataset()
dataset_val.load_images(os.path.join(ROOT_DIR, '../data/val/'), config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_val.prepare()


drise = DRISE([640,640], 5000, 16, 0.5, weights,classes, dataset_val, config)

#drise.explain_one(csv_path, imID = 0, prediction_saliencies = True)

drise.explain_many(csv_path, IDs = [4,5], prediction_saliencies = True)


