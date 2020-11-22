import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
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

from rgbd import RGBDConfig

GpuIndex = 0
os.environ['CUDA_VISIBLE_DEVICES'] = str(GpuIndex)

class DRISE():
    def __init__(self, input_size, num_masks, init_mask_res, mask_prob, model_weights, class_list_path):
        self.input_size = input_size
        self.num_masks = num_masks
        self.s = init_mask_res
        self.p1 = mask_prob
        self.get_model(model_weights)
        self.get_class_list(class_list_path)
        self.create_encoder()
        self.predictions_assigned = False

    def get_dataset(self, gt_csv_path):
        return CsvDataset(gt_csv_path)

    def get_model(self, weights):
        class InferenceConfig(RGBDConfig):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DEPTH_NULL = False

        inference_config = InferenceConfig()
        inference_config.display()

        self.model = modellib.MaskRCNN(mode='inference', config = inference_config, model_dir = ROOT_DIR)
        self.model.load_weights(weights, by_name=True)


    def generate_mask(self):
        cell_size = np.ceil(np.array(self.input_size) / self.s)
        up_size = (self.s + 1) * cell_size

        grid = np.random.rand(self.s,self.s) < self.p1
        grid = grid.astype('float32')

        x = np.random.randint(0, cell_size[0])
        y = np.random.randint(0, cell_size[1])
        #Could generator be faster here?
        mask = resize(grid, up_size, order = 1, mode = 'reflect', anti_aliasing = False)[x:x+self.input_size[0], y:y+self.input_size[1]]
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

    def transform_gt_classes(self, classes):
        return self.ohe.fit_transform(classes.reshape(-1,1))

    def read_image(self, impath):
        return np.load(impath)

    def get_predictions(self, impath):
        im = self.read_image(impath)
        mask = self.generate_mask()
        results = self.model.detect([im*mask]) 
        return results[0], mask

    def get_predictions_no_mask(self, impath):
        im = self.read_image(impath)
        print(impath)
        results = self.model.detect([im])
        return results[0]

    def assign_predictions(self, gt_rois, gt_classes, results):
        # assign predictions to their respective targets
        iou_thres = 0.4
        print(results['rois'])
        iou = self.np_iou(gt_rois, results['rois'])
        iou = np.where(iou > iou_thres, 1, 0)
        confs = iou * results['scores']
        confs = np.argmax(confs, axis = 1)
        mask = np.any(confs, axis = 1)
        confs = np.where(mask == True, confs, None)
        self.inds = confs[mask]
        self.maxes = dict(zip(np.arange(len(gt_classes)), confs))
        #assert is_unique(maxes) any duplicates?
        # Check that rois are in x1, y1, x2, y2 order 
        self.predicted_boxes = results['rois']
        self.predicted_classes = results['classes']
        self.predictions_assigned = True


    def compute_saliency(self,impath, gt_rois, gt_classes):
        if self.predictions_assigned:
            gt_rois = np.append(gt_rois, self.predicted_boxes[self.inds,:], axis = 0)
            gt_classes = np.append(gt_classes, self.predicted_classes[self.inds])

        sal = np.zeros(len(gt_classes), *self.input_shape)
        gt_classes = self.transform_gt_classes(gt_classes)
        for i in tqdm(range(self.num_masks), desc="Detecting Masked Images"):
            # mask shape = (1, h, w)
            results, mask = self.get_predictions(impath)
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
           
            sal += (mask.transpose(2,1,0)*sensitivity).transpose(2,1,0)

        return sal



    def process_logits(self, logits):
        # Add any processing to logits here
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
            ID = dataset.get_image_id(impath)
        elif imID is not None:
            impath = dataset.get_image_path(imID)
            ID = imID
        else:
            print("Please select either one image or an image ID")

        
            
        gt_rois, gt_classes = dataset.get_rois_and_classes(ID)

        results_noMask = self.get_predictions_no_mask(impath)

        if prediction_saliencies:
            assignments = self.assign_predictions(gt_rois, gt_classes, results_noMask)
        else:
            assignments = None

        sal = self.compute_saliency(impath, gt_rois, gt_classes, assignments = None)

        for i, im in enumerate(sal[:len(gt_classes),:,:]):
            found = True
            this_gt_box = gt_rois[i % len(gt_classes)]
            if self.predictions_assigned:
                key = self.maxes[i]
                if key is not None:
                    this_pred_box = self.predicted_boxes[self.maxes[i]]
                else:
                    found = False
                    #missed detection
            image = cv2.imread(impath)
            image = cv2.rectangle(image, (this_gt_box[0], this_gt_box[1]), (this_gt_box[2], this_gt_box[3]), (0,255,0), 2)
            if self.predictions_assigned and found:
                image = cv2.rectangle(image, (this_pred_box[0], this_pred_box[1]), (this_pred_box[2], this_pred_box[3]), (255,0,0), 2)
            plt.imshow(image)
            if found and self.predictions_assigned:
                plt.savefig('./saliency_map_'+str(i)+'_'+self.predicted_classes[i]+'_predictions')
            plt.savefig('./saliency_map_'+str(i)+'_'+gt_classes[i])



#self, input_size, num_masks, init_mask_res, mask_prob, model_weights, class_list_path
weights = '../Mask_RCNN-master/logs/train1_noAug/mask_rcnn_rgbd_0004.h5'
classes = '../classes.txt'
csv_path = '../rgbd-dataset.csv'
drise = DRISE([640,480], 5000, 16, 0.5, weights,classes)

drise.explain_one(csv_path, imID = 0)


