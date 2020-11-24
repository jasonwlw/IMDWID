import numpy as np

class CsvDataset():
    def __init__(self, csv_path):
        self.read_csv(csv_path)


    def read_csv(self, csv_path):
        self.dataset = {}
        self.ids_to_path = {}
        with open(csv_path, 'r') as f0:
            ID = 0
            for i,line in enumerate(f0.readlines()):
                line = line.strip()
                impath = line.split(',')[0]
                x1 = int(line.split(',')[1])
                y1 = int(line.split(',')[2])
                x2 = int(line.split(',')[3])
                y2 = int(line.split(',')[4])
                cls = line.split(',')[5]
                if impath in self.ids_to_path.values():
                    ID_found = self.get_image_id(impath)
                    self.dataset[ID_found]['rois'].append([x1,y1,x2,y2])
                    self.dataset[ID_found]['classes'].append(cls)
                else:
                    self.ids_to_path[ID] = impath
                    self.dataset[ID] = {}
                    self.dataset[ID]['rois'] = [[x1,y1,x2,y2]]
                    self.dataset[ID]['classes'] = [cls]
                    ID += 1

    def get_image_path(self, ID):
        return self.ids_to_path[ID]

    def get_image_id(self, impath):
        return list(self.ids_to_path.keys())[list(self.ids_to_path.values()).index(impath)]

    def get_rois(self, ID):
        return np.array(self.dataset[ID]['rois'])

    def get_classes(self, ID):
        print("HERE", np.array(self.dataset[ID]['classes']))
        return np.array(self.dataset[ID]['classes'])

    def get_rois_and_classes(self, ID):
        rois = self.get_rois(ID)
        clses = self.get_classes(ID)
        return rois, clses

