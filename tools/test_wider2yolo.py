import os.path
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np


class WiderFaceDetection(data.Dataset):
    def __init__(self, txt_path, preproc=None):
        self.preproc = preproc
        self.imgs_path = []
        self.words = []
        f = open(txt_path, 'r')
        lines = f.readlines()
        isFirst = True
        labels = []
        for line in lines:
            line = line.rstrip()
            if line.startswith('#'):
                if isFirst is True:
                    isFirst = False
                else:
                    labels_copy = labels.copy()
                    self.words.append(labels_copy)
                    labels.clear()
                path = line[2:]
                path = txt_path.replace('label.txt', 'images/') + path #.split("/")[-1]
                self.imgs_path.append(path)
            else:
                line = line.split(' ')
                label = [float(x) for x in line]
                labels.append(label)
        self.words.append(labels)

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        img = cv2.imread(self.imgs_path[index])
        height, width, _ = img.shape

        labels = self.words[index]
        annotations = np.zeros((0, 19))
        if len(labels) == 0:
            return annotations
        for idx, label in enumerate(labels):
            annotation = np.zeros((1, 19))
            # bbox
            annotation[0, 0] = int(label[0] ) # x1
            annotation[0, 1] = int(label[1] ) # y1
            annotation[0, 2] = int(label[0]) + int(label[2] ) # x2
            annotation[0, 3] = int(label[1] )+ int(label[3])  # y2

            # landmarks
            # print(int(label[6]))
            annotation[0, 4] = int(label[4] )   # l0_x
            annotation[0, 5] = int(label[5] )   # l0_y
            annotation[0, 6] = 2 if (int(label[6])==0 or int(label[6])==1) else 0  # l4_y
            annotation[0, 7] = int(label[7] )   # l1_x
            annotation[0, 8] = int(label[8] )   # l1_y
            annotation[0, 9] = 2 if (int(label[9])==0 or int(label[9])==1) else 0   # l4_y
            annotation[0, 10] = int(label[10] )  # l2_x
            annotation[0, 11] = int(label[11] )  # l2_y
            annotation[0, 12] = 2 if (int(label[12])==0 or int(label[12])==1) else 0   # l4_y
            annotation[0, 13] = int(label[13])  # l3_x
            annotation[0, 14] = int(label[14] ) # l3_y
            annotation[0, 15] = 2 if (int(label[15])==0 or int(label[15])==1) else 0   # l4_y
            annotation[0, 16] = int(label[16] ) # l4_x
            annotation[0, 17] = int(label[17] ) # l4_y
            annotation[0, 18] = 2 if (int(label[18])==0 or int(label[18])==1) else 0   # l4_y
            
            if annotation[0, 4] < 0:
                annotation[0, 14] = -1
            else:
                annotation[0, 14] = 1

            annotations = np.append(annotations, annotation, axis=0)
        target = np.array(annotations)
        if self.preproc is not None:
            img, target = self.preproc(img, target)

        return torch.from_numpy(img), target


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).
    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations
    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                imgs.append(tup)
            elif isinstance(tup, type(np.empty(0))):
                annos = torch.from_numpy(tup).float()
                targets.append(annos)

    return torch.stack(imgs, 0), targets


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('Missing path to WIDERFACE train folder.')
        print('Run command: python3 train2yolo.py /path/to/original/widerface/train [/path/to/save/widerface/train]')
        exit(1)
    elif len(sys.argv) > 3:
        print('Too many arguments were provided.')
        print('Run command: python3 train2yolo.py /path/to/original/widerface/train [/path/to/save/widerface/train]')
        exit(1)
    original_path = sys.argv[1]


    if len(sys.argv) == 2:
        if not os.path.isdir('widerface'):
            os.mkdir('widerface')
        if not os.path.isdir('widerface/train'):
            os.mkdir('widerface/train')

        save_path = 'widerface/train'
    else:
        save_path = sys.argv[2]

    if not os.path.isfile(os.path.join(original_path, 'label.txt')):
        print('Missing label.txt file.')
        exit(1)

    aa = WiderFaceDetection(os.path.join(original_path, 'label.txt'))
    for i in range(len(aa.imgs_path)):
        img_name = aa.imgs_path[i].split('.')[0].split('/')[-1]
        img = cv2.imread(aa.imgs_path[i])#.split('.')[0].split('/')[-1])
        base_img = os.path.basename(aa.imgs_path[i])
        base_txt = os.path.basename(aa.imgs_path[i])[:-4] + ".txt"
        save_img_path = os.path.join(save_path, base_img)
        save_txt_path = os.path.join(save_path, base_txt)
        with open(save_txt_path, "w") as f:
            height, width, _ = img.shape
            labels = aa.words[i]
            annotations = np.zeros((0, 19))
            if len(labels) == 0:
                continue
            for idx, label in enumerate(labels):
                annotation = np.zeros((1, 19))
                # bbox
                label[0] = max(0, int(label[0]))
                label[1] = max(0, int(label[1]))
                label[2] = min(width - 1, int(label[2]))
                label[3] = min(height - 1, int(label[3]))

                annotation[0, 0] = (int(label[0]) + int(label[2]) / 2) / width  # cx
                annotation[0, 1] = (int(label[1]) + int(label[3]) / 2) / height  # cy
                annotation[0, 2] = int(label[2]) / width  # w
                annotation[0, 3] = int(label[3]) / height  # h
                # landmarks
                annotation[0, 4] = max(0, int(label[4]) / width)  # l0_x
                annotation[0, 5] = max(0, int(label[5]) / height)  # l0_y
                annotation[0, 6] = 2 if (int(label[6])==0 or int(label[6])==1) else 0 
                annotation[0, 7] = max(0, int(label[7]) / width)# l1_x
                annotation[0, 8] = max(0, int(label[8]) / height)  # l1_y
                annotation[0, 9] = 2 if (int(label[9])==0 or int(label[9])==1) else 0 
                annotation[0, 10] = max(0, int(label[10]) / width) # l2_x
                annotation[0, 11] = max(0, int(label[11]) / height)  # l2_y
                annotation[0, 12] = 2 if (int(label[12])==0 or int(label[12])==1) else 0 
                annotation[0, 13] = max(0, int(label[13]) / width)  # l3_x
                annotation[0, 14] = max(0, int(label[14]) / height)  # l3_y
                annotation[0, 15] = 2 if (int(label[15])==0 or int(label[15])==1) else 0 
                annotation[0, 16] = max(0, int(label[16]) / width)  # l4_x
                annotation[0, 17] = max(0, int(label[17]) / height)  # l4_y
                annotation[0, 18] = 2 if (int(label[18])==0 or int(label[18])==1) else 0 
                str_label = "0 "
                # for i in range(len(annotation[0])):
                #     str_label = str_label + " " + str(annotation[0][i])
                # str_label = str_label.replace('[', '').replace(']', '')
                # str_label = str_label.replace(',', '') + '\n'
                # f.write(str_label)
                f.write("%.0f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f \n"%(0, annotation[0, 0], annotation[0, 1], annotation[0, 2], annotation[0, 3], annotation[0, 4], annotation[0, 5], annotation[0, 6], annotation[0, 7], annotation[0, 8], annotation[0, 9], annotation[0, 10], annotation[0, 11], annotation[0, 12], annotation[0, 13], annotation[0, 14], annotation[0, 15], annotation[0, 16], annotation[0, 17], annotation[0, 18]))

        cv2.imwrite(save_img_path, img)
