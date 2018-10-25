import numpy as np
import torch.nn
from torch.utils.data import Dataset
from PIL import Image
import os
import cv2
import json
from torchvision import transforms
from bbox_helper import generate_prior_bboxes, match_priors
from augmentation_helper import SSDAugmentation
import random
import matplotlib.patches as patches
from matplotlib import pyplot as plt


def get_bbox_label(json_dir):
    gt_bboxes = []
    gt_labels = []
    human_label = ['person', 'persongroup', 'rider']
    vehicle_label = ['car', 'cargroup', 'truck', 'bus', 'bicycle']
    with open(json_dir) as f:
        frame_info = json.load(f)
        for selected_object in frame_info['objects']:
            polygons = np.asarray(selected_object['polygon'])
            left_top = np.min(polygons, axis=0)
            right_bottom = np.max(polygons, axis=0)
            wh = right_bottom - left_top
            cx = left_top[0] + wh[0] / 2
            cy = left_top[1] + wh[1] / 2
            gt_bboxes.append([cx, cy, wh[0], wh[1]])
            if selected_object['label'] in human_label:
                gt_labels.append(1)
            elif selected_object['label'] in vehicle_label:
                gt_labels.append(2)
            else:
                gt_labels.append(0)
    assert len(gt_labels) == len(gt_labels)
    return gt_bboxes, gt_labels


def resize(img, bbox, w, h):
    # print(img.size)
    w_ratio = float(w / img.size[0])
    h_ratio = float(h / img.size[1])
    img = img.resize((w, h))
    #print(img.size)
    bbox[:, [0, 2]] *= w_ratio
    bbox[:, [1, 3]] *= h_ratio
    bbox[:, [0, 2]] /= w
    bbox[:, [1, 3]] /= h
    return img, bbox

class CityScapeDataset(Dataset):

    def __init__(self, img_dir_list, json_dir_list, transform, mode = 'train' ,augmentation_ratio = 50):
        self.transform = transform
        self.mode = mode
        # TODO: implement prior bounding box
        self.prior_bboxes = generate_prior_bboxes(prior_layer_cfg=
                                                  [{'layer_name': '1', 'feature_dim_hw': (19, 19), 'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, 1.0)},
                                                   {'layer_name': '2', 'feature_dim_hw': (10, 10), 'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, 1.0)},
                                                   {'layer_name': '3', 'feature_dim_hw': (5, 5),'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, 1.0)},
                                                   {'layer_name': '4', 'feature_dim_hw': (3, 3),'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, 1.0)},
                                                   {'layer_name': '5', 'feature_dim_hw': (3, 3),'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, 1.0)},
                                                   {'layer_name': '6', 'feature_dim_hw': (1, 1),'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, 1.0)}
                                                   ])

        # Pre-process parameters:
        # Normalize: (I-self.mean)/self.std
        self.mean = np.asarray((127, 127, 127)).reshape(3, 1, 1)
        self.img_dir_list = img_dir_list
        self.json_dir_list = json_dir_list
        self.original_len = len(self.img_dir_list)
        # augmentation_img_dir_list = []
        # augmentation_json_dir_list = []
        # for i in range(len(img_dir_list)):
        #     if random.randint(1, 101) < augmentation_ratio:
        #         augmentation_img_dir_list.append(img_dir_list[i])
        #         augmentation_json_dir_list.append(json_dir_list[i])
        # self.img_dir_list = self.img_dir_list + augmentation_img_dir_list
        # self.json_dir_list = self.json_dir_list + augmentation_json_dir_list
        # assert len(self.img_dir_list) == len(self.json_dir_list)
        self.std = 128.0

    def get_prior_bbox(self):
        return self.prior_bboxes

    def __len__(self):
        return len(self.img_dir_list)

    def __getitem__(self, idx):
        """
        Load the data from list, and match the ground-truth bounding boxes with prior bounding boxes.
        :return bbox_tensor: matched bounding box, dim: (num_priors, 4)
        :return bbox_label: matched classification label, dim: (num_priors)
        """

        # TODO: implement data loading
        # 1. Load image as well as the bounding box with its label
        # 2. Normalize the image with self.mean and self.std
        # 3. Convert the bounding box from corner form (left-top, right-bottom): [(x,y), (x+w, y+h)] to
        #    center form: [(center_x, center_y, w, h)]
        # 4. Normalize the bounding box position value from 0 to 1
        img_dir = self.img_dir_list[idx]
        json_dir = self.json_dir_list[idx]

        #sample_bboxes = None
        # sample_img = Image.open(img_dir)
        sample_img = cv2.imread(img_dir)
        #print("size of image is: ", sample_img.size)

        gt_bboxes, gt_labels = get_bbox_label(json_dir)

        gt_bboxes = torch.tensor(gt_bboxes, dtype=torch.float32)
        gt_labels = torch.tensor(gt_labels, dtype=torch.int32)


        # '''
        #     show loaded image with bbox
        # '''
        # npimg = np.array(sample_img, dtype=np.int32)
        # # print(npimg.shape)
        # fig, ax = plt.subplots(1)
        # ax.imshow(npimg, interpolation='nearest')
        # _gt_locs = np.asarray(gt_bboxes)
        # _gt_labels = np.asarray(gt_labels)
        # print(_gt_labels[np.where(_gt_labels > 0)])
        # print(_gt_locs[np.where(_gt_labels > 0)])
        # print('-' * 15)
        # for i in range(_gt_locs[np.where(_gt_labels > 0)].shape[0]):
        #     w = _gt_locs[np.where(_gt_labels > 0)][i][2]
        #     h = _gt_locs[np.where(_gt_labels > 0)][i][3]
        #     left_top = [_gt_locs[np.where(_gt_labels > 0)][i][0] - _gt_locs[np.where(_gt_labels > 0)][i][2] / 2,
        #                 _gt_locs[np.where(_gt_labels > 0)][i][1] - _gt_locs[np.where(_gt_labels > 0)][i][3] / 2]
        #     rect = patches.Rectangle(left_top, w, h, linewidth=1, edgecolor='r', facecolor='none')
        #     ax.add_patch(rect)
        # plt.show()


        #
        # sample_img, gt_bboxes = resize(sample_img, gt_bboxes, 300, 300)
        # sample_img = np.array(sample_img, dtype=np.float64)
        # sample_img = sample_img.transpose((2, 0, 1))
        # sample_img = (sample_img - self.mean) / self.std
        # data augmentation
        data_augmentation = SSDAugmentation(mode= self.mode)
        sample_img = np.array(sample_img, dtype=np.float64)
        sample_img, gt_bboxes, gt_labels = data_augmentation(sample_img, gt_bboxes, gt_labels)

        # 5. Do the matching prior and generate ground-truth labels as well as the boxes
        bbox_tensor, bbox_label_tensor = match_priors(self.prior_bboxes, gt_bboxes, gt_labels)
        output_prior_bboxes = self.prior_bboxes
        # [DEBUG] check the output.
        # assert isinstance(bbox_label_tensor, torch.Tensor)
        # assert isinstance(bbox_tensor, torch.Tensor)
        # assert bbox_tensor.dim() == 2
        # assert bbox_tensor.shape[1] == +4
        # assert bbox_label_tensor.dim() == 1
        # assert bbox_label_tensor.shape[0] == bbox_tensor.shape[0]

        # return sample_img, bbox_tensor, bbox_label_tensor.long()
        # print(gt_bboxes.shape, gt_labels.shape)
        return sample_img, bbox_tensor, bbox_label_tensor.long(), output_prior_bboxes
