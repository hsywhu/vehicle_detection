import sys
import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch
from bbox_helper import nms, loc2bbox, generate_prior_bboxes
import matplotlib.patches as patches


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if not torch.cuda.is_available():
    test_model = torch.load('trained_model/ssd_net_1022_augmentation_100_zoomout.pth', map_location='cpu')
else:
    test_model = torch.load('trained_model/ssd_net_1022_augmentation_100_zoomout.pth')
test_model.to(device)
test_model.eval()

img_file_path = sys.argv[1] # the index should be 1, 0 is the 'eval.py'
sample_img = cv2.imread(img_file_path)
sample_img = cv2.resize(sample_img, (300, 300))
fig, ax = plt.subplots(1)
sample_img_rgb = sample_img[:, :, (2, 1, 0)]
ax.imshow(sample_img_rgb)

# normalization
mean = np.asarray((127, 127, 127)).reshape(1, 1, 3)
std = 128.0
sample_img = (sample_img - mean) / std

# to tensor
sample_img = sample_img.transpose((2, 0, 1))

sample_img = torch.tensor(sample_img).unsqueeze(0).float()
sample_img = sample_img.to(device)
pred_conf, pred_locs = test_model.forward(sample_img)

pred_conf = pred_conf.squeeze(0)
pred_locs = pred_locs.squeeze(0)

priors = generate_prior_bboxes(prior_layer_cfg=[{'layer_name': '1', 'feature_dim_hw': (19, 19), 'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, 1.0)},
                                               {'layer_name': '2', 'feature_dim_hw': (10, 10), 'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, 1.0)},
                                               {'layer_name': '3', 'feature_dim_hw': (5, 5),'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, 1.0)},
                                               {'layer_name': '4', 'feature_dim_hw': (3, 3),'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, 1.0)},
                                               {'layer_name': '5', 'feature_dim_hw': (3, 3),'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, 1.0)},
                                               {'layer_name': '6', 'feature_dim_hw': (1, 1),'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, 1.0)}
                                               ])

priors = priors.unsqueeze(0)
priors = priors.to(device)
pred_locs = pred_locs.to(device)
pred_locs = loc2bbox(pred_locs, priors)[0, :, :]

for label_id in range(1, 3):
    if label_id == 1:
        keep = nms(pred_locs, pred_conf[:, label_id], prob_threshold=0.2, top_k=20)
    elif label_id == 2:
        keep = nms(pred_locs, pred_conf[:, label_id], prob_threshold=0.4)
    else:
        print("incorrect label id")
        exit(-1)

    for j in range(keep[1]):
        i = keep[0][j]
        # print('i', i)
        # print(keep.shape)
        w = pred_locs[i][2] * 300
        # print('w:', w)
        h = pred_locs[i][3] * 300
        left_top = [pred_locs[i][0] - pred_locs[i][2] / 2,
                    pred_locs[i][1] - pred_locs[i][3] / 2]
        left_top[0] *= 300
        left_top[1] *= 300
        # print(left_top, w, h)
        if label_id == 1:
            rect = patches.Rectangle(left_top, w, h, linewidth=1, edgecolor='green', facecolor='none')
        elif label_id == 2:
            rect = patches.Rectangle(left_top, w, h, linewidth=1, edgecolor='red', facecolor='none')
        ax.add_patch(rect)

plt.show()