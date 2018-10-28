import torch
import pandas as pd
from torch.utils.data import DataLoader
import cityscape_dataset
from bbox_helper import nms, loc2bbox
import numpy as np
import matplotlib.patches as patches
from matplotlib import pyplot as plt
import torch.nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

test_model = torch.load('trained_model/ssd_net_1022_augmentation_100_zoomout.pth', map_location = 'cpu')
test_model.to(device)
test_model.eval()

test_img_dir_list = pd.read_csv('test_img_dir_list.csv').iloc[:, 1].tolist()
test_json_dir_list = pd.read_csv('test_json_dir_list.csv').iloc[:, 1].tolist()

test_dataset = cityscape_dataset.CityScapeDataset(test_img_dir_list, test_json_dir_list, mode = 'eval', transform = None)

test_loader = DataLoader(test_dataset, batch_size = 16, shuffle = True, num_workers = 0)

for batch_idx, (img, bbox, label, priors) in enumerate(test_loader):
    # print(batch_idx)
    inputs = img.float().to(device)
    pred_conf, pred_locs = test_model.forward(inputs)
    # print(pred_conf)
    # softmax = torch.nn.LogSoftmax(dim = 2)
    # pred_conf = softmax(pred_conf)
    for i in range(pred_locs.shape[0]):
        single_pred_locs = pred_locs[i, :, :]
        single_pred_conf = pred_conf[i, :, :]

        single_pred_locs = loc2bbox(single_pred_locs, priors[i:i + 1, :, :].to(device))[0, :, :]
        img[i, :, :, :] = img[i, (2, 1, 0)]
        img[i, :, :, :] = img[i, :, :, :] * 128.0 + 127.0
        npimg = img[i, :, :, :].numpy().reshape(3, 300, 300)
        npimg = npimg.astype(dtype='int')
        fig, ax = plt.subplots(1)
        ax.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')

        for label_id in range(1,3):
            if label_id == 1:
                keep = nms(single_pred_locs, single_pred_conf[:, label_id], prob_threshold = 0.2, top_k=20)
            elif label_id == 2:
                keep = nms(single_pred_locs, single_pred_conf[:, label_id], prob_threshold=0.4)
            else:
                print("incorrect label id")
                exit(-1)
            # keep = single_pred_locs
            # print('keep:', keep.shape)


            # print(keepbase_output_layer_indices[0][0])
            # keep = keep.numpy()
            for j in range(keep[1]):
                i = keep[0][j]
                # print('i', i)
                # print(keep.shape)
                w = single_pred_locs[i][2] * 300
                # print('w:', w)
                h = single_pred_locs[i][3] * 300
                left_top = [single_pred_locs[i][0] - single_pred_locs[i][2] / 2,
                            single_pred_locs[i][1] - single_pred_locs[i][3] / 2]
                left_top[0] *= 300
                left_top[1] *= 300
                # print(left_top, w, h)
                if label_id == 1:
                    rect = patches.Rectangle(left_top, w, h, linewidth=1, edgecolor='green', facecolor='none')
                elif label_id ==2:
                    rect = patches.Rectangle(left_top, w, h, linewidth=1, edgecolor='red', facecolor='none')
                ax.add_patch(rect)
        plt.show()
    break