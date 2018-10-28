# import torch.nn as nn
# import torch.nn.functional as F
# import torch
# from torch.autograd import Variable
# from cityscape_dataset import match_priors
# import numpy as np
#
#
#
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
# def hard_negative_mining(predicted_prob, gt_label, neg_pos_ratio=3.0):
#     """
#     The training sample has much more negative samples, the hard negative mining and produce balanced
#     positive and negative examples.
#     :param predicted_prob: predicted probability for each prior item, dim: (N, H*W*num_prior)
#     :param gt_label: ground_truth label, dim: (N, H*W*num_prior)
#     :param neg_pos_ratio:
#     :return:
#     """
#     pos_flag = gt_label > 0                                        # 0 = negative label
#
#     # Sort the negative samples
#     predicted_prob[pos_flag] = -1.0                                # temporarily remove positive by setting -1
#     _, indices = predicted_prob.sort(dim=1, descending=True)       # sort by descend order, the positives are at the end
#     _, orders = indices.sort(dim=1)                                # sort the negative samples by its original index
#
#     # Remove the extra negative samples
#     num_pos = pos_flag.sum(dim=1, keepdim=True)                     # compute the num. of positive examples
#     num_neg = neg_pos_ratio * num_pos                               # determine of neg. examples, should < neg_pos_ratio
#     neg_flag = orders < num_neg                                     # retain the first 'num_neg' negative samples index.
#
#     return pos_flag, neg_flag
#
#
# class MultiboxLoss(nn.Module):
#
#     def __init__(self, use_gpu= True, iou_threshold=0.5, neg_pos_ratio=3.0):
#         super(MultiboxLoss, self).__init__()
#         self.iou_thres = iou_threshold
#         self.neg_pos_ratio = neg_pos_ratio
#         self.neg_label_idx = 0
#         self.use_gpu = use_gpu
#         #self.matched_label = matched_labels
#
#     def forward(self, confidence, pred_loc, gt_class_labels, gt_bbox_locs):
#         """
#          Compute the Multibox joint loss:
#             L = (1/N) * L_{loc} + L_{class}
#         :param confidence: predicted class probability, dim: (N, H*W*num_prior, num_classes)
#         :param pred_loc: predicted prior bounding boxes, dim: (N, H*W*prior_num, 4)
#         :param gt_class_labels: ground-truth class label, dim:(batch_size, num_objects)
#         :param gt_bbox_locs: ground-truth bounding box for prior, dim: (batch_size, num_objects, 4)
#         :return:
#         """
#
#
#         # For lower pytorch
#         num_batches = pred_loc.size(0)
#         num_classes = confidence.shape[2]
#         loc_target = gt_bbox_locs
#         confidence_target = gt_class_labels
#         '''
#         for i in range(num_batches):
#             gt_class_label  = gt_class_labels[i].data
#             gt_bbox_loc = gt_bbox_locs[i].data
#             default = priors.data
#             default = default[i, :, :]
#             _, __,loc_target, confidence_target = match_priors(default, gt_bbox_loc, gt_class_label,
#                                                            loc_target, confidence_target, i)
#         if self.use_gpu:
#             loc_target = loc_target.cuda()
#             confidence_target = confidence_target.cuda()
#         '''
#         loc_target = Variable(loc_target, requires_grad= False)
#         confidence_target = Variable(confidence_target, requires_grad= False)
#         pos = confidence_target > 0
#         #print("gt_class_labels greater than 0: ", np.any(gt_class_labels))
#         #print("pos greater than 0: ", np.any(pos))
#         # print("confidence target greater than 0: ", np.any(confidence_target))
#         #print("shape of pos is: ", pos.shape)
#         #print("pos is: ", pos.shape)
#         #num_pos = pos.sum(dim=1, keepdim= True)
#         pos_idx = pos.unsqueeze(pos.dim()).expand_as(pred_loc)
#         loc_predicted = pred_loc[pos_idx].view(-1, 4)
#         loc_target = loc_target[pos_idx].view(-1, 4).float()
#         loc_huber_loss  = F.smooth_l1_loss(loc_predicted, loc_target, size_average=False)
#
#
#         # Compute confidence loss
#         batch_confidences = confidence.view(-1, num_classes)
#         batch_confidences_max = batch_confidences.data.max()
#         loss_ = torch.log(torch.sum(torch.exp(batch_confidences - batch_confidences_max), 1, keepdim=True))\
#         + batch_confidences_max
#         conf_loss = loss_ - batch_confidences.gather(1, confidence_target.view(-1, 1))
#
#         # Hard Negative Mining
#
#         conf_loss = conf_loss.view(num_batches, -1)
#         conf_loss[pos] = 0  # filter out pos boxes for now
#
#         _, loss_idx = conf_loss.sort(1, descending=True)
#         _, idx_rank = loss_idx.sort(1)
#         num_pos = pos.long().sum(1, keepdim=True)
#         # print("number of positive is: ", num_pos)
#         num_neg = torch.clamp(self.neg_pos_ratio*num_pos, max=pos.size(1)-1)
#         # print("number of negative is: ", num_neg)
#         neg = idx_rank < num_neg.expand_as(idx_rank)
#
#         # Confidence Loss Including Positive and Negative Examples
#         pos_idx = pos.unsqueeze(2).expand_as(confidence)
#         neg_idx = neg.unsqueeze(2).expand_as(confidence)
#         # print("pos_idx shape: ", pos_idx.shape)
#         # print("neg_idx shape: ", neg_idx.shape)
#         #print("pos_idx are: ", pos_idx)
#         #print("neg_idx are: ", neg_idx)
#         conf_p = confidence[(pos_idx+neg_idx).gt(0)].view(-1, num_classes)
#         targets_weighted = confidence_target[(pos+neg).gt(0)]
#         # print("confidence_target are: ", targets_weighted)
#         # print("shape of gt_class_labels", gt_class_labels.shape)
#         # print("shape of confidence are： ", confidence.shape)
#         # print("shape of confp and targets_weights are: ", conf_p.shape, confidence_target.shape)
#         conf_loss = F.cross_entropy(conf_p, targets_weighted, size_average=False)
#
#         # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
#
#         N = num_pos.data.sum().float()
#         loc_huber_loss /= N
#         conf_loss /= N
#         # print(conf_loss, loc_huber_loss)
#
#         return conf_loss, loc_huber_loss





















import torch.nn as nn
import torch.nn.functional as F
import torch

def hard_negative_mining(predicted_prob, gt_label, neg_pos_ratio=3.0):
    """
    The training sample has much more negative samples, the hard negative mining produces balanced
    positive and negative examples.
    :param predicted_prob: predicted probability for each prior item, dim: (N, H*W*num_prior)
    :param gt_label: ground_truth label, dim: (N, H*W*num_prior)
    :param neg_pos_ratio:
    :return:
    """
    pos_flag = gt_label > 0                                        # 0 = negative label

    # Sort the negative samples
    predicted_prob[pos_flag] = -1.0                                # temporarily remove positive by setting -1
    _, indices = predicted_prob.sort(dim=1, descending=True)       # sort by descend order, the positives are at the end
    _, orders = indices.sort(dim=1)                                # sort the negative samples by its original index

    # Remove the extra negative samples
    num_pos = pos_flag.sum(dim=1, keepdim=True)                     # compute the num. of positive examples
    num_neg = neg_pos_ratio * num_pos                               # determine of neg. examples, should < neg_pos_ratio
    neg_flag = orders < num_neg                                     # retain the first 'num_neg' negative samples index.

    return pos_flag, neg_flag


class MultiboxLoss(nn.Module):

    def __init__(self, iou_threshold=0.5, neg_pos_ratio=3.0):
        super(MultiboxLoss, self).__init__()
        self.neg_pos_ratio = neg_pos_ratio
        self.neg_label_idx = 0

    def forward(self, confidence, pred_loc, gt_class_labels, gt_bbox_loc):
        """
         Compute the Multibox joint loss:
            L = (1/N) * L_{loc} + L_{class}
        :param confidence: predicted class probability, dim: (N, H*W*num_prior, num_classes)
        :param pred_loc: predicted prior bounding boxes, dim: (N, H*W*prior_num, 4)
        :param gt_class_labels: ground-truth class label, dim:(N, H*W*num_prior)
        :param gt_bbox_loc: ground-truth bounding box for prior, dim: (N, H*W*num_prior, 4)
        :return:
        """
        # Do the hard negative mining and produce balanced positive and negative examples
        with torch.no_grad():
            # print(confidence.shape)
            neg_class_prob = -F.log_softmax(confidence, dim=2)[:, :, self.neg_label_idx]      # select neg. class prob.
            # print("neg_class_prob is: ", neg_class_prob)
            pos_flag, neg_flag = hard_negative_mining(neg_class_prob, gt_class_labels, neg_pos_ratio=self.neg_pos_ratio)
            sel_flag = pos_flag | neg_flag
            num_pos = pos_flag.sum(dim=1, keepdim=True).float().sum()

        # Loss for the classification
        num_classes = confidence.shape[2]
        sel_conf = confidence[sel_flag]
        #sel_conf = F.softmax(sel_conf)

        # conf_loss = F.cross_entropy(sel_conf.view(-1, num_classes), gt_class_labels[sel_flag], reduction='sum') / num_pos
        conf_loss = F.cross_entropy(sel_conf.view(-1, num_classes), gt_class_labels[sel_flag], reduction='elementwise_mean')


        # TODO: implementation on bounding box regression
        pos_pred_loc = pred_loc[pos_flag, :]
        pos_gt_bbox_loc = gt_bbox_loc[pos_flag, :]
        # loc_huber_loss = F.smooth_l1_loss(pos_pred_loc, pos_gt_bbox_loc, reduction='sum') / num_pos
        loc_huber_loss = F.smooth_l1_loss(pos_pred_loc, pos_gt_bbox_loc, reduction='elementwise_mean')


        return conf_loss, loc_huber_loss