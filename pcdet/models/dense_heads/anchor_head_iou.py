import numpy as np
import torch
import math
import torch.nn as nn

from ...utils import loss_utils
from .anchor_head_template import AnchorHeadTemplate
from .target_assigner.atss_target_assigner import ATSSTargetAssigner
from .target_assigner.axis_aligned_target_assigner_add_gt import AxisAlignedTargetAssigner
from ...ops.iou3d.oriented_iou_loss import cal_iou_3d, cal_diou_3d, cal_giou_3d
import pdb

eps = 1e-6


def clamp(x, min=1e-6, max=1e6):
    return torch.clamp(x, min=min, max=max)


class AnchorHeadIoU(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )
        self.num_anchors_per_location = sum(self.num_anchors_per_location)

        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )


        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None
        self.init_weights()

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def get_target_assigner(self, anchor_target_cfg, sec = False):
        if anchor_target_cfg.NAME == 'ATSS':
            target_assigner = ATSSTargetAssigner(
                topk=anchor_target_cfg.TOPK,
                box_coder=self.box_coder,
                use_multihead=self.use_multihead,
                match_height=anchor_target_cfg.MATCH_HEIGHT
            )
        elif anchor_target_cfg.NAME == 'AxisAlignedTargetAssigner':
            target_assigner = AxisAlignedTargetAssigner(
                model_cfg=self.model_cfg,
                class_names=self.class_names,
                box_coder=self.box_coder,
                match_height=anchor_target_cfg.MATCH_HEIGHT,
                sec = sec
            )
        else:
            raise NotImplementedError
        return target_assigner

    def build_losses(self, losses_cfg):
        self.add_module(
            'cls_loss_func',
            loss_utils.SigmoidQualityFocalClassificationLoss(alpha=0.25, gamma=2.0)
        )
        self.add_module(
            'dir_loss_func',
            loss_utils.WeightedCrossEntropyLoss()
        )
    
    def get_iou(self, bboxes1, bboxes2):
        x1u, y1u, z1u = bboxes1[:,:,0], bboxes1[:,:,1], bboxes1[:,:,2]
        l1, w1, h1 =  torch.exp(bboxes1[:,:,3]), torch.exp(bboxes1[:,:,4]), torch.exp(bboxes1[:,:,5])
        x2u, y2u, z2u = bboxes2[:,:,0], bboxes2[:,:,1], bboxes2[:,:,2]
        l2, w2, h2 =  torch.exp(bboxes2[:,:,3]), torch.exp(bboxes2[:,:,4]), torch.exp(bboxes2[:,:,5])

        # clamp is necessray to aviod inf.
        # l1, w1, h1 = torch.clamp(l1, max=10), torch.clamp(w1, max=10), torch.clamp(h1, max=10)

        # we emperically scale the y/z to make their predictions more sensitive.
        factor = 2
        x1 = x1u
        y1 = y1u * factor
        z1 = z1u * factor
        x2 = x2u
        y2 = y2u * factor
        z2 = z2u * factor

        debox1 = torch.stack([x1, y1, z1, l1, w1, h1, bboxes1[:,:,6]], -1) 
        debox2 = torch.stack([x2, y2, z2, l2, w2, h2, bboxes2[:,:,6]], -1) 

        ## gciou
        _, ious, cd = cal_diou_3d(debox1, debox2, return_dist=True)
        da = (debox1[:,:,6] - debox2[:,:,6]) % math.pi
        delta_a = torch.where(da > 0.5 * math.pi, math.pi - da, da).clamp(min=eps,max=0.49 * math.pi)
        
        theta_bd = abs(torch.atan(
            debox2[:,:,4] / (debox2[:,:,5] - debox1[:,:,4]/torch.sin(delta_a + eps) + eps)
        ))
   

        gc_loss = -torch.log(ious + eps)  * torch.exp(torch.pow(delta_a, 2)) + torch.tan(delta_a + eps)
        # gc_loss =  torch.where( (delta_a > theta_bd), 
        #                 - torch.log(ious + eps)  /  (ious + 1) + torch.tan(delta_a + eps),
        #                 -torch.log(ious+ eps)  /  (ious + 1)+ torch.tan(delta_a + eps))

        # gc_loss = -torch.log(ious+ eps)  /  (1 + ious) - torch.log(torch.cos(delta_a) + eps)
        # gc_loss = -torch.log(ious + eps) / (1 + ious)  + torch.tan(delta_a + eps)
        # center_loss = - 0.1 * torch.log(1 - cd + eps) 

        # loss = gc_loss + center_loss

        # sgciou
        union =  l1 * w1 * h1 + l2 * w2 * h2
        normed_u = 1 + (union - union.min()) / (union.max() - union.min())
        sgc_loss = gc_loss * normed_u.detach()

        return ious, gc_loss


    def get_clsreg_targets(self):
        box_preds = self.forward_ret_dict['box_preds']
        box_dir_cls_preds = self.forward_ret_dict.get('dir_cls_preds', None)
        box_reg_targets = self.forward_ret_dict['box_reg_targets']
        box_cls_labels = self.forward_ret_dict['box_cls_labels'].clone()
        batch_size = int(box_preds.shape[0])
        h = box_preds.shape[1]
        w = box_preds.shape[2]

        # enlarge the positive samples
        box_cls_labels = box_cls_labels.view(batch_size, h, w, -1)
        box_cls_labels_an0 = box_cls_labels[:,:,:,0].unsqueeze(-1)
        box_cls_labels_an1 = box_cls_labels[:,:,:,1].unsqueeze(-1)
        box_cls_labels_an0_tmp1 = box_cls_labels_an0.roll(shifts = 1, dims = 1)
        box_cls_labels_an0_tmp2 = box_cls_labels_an0.roll(shifts = -1, dims = 1)
        box_cls_labels_an1_tmp1 = box_cls_labels_an1.roll(shifts = 1, dims = 2)
        box_cls_labels_an1_tmp2 = box_cls_labels_an1.roll(shifts = -1, dims = 2)
        box_cls_labels_an1[box_cls_labels_an1_tmp1==1] = 1
        box_cls_labels_an1[box_cls_labels_an1_tmp2==1] = 1
        box_cls_labels_an0[box_cls_labels_an0_tmp1==1] = 1
        box_cls_labels_an0[box_cls_labels_an0_tmp2==1] = 1

        box_cls_labels = torch.cat([box_cls_labels_an0, box_cls_labels_an1], dim = -1)
        re_box_cls_labels = box_cls_labels.view(batch_size, -1)

        if isinstance(self.anchors, list):
            if self.use_multihead:
                anchors = torch.cat(
                    [anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1]) for anchor in
                     self.anchors], dim=0)
            else:
                anchors = torch.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors
        anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
        box_preds = box_preds.view(batch_size, -1,
                                   box_preds.shape[-1] // self.num_anchors_per_location if not self.use_multihead else
                                   box_preds.shape[-1])

        iou, _ =  self.get_iou(box_preds, box_reg_targets)

        # filter the background.
        with torch.no_grad():
            iou_guided_cls_labels = re_box_cls_labels * iou.detach()
        return re_box_cls_labels, iou_guided_cls_labels


    def get_iou_guided_reg_loss(self):
        box_preds = self.forward_ret_dict['box_preds']
        box_dir_cls_preds = self.forward_ret_dict.get('dir_cls_preds', None)
        box_reg_targets = self.forward_ret_dict['box_reg_targets']
        box_cls_labels = self.re_box_cls_labels
        batch_size = int(box_preds.shape[0])
 
        box_cls_labels = box_cls_labels.view(batch_size, -1)
        positives = box_cls_labels > 0
        reg_weights = positives.float()
        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)

        if isinstance(self.anchors, list):
            if self.use_multihead:
                anchors = torch.cat(
                    [anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1]) for anchor in
                     self.anchors], dim=0)
            else:
                anchors = torch.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors
        anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
        box_preds = box_preds.view(batch_size, -1,
                                   box_preds.shape[-1] // self.num_anchors_per_location if not self.use_multihead else
                                   box_preds.shape[-1])

        iou, iou_loss = self.get_iou(box_preds, box_reg_targets)

        # iou_loss_n = iou_loss
        # iou_loss_n = torch.clamp(iou_loss_n,min=0.,max = 1.0)
        # iou_loss_m = 1 - iou_loss_n
        iou_loss_src = iou_loss * reg_weights
        iou_loss = iou_loss_src.sum() / batch_size
        iou_loss = iou_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']
        box_loss = iou_loss
        tb_dict = {
            'rpn_loss_loc': iou_loss.item()
        }


        if box_dir_cls_preds is not None:
            dir_targets = self.get_direction_target(
                anchors, box_reg_targets,
                dir_offset=self.model_cfg.DIR_OFFSET,
                num_bins=self.model_cfg.NUM_DIR_BINS
            )

            dir_logits = box_dir_cls_preds.view(batch_size, -1, self.model_cfg.NUM_DIR_BINS)
            weights = positives.type_as(dir_logits)
            weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)
            dir_loss = self.dir_loss_func(dir_logits, dir_targets, weights=reg_weights)
            dir_loss = dir_loss.sum() / batch_size
            dir_loss = dir_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['dir_weight']
            box_loss += dir_loss
            tb_dict['rpn_loss_dir'] = dir_loss.item()

        return box_loss, tb_dict

    

    def get_iou_guided_cls_loss(self):
        cls_preds = self.forward_ret_dict['cls_preds']
        box_cls_labels = self.iou_guided_cls_labels
        batch_size = int(cls_preds.shape[0])
        cared = box_cls_labels >= 0  
        positives = box_cls_labels > 0
        negatives = box_cls_labels == 0
        negative_cls_weights = negatives * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        reg_weights = positives.float()

        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_targets = box_cls_labels * cared.type_as(box_cls_labels)
        cls_targets = cls_targets.unsqueeze(dim=-1)
        cls_preds = cls_preds.view(batch_size, -1, self.num_class)

        cls_loss_src = self.cls_loss_func(cls_preds, cls_targets, weights=cls_weights)  # [N, M]
        cls_loss = cls_loss_src.sum() / batch_size

        cls_loss = cls_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']
        tb_dict = {
            'rpn_loss_cls': cls_loss.item()
        }
        return cls_loss, tb_dict



    def get_loss(self):
        self.re_box_cls_labels, self.iou_guided_cls_labels = self.get_clsreg_targets()
        box_loss, tb_dict = self.get_iou_guided_reg_loss()
        cls_loss, tb_dict_cls = self.get_iou_guided_cls_loss()
        tb_dict.update(tb_dict_cls)

        rpn_loss = cls_loss + box_loss

        if rpn_loss.isnan():
            print(cls_loss)
            print(box_loss)
            pdb.set_trace()

        tb_dict['rpn_loss'] = rpn_loss.item()
        return rpn_loss, tb_dict


    def assign_targets_sec(self, gt_boxes):
        """
        Args:
            gt_boxes: (B, M, 8)
        Returns:

        """
        targets_dict = self.target_assigner_sec.assign_targets(
            self.anchors, gt_boxes
        )
        return targets_dict

    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']

        box_preds = self.conv_box(spatial_features_2d)
        cls_preds = self.conv_cls(spatial_features_2d) 

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]


        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds


        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None

        if self.training:
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )
            self.forward_ret_dict.update(targets_dict)

        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = False

        return data_dict

