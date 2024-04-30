import torch
import torch.nn as nn

import torch.nn.functional as F       # fusion the rgb and infrared concat
from mmcv.cnn import normal_init      # fusion the rgb and infrared concat

from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from .. import builder
from ..registry import DETECTORS
from .base import BaseDetector
from .test_mixins import RPNTestMixin, BBoxTestMixin

import numpy as np
from torchvision import transforms


@DETECTORS.register_module
class MoEFusion(BaseDetector, RPNTestMixin, BBoxTestMixin):
    """
    MoE Fusion 
    """

    def __init__(self,        
                 backbone_r,         # rgb
                 backbone_i,         # infrared
                 neck_r=None,
                 neck_i=None,
                 fusion_head=None,   # image fusion 
                 rpn_head_r=None,
                 rpn_head_i=None,
                 bbox_roi_extractor_r=None,
                 bbox_roi_extractor_i=None,
                 bbox_head_r=None,
                 bbox_head_i=None,
                 shared_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):

        super(MoEFusion, self).__init__()

        self.backbone_r = builder.build_backbone(backbone_r)    # rgb
        self.backbone_i = builder.build_backbone(backbone_i)    # infrared

        if neck_r is not None:                                  # rgb
            self.neck_r = builder.build_neck(neck_r)
        if neck_i is not None:                                  # infrared
            self.neck_i = builder.build_neck(neck_i)

        if fusion_head is not None:                                # fusion_head
            self.fusion_head = builder.build_head(fusion_head)

        if shared_head is not None:
            self.shared_head = builder.build_shared_head(shared_head)

        if rpn_head_r is not None:                                # rgb
            self.rpn_head_r = builder.build_head(rpn_head_r)
        if rpn_head_i is not None:                                # infrared
            self.rpn_head_i = builder.build_head(rpn_head_i)

        if bbox_head_r is not None:                                      # rgb
            self.bbox_roi_extractor_r = builder.build_roi_extractor(
                bbox_roi_extractor_r)
            self.bbox_head_r = builder.build_head(bbox_head_r)
        if bbox_head_i is not None:                                      # infrared
            self.bbox_roi_extractor_i = builder.build_roi_extractor(
                bbox_roi_extractor_i)
            self.bbox_head_i = builder.build_head(bbox_head_i)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        
        self._init_layers() 
        self.init_weights(pretrained=pretrained)

    @property
    def with_rpn_r(self):                                                 # rgb
        return hasattr(self, 'rpn_head_r') and self.rpn_head_r is not None

    @property
    def with_rpn_i(self):                                                 # infrared
        return hasattr(self, 'rpn_head_i') and self.rpn_head_i is not None

    def _init_layers(self):
        self.fusion_conv = nn.Conv2d(512, 256, 1) 

        # attention module **************************************************************************
        self.conv_rgbatt1 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_rgbatt1 = nn.BatchNorm2d(128)

        self.conv_tiratt1 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_tiratt1 = nn.BatchNorm2d(128)

        self.conv_rgbatt2 = nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_rgbatt2 = nn.BatchNorm2d(1)
        self.Sigmoid1 = nn.Sigmoid()
                
        self.conv_tiratt2 = nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_tiratt2 = nn.BatchNorm2d(1)
        self.Sigmoid2 = nn.Sigmoid()

    def init_weights(self, pretrained=None):
        super(MoEFusion, self).init_weights(pretrained)

        self.backbone_r.init_weights(pretrained=pretrained)            # rgb
        self.backbone_i.init_weights(pretrained=pretrained)            # infrared

        normal_init(self.fusion_conv, std=0.01)                  # fusion conv init weights
        normal_init(self.conv_rgbatt1, std=0.01)                 # attention conv init weights
        normal_init(self.conv_tiratt1, std=0.01)                 # attention conv init weights
        normal_init(self.conv_rgbatt2, std=0.01)                 # attention conv init weights
        normal_init(self.conv_tiratt2, std=0.01)                 # attention conv init weights

        # ************************* rgb ***************************************
        if self.with_neck_r:
            if isinstance(self.neck_r, nn.Sequential):
                for m in self.neck_r:
                    m.init_weights()
            else:
                self.neck_r.init_weights()
        if self.with_rpn_r:
            self.rpn_head_r.init_weights()
        if self.with_bbox_r:
            self.bbox_roi_extractor_r.init_weights()
            self.bbox_head_r.init_weights()

        # *************************** infrared *************************************
        if self.with_neck_i:
            if isinstance(self.neck_i, nn.Sequential):
                for m in self.neck_i:
                    m.init_weights()
            else:
                self.neck_i.init_weights()
        if self.with_rpn_i:
            self.rpn_head_i.init_weights()
        if self.with_bbox_i:
            self.bbox_roi_extractor_i.init_weights()
            self.bbox_head_i.init_weights()
        
        if self.with_shared_head:
            self.shared_head.init_weights(pretrained=pretrained)
                        
    #Directly extract features from the backbone+neck
    def extract_feat_rgb(self, img):
        x = self.backbone_r(img)
        if self.with_neck_r:
            x = self.neck_r(x)
        return x

    def extract_feat_infrared(self, img):
        x = self.backbone_i(img)
        if self.with_neck_i:
            x = self.neck_i(x)
        return x

    def forward_train(self,
                      img_r,
                      img_i,
                      img_meta,
                      source_img_r,
                      source_img_i,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      proposals=None):

        x_r = self.extract_feat_rgb(img_r)
        x_i = self.extract_feat_infrared(img_i)

        att_rgb = []
        att_tir = []
        x_rf = []
        x_if = []
        for i in range(len(x_r)):    
            concat = torch.cat((x_r[i], x_i[i]), dim=1) 
            concat = self.fusion_conv(concat)  
            concat = F.relu(concat, inplace=True)
            fusion_r = x_r[i] + concat 
            fusion_i = x_i[i] + concat 

            
            # RGB attention
            xr_att = self.conv_rgbatt1(fusion_r)
            xr_att = self.bn_rgbatt1(xr_att)
            xr_att = F.relu(xr_att, inplace=True)
            xr_att = self.conv_rgbatt2(xr_att)
            xr_att = self.bn_rgbatt2(xr_att)
            xr_att = self.Sigmoid1(xr_att)
            fusion_r = fusion_r + fusion_r * xr_att

            # TIR attention
            xi_att = self.conv_tiratt1(fusion_i)
            xi_att = self.bn_tiratt1(xi_att)
            xi_att = F.relu(xi_att, inplace=True)
            xi_att = self.conv_tiratt2(xi_att)
            xi_att = self.bn_tiratt2(xi_att)
            xi_att = self.Sigmoid2(xi_att)
            fusion_i = fusion_i + fusion_i * xi_att

            att_rgb.append(xr_att.detach())
            att_tir.append(xi_att.detach())
            x_rf.append(fusion_r)
            x_if.append(fusion_i)

        x_rf = tuple(x_rf)
        x_if = tuple(x_if)

        losses = dict()

        if self.with_rpn_r:
            rpn_outs_r = self.rpn_head_r(x_rf)
            rpn_loss_inputs_r = rpn_outs_r + (gt_bboxes, img_meta,
                                          self.train_cfg.rpn)
            rpn_losses_r = self.rpn_head_r.loss_r(
                *rpn_loss_inputs_r, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses_r)

            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)

            proposal_inputs_r = rpn_outs_r + (img_meta, proposal_cfg)

            proposal_list_r = self.rpn_head_r.get_bboxes(*proposal_inputs_r)
        else:
            proposal_list_r = proposals

        if self.with_rpn_i:
            rpn_outs_i = self.rpn_head_i(x_if) 
            rpn_loss_inputs_i = rpn_outs_i + (gt_bboxes, img_meta,
                                          self.train_cfg.rpn)
            rpn_losses_i = self.rpn_head_i.loss_i(
                *rpn_loss_inputs_i, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses_i)

            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)

            proposal_inputs_i = rpn_outs_i + (img_meta, proposal_cfg)

            proposal_list_i = self.rpn_head_i.get_bboxes(*proposal_inputs_i)
        else:
            proposal_list_i = proposals

        if self.with_bbox_r:
            bbox_assigner_r = build_assigner(self.train_cfg.rcnn.assigner)
            bbox_sampler_r = build_sampler(
                self.train_cfg.rcnn.sampler, context=self)
            num_imgs_r = img_r.size(0)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs_r)]
            sampling_results_r = []
            for i in range(num_imgs_r):
                assign_result_r = bbox_assigner_r.assign(
                    proposal_list_r[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])                                             
                sampling_result_r = bbox_sampler_r.sample(
                    assign_result_r,
                    proposal_list_r[i],
                    gt_bboxes[i],                                                
                    gt_labels[i],                                                 
                    feats_r=[lvl_feat_r[i][None] for lvl_feat_r in x_rf])           
                sampling_results_r.append(sampling_result_r)

        if self.with_bbox_i:
            bbox_assigner_i = build_assigner(self.train_cfg.rcnn.assigner)
            bbox_sampler_i = build_sampler(
                self.train_cfg.rcnn.sampler, context=self)
            num_imgs_i = img_i.size(0)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs_i)]
            sampling_results_i = []
            for i in range(num_imgs_i):
                assign_result_i = bbox_assigner_i.assign(
                    proposal_list_i[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result_i = bbox_sampler_i.sample(
                    assign_result_i,
                    proposal_list_i[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats_i=[lvl_feat_i[i][None] for lvl_feat_i in x_if])
                sampling_results_i.append(sampling_result_i)

        if self.with_bbox_r:
            rois_r = bbox2roi([res_r.bboxes for res_r in sampling_results_r])
            bbox_feats_r = self.bbox_roi_extractor_r(
                x_rf[:self.bbox_roi_extractor_r.num_inputs], rois_r)

            if self.with_shared_head:
                bbox_feats_r = self.shared_head(bbox_feats_r)

            cls_score_r, bbox_pred_r = self.bbox_head_r(bbox_feats_r)

            bbox_targets_r = self.bbox_head_r.get_target(
                sampling_results_r, gt_bboxes, gt_labels, self.train_cfg.rcnn) 

            loss_bbox_r = self.bbox_head_r.loss_r(cls_score_r, bbox_pred_r,
                                            *bbox_targets_r)
            losses.update(loss_bbox_r)

        if self.with_bbox_i:
            rois_i = bbox2roi([res_i.bboxes for res_i in sampling_results_i])
            bbox_feats_i = self.bbox_roi_extractor_i(
                x_if[:self.bbox_roi_extractor_i.num_inputs], rois_i)

            if self.with_shared_head:
                bbox_feats_i = self.shared_head(bbox_feats_i)

            cls_score_i, bbox_pred_i = self.bbox_head_i(bbox_feats_i)
            
            bbox_targets_i = self.bbox_head_i.get_target(
                sampling_results_i, gt_bboxes, gt_labels, self.train_cfg.rcnn)

            loss_bbox_i = self.bbox_head_i.loss_i(cls_score_i, bbox_pred_i,
                                            *bbox_targets_i)
            losses.update(loss_bbox_i)

        if self.with_fusionhead:
            im_fusion, aux_loss = self.fusion_head(source_img_r/255.0, source_img_i/255.0, att_rgb, att_tir)
            gard_losses = self.fusion_head.gard_loss(im_fusion, source_img_r/255.0, source_img_i/255.0) 
            pixel_losses = self.fusion_head.pixel_loss(im_fusion, source_img_r/255.0, source_img_i/255.0, gt_bboxes) 
            
            losses['max_loss_grad'] = gard_losses['max_loss_grad']
            losses['detcrop_pixel_loss'] = pixel_losses['detcrop_pixel_loss']
            losses['aux_loss'] = aux_loss

        return losses

    def simple_test(self, img_r, img_i, img_meta, source_img_r, source_img_i, proposals=None, rescale=False):
        """Test without augmentation."""
        x_r = self.extract_feat_rgb(img_r)
        x_i = self.extract_feat_infrared(img_i)

        att_rgb = []
        att_tir = []
        x_rf = []
        x_if = []

        for i in range(len(x_r)):         
            concat = torch.cat((x_r[i], x_i[i]), dim=1) 
            concat = self.fusion_conv(concat)
            concat = F.relu(concat, inplace=True)
            fusion_r = x_r[i] + concat
            fusion_i = x_i[i] + concat

            # RGB attention
            xr_att = self.conv_rgbatt1(fusion_r)
            xr_att = self.bn_rgbatt1(xr_att)
            xr_att = F.relu(xr_att, inplace=True)
            xr_att = self.conv_rgbatt2(xr_att)
            xr_att = self.bn_rgbatt2(xr_att)
            xr_att = self.Sigmoid1(xr_att)
            fusion_r = fusion_r + fusion_r * xr_att

            # TIR attention
            xi_att = self.conv_tiratt1(fusion_i)
            xi_att = self.bn_tiratt1(xi_att)
            xi_att = F.relu(xi_att, inplace=True)
            xi_att = self.conv_tiratt2(xi_att)
            xi_att = self.bn_tiratt2(xi_att)
            xi_att = self.Sigmoid2(xi_att)
            fusion_i = fusion_i + fusion_i * xi_att

            att_rgb.append(xr_att.detach())
            att_tir.append(xi_att.detach())
            x_rf.append(fusion_r)
            x_if.append(fusion_i)

        x_rf = tuple(x_rf)
        x_if = tuple(x_if)

        im_fusion, aux_loss = self.fusion_head(source_img_r/255.0, source_img_i/255.0, att_rgb, att_tir)
        
        proposal_list_r = self.simple_test_rpn_r(
            x_rf, img_meta, self.test_cfg.rpn) if proposals is None else proposals
        proposal_list_i = self.simple_test_rpn_i(
            x_if, img_meta, self.test_cfg.rpn) if proposals is None else proposals

        det_bboxes_r, det_labels_r = self.simple_test_bboxes_r(
            x_rf, img_meta, proposal_list_r, self.test_cfg.rcnn, rescale=rescale)
        det_bboxes_i, det_labels_i = self.simple_test_bboxes_i(
            x_if, img_meta, proposal_list_i, self.test_cfg.rcnn, rescale=rescale)

        bbox_results_r = bbox2result(det_bboxes_r, det_labels_r,
                                   self.bbox_head_r.num_classes)
        bbox_results_i = bbox2result(det_bboxes_i, det_labels_i,
                                   self.bbox_head_i.num_classes)

        return bbox_results_r, bbox_results_i, im_fusion
