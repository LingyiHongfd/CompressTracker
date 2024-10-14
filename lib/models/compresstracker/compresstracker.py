"""
CompressTracker model.
"""
import math
import os
from typing import List

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head
from lib.models.compresstracker.vit_compress import vit_base_patch16_224_compress
from lib.models.compresstracker.vit_teacher import vit_base_patch16_224_teacher
from lib.utils.box_ops import box_xyxy_to_cxcywh


class CompressTracker(nn.Module):
    """ This is the base class for CompressTracker """

    def __init__(self, transformer, box_head, ori_transformer, ori_box_head, aux_loss=False, head_type="CORNER", sup_dict=None):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = transformer
        self.box_head = box_head

        self.ori_backbone = ori_transformer
        self.ori_box_head = ori_box_head

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

        self.sup_dict = sup_dict

    def set_bernoulli(self, sample_prob):
        self.backbone.set_bernoulli(sample_prob)
    
    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                is_training=False,
                ):
        x, aux_dict = self.backbone(z=template, x=search,
                                    ce_template_mask=ce_template_mask,
                                    ce_keep_rate=ce_keep_rate,
                                    return_last_attn=return_last_attn,
                                    is_training=is_training)

        if is_training != 'test':
            with torch.no_grad():
                ori_x, ori_aux_dict = self.ori_backbone(z=template, x=search,
                                            ce_template_mask=ce_template_mask,
                                            ce_keep_rate=ce_keep_rate,
                                            return_last_attn=return_last_attn,
                                            is_training=is_training)

        # Forward head
        feat_last = x
        if isinstance(x, list):
            feat_last = x[-1]
        out = self.forward_head(feat_last, None)
        
        aux_dict["feat_sz"] = self.feat_sz_s

        if is_training != 'test':
            if self.sup_dict['SUP_LOGIT']:
                with torch.no_grad():
                    teacher_feat_last= ori_aux_dict["distill_teacher_features_norm_last"]
                    teacher_out = self.forward_ori_head(teacher_feat_last, None)["pred_boxes"]
                    aux_dict["pred_boxes_teacher"] = teacher_out
            
            if self.sup_dict['SUP_FEAT']:
                aux_dict['distill_teacher_features'] = ori_aux_dict['distill_teacher_features']

        out.update(aux_dict)
        out['backbone_feat'] = x

        return out

    def forward_head(self, cat_feature, gt_score_map=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        enc_opt = cat_feature[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError
    
    def forward_ori_head(self, cat_feature, gt_score_map=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        enc_opt = cat_feature[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.ori_box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.ori_box_head(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError


def build_compresstracker(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    if cfg.MODEL.BACKBONE.PRETRAIN_FILE and ('OSTrack' not in cfg.MODEL.BACKBONE.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.BACKBONE.PRETRAIN_FILE)
    else:
        pretrained = ''

    sup_dict = {'SUP_LOGIT': cfg.TRAIN.SUP_LOGIT, 'SUP_FEAT': cfg.TRAIN.SUP_FEAT, 'SUP_REPLACE': cfg.TRAIN.SUP_REPLACE,}

    
    
    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_compress':
        backbone = vit_base_patch16_224_compress(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                                distill_layers=cfg.MODEL.BACKBONE.DISTILL_LAYER, sup_dict=sup_dict,
                                                )
        ori_backbone = vit_base_patch16_224_teacher(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                                distill_layers=cfg.MODEL.BACKBONE.DISTILL_LAYER, sup_dict=sup_dict,
                                                )
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    else:
        raise NotImplementedError

    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)
    ori_backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    box_head = build_box_head(cfg, hidden_dim)

    ori_box_head = build_box_head(cfg, hidden_dim)

    model = CompressTracker(
        backbone,
        box_head,
        ori_backbone,
        ori_box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
        sup_dict=sup_dict,
    )


    if 'OSTrack' in cfg.MODEL.PRETRAIN_FILE and training:
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('Load pretrained model init from: ' + cfg.MODEL.PRETRAIN_FILE)

    # load pretrained model for distill model
    if 'OSTrack' in cfg.MODEL.PRETRAIN_FILE and training:
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")["net"]
        model_dict = model.state_dict()
        distill_ratio = model.backbone.distill_ratio
        backbone_depth = model.backbone.depth
        distill_block_maps = list(range(-1, backbone_depth, distill_ratio))[1::]
        pretrained_dict_update = {}
        print ('init ',distill_block_maps, ' -> ', [bi//distill_ratio for bi in distill_block_maps])
        for k, v in checkpoint.items():
            if k in model_dict:
                pretrained_dict_update[k] = v
            elif k[:7] == 'module.':
                if k[7:] in model_dict:
                    pretrained_dict_update[k[7:]] = v
                    
            for bi in distill_block_maps:
                if f'blocks.{bi}.' in k:
                    pretrained_dict_update[k.replace(f'backbone.blocks.{bi}.', f'backbone.distill_blocks.{bi//distill_ratio}.')] = v
                elif k[:7] == 'module.' and f'blocks.{bi}.' in k[7:]:
                    pretrained_dict_update[k[7:].replace(f'backbone.blocks.{bi}.', f'backbone.distill_blocks.{bi//distill_ratio}.')] = v
                
        model_dict.update(pretrained_dict_update)
        model.load_state_dict(model_dict)
        print('Load pretrained model for distill model from: ' + cfg.MODEL.PRETRAIN_FILE)

    
    # load pretrained model for ori model
    if 'OSTrack' in cfg.MODEL.PRETRAIN_FILE and training:
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")["net"]
        model_dict = model.state_dict()
        distill_ratio = model.backbone.distill_ratio
        backbone_depth = model.backbone.depth
        distill_block_maps = list(range(-1, backbone_depth, distill_ratio))[1::]
        pretrained_dict_update = {}
        print ('init ',distill_block_maps, ' -> ', [bi//distill_ratio for bi in distill_block_maps])
        for k, v in checkpoint.items():
            if 'ori_'+k in model_dict:
                pretrained_dict_update['ori_'+k] = v
            elif k[:7] == 'module.':
                if k[7:] in model_dict:
                    pretrained_dict_update['ori_'+k[7:]] = v

        model_dict.update(pretrained_dict_update)
        model.load_state_dict(model_dict)
        print('Load pretrained model for ori model from: ' + cfg.MODEL.PRETRAIN_FILE)
    


    return model
