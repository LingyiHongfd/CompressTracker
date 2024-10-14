from . import BaseActor
from lib.utils.misc import NestedTensor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
from lib.utils.merge import merge_template_search
from ...utils.heapmap_utils import generate_heatmap
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate
import torch.nn.functional as F



class CompressTrackerActor(BaseActor):
    """ Actor for training CompressTracker models """

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg
        self.sample_prob = 0.

        self.sup_dict = {'SUP_LOGIT': cfg.TRAIN.SUP_LOGIT, 'SUP_FEAT': cfg.TRAIN.SUP_FEAT, 'SUP_REPLACE': cfg.TRAIN.SUP_REPLACE,}
        print ('Distill Supervision: ', f'SUP_LOGIT: {self.sup_dict["SUP_LOGIT"]}, SUP_FEAT: {self.sup_dict["SUP_FEAT"]}, SUP_REPLACE: {self.sup_dict["SUP_REPLACE"]}')

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        out_dict = self.forward_pass(data)

        # compute losses
        loss, status = self.compute_losses(out_dict, data)

        return loss, status

    def set_bernoulli(self, epoch, max_epoch):
        warm_up_epoch = max_epoch // 10
        end_epoch = max_epoch // 10
        sample_prob = 0.5
        if (epoch-1) > warm_up_epoch and (epoch-1) < (max_epoch - end_epoch):
            sample_prob = (1.0 - sample_prob) * (epoch - warm_up_epoch) / (max_epoch - warm_up_epoch - end_epoch) + sample_prob
        elif (epoch-1) >= (max_epoch - end_epoch):
            sample_prob = 1.0
        else:
            sample_prob = 0.5
        self.sample_prob = sample_prob
        self.net.module.set_bernoulli(sample_prob)
        print ('Epoch : ', epoch, ' Bernoulli sample prob: ', sample_prob)

    def forward_pass(self, data):
        # currently only support 1 template and 1 search region
        assert len(data['template_images']) == 1
        assert len(data['search_images']) == 1

        template_list = []
        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][i].view(-1,
                                                             *data['template_images'].shape[2:])  # (batch, 3, 128, 128)
            template_list.append(template_img_i)

        search_img = data['search_images'][0].view(-1, *data['search_images'].shape[2:])  # (batch, 3, 320, 320)

        box_mask_z = None
        ce_keep_rate = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            box_mask_z = generate_mask_cond(self.cfg, template_list[0].shape[0], template_list[0].device,
                                            data['template_anno'][0])

            ce_start_epoch = self.cfg.TRAIN.CE_START_EPOCH
            ce_warm_epoch = self.cfg.TRAIN.CE_WARM_EPOCH
            ce_keep_rate = adjust_keep_rate(data['epoch'], warmup_epochs=ce_start_epoch,
                                                total_epochs=ce_start_epoch + ce_warm_epoch,
                                                ITERS_PER_EPOCH=1,
                                                base_keep_rate=self.cfg.MODEL.BACKBONE.CE_KEEP_RATIO[0])

        if len(template_list) == 1:
            template_list = template_list[0]

        out_dict = self.net(template=template_list,
                            search=search_img,
                            ce_template_mask=box_mask_z,
                            ce_keep_rate=ce_keep_rate,
                            return_last_attn=False,
                            is_training=self.training_mode)

        return out_dict

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        # gt gaussian map
        gt_bbox = gt_dict['search_anno'][-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
        gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)

        # Get boxes
        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
                                                                                                           max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        # compute giou and iou
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(non_blocking=True), torch.tensor(0.0).cuda(non_blocking=True)
        # compute l1 loss
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        # compute location loss
        if 'score_map' in pred_dict:
            location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)
        
        distill_losses = self.compute_losses_distill(pred_dict)
        distill_loss = 0
        
        if self.sup_dict['SUP_LOGIT']:
            distill_logit_loss = distill_losses['distill_logit_loss']
            distill_loss = distill_loss + distill_logit_loss
        if self.sup_dict['SUP_FEAT'] or self.sup_dict['SUP_FEAT_ONLY_LAST']:
            distill_feat_loss = distill_losses['distill_feat_loss']
            distill_loss = distill_loss + distill_feat_loss

        # weighted sum
        loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss
        loss = loss + distill_loss * self.loss_weight['dst']
        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            status = {"Loss/total": loss.item(),
                      "Loss/giou": giou_loss.item(),
                      "Loss/l1": l1_loss.item(),
                      "Loss/location": location_loss.item(),
                      }

            if self.sup_dict['SUP_LOGIT']:
                status["Loss/dst_logit"] = distill_logit_loss.item()
            if self.sup_dict['SUP_FEAT']:
                status["Loss/dst_feat_all"] = distill_feat_loss.item()
            
            status["IoU"] = mean_iou.item()

            return loss, status
        else:
            return loss

    def compute_losses_distill(self, pred_dict):
        if self.sup_dict['SUP_LOGIT']:
            # distill logit loss
            pred_boxes = pred_dict['pred_boxes']
            pred_boxes_teacher = pred_dict['pred_boxes_teacher']
            if torch.isnan(pred_boxes).any():
                raise ValueError("Network outputs is NAN! Stop Training")
            num_queries = pred_boxes.size(1)
            pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
            pred_boxes_teacher_vec = box_cxcywh_to_xyxy(pred_boxes_teacher).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)

            distill_l1_loss = self.objective['l1'](pred_boxes_vec, pred_boxes_teacher_vec)  # (BN,4) (BN,4)
            try:
                distill_giou_loss, iou = self.objective['giou'](pred_boxes_vec, pred_boxes_teacher_vec)  # (BN,4) (BN,4)
            except:
                distill_giou_loss, iou = torch.tensor(0.0).cuda(non_blocking=True), torch.tensor(0.0).cuda(non_blocking=True)
            
            distill_logit_loss = (self.loss_weight['giou'] * distill_giou_loss + self.loss_weight['l1'] * distill_l1_loss) * self.loss_weight['dst_logit']

        if self.sup_dict['SUP_FEAT']:
            # distill feature loss
            inference_loss_block_indexes = pred_dict['inference_feature_loss_block_indexes']
            pred_features = pred_dict['distill_student_features']
            pred_features_teacher = pred_dict['distill_teacher_features']
            feat_sz = pred_dict['feat_sz']
            feat_len_s = feat_sz ** 2

            inference_return_block_nums = 0
            distill_feat_loss = 0
            for ii in range(len(inference_loss_block_indexes)):
                if inference_loss_block_indexes[ii]:
                    distill_feat_layer_loss = 0
                    _pred_features = pred_features[ii][:, -feat_len_s:].unsqueeze(-1).permute((0, 3, 2, 1))
                    _pred_features_teacher = pred_features_teacher[ii][:, -feat_len_s:].detach().unsqueeze(-1).permute((0, 3, 2, 1))
                    distill_feat_layer_loss = distill_feat_layer_loss + F.mse_loss(_pred_features, _pred_features_teacher, reduction="mean")
                    _pred_features = pred_features[ii][:, :-feat_len_s].unsqueeze(-1).permute((0, 3, 2, 1))
                    _pred_features_teacher = pred_features_teacher[ii][:, :-feat_len_s].detach().unsqueeze(-1).permute((0, 3, 2, 1))
                    distill_feat_layer_loss = distill_feat_layer_loss + F.mse_loss(_pred_features, _pred_features_teacher, reduction="mean")
                    distill_feat_loss = distill_feat_loss + (distill_feat_layer_loss / 2)
                    inference_return_block_nums = inference_return_block_nums + 1
            distill_feat_loss = distill_feat_loss / float(inference_return_block_nums) * self.loss_weight['dst_feat']
        
        
        distill_losses = {}
        if self.sup_dict['SUP_LOGIT']:
            distill_losses["distill_logit_loss"] = distill_logit_loss
        if self.sup_dict['SUP_FEAT']:
            distill_losses["distill_feat_loss"] = distill_feat_loss

        return distill_losses

