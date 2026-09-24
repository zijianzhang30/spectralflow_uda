"""Verbatim official full-MLUDA training objective, separated from our algorithm."""
import math
import torch
from torch import nn
import mmd
import utils
from contrastive_loss import SupConLoss
BATCH_SIZE=32
CLASS_NUM=7
crossEntropy=nn.CrossEntropyLoss()
ContrastiveLoss_s=SupConLoss(temperature=0.1)
ContrastiveLoss_t=SupConLoss(temperature=0.1)
DSH_loss=utils.Domain_Occ_loss()

def objective(feature_encoder, source_data, target_data, source_label, epoch, epochs=100):
    # 0
    source_data0 = utils.radiation_noise(source_data)
    source_data0 = source_data0.type(torch.FloatTensor)
    # 1
    source_data1 = utils.flip_augmentation(source_data)
    # 2
    target_data0 = utils.radiation_noise(target_data)
    target_data0 = target_data0.type(torch.FloatTensor)
    # 3
    target_data1 = utils.flip_augmentation(target_data)

    (source_features, source1, _, source_outputs, source_out,
     target_features,_, target1, target_outputs, target_out) = feature_encoder(source_data.cuda(),target_data.cuda())
    (_, source2, _, source_outputs2 ,_,
     _, _, target2, t1, _) = feature_encoder(source_data0.cuda(),target_data0.cuda())
    (_, source3, _, source_outputs3,_,
    _, _, target3, t2, _) =  feature_encoder(source_data1.cuda(),target_data1.cuda())

    softmax_output_t = nn.Softmax(dim=1)(target_outputs).detach()
    _, pseudo_label_t = torch.max(softmax_output_t, 1)

    # Supervised Contrastive Loss
    all_source_con_features = torch.cat([source2.unsqueeze(1), source3.unsqueeze(1)],dim=1)
    all_target_con_features = torch.cat([target2.unsqueeze(1), target3.unsqueeze(1)], dim=1)

    # Loss Cls
    cls_loss = crossEntropy(source_outputs, source_label.cuda())
    # Loss Lmmd
    lmmd_loss = mmd.lmmd(source_features, target_features, source_label,
                         torch.nn.functional.softmax(target_outputs, dim=1), BATCH_SIZE=BATCH_SIZE,
                         CLASS_NUM=CLASS_NUM)
    lambd = 2 / (1 + math.exp(-10 * (epoch) / epochs)) - 1
    # Loss Con_s
    contrastive_loss_s = ContrastiveLoss_s(all_source_con_features, source_label)
    # Loss Con_t
    contrastive_loss_t = ContrastiveLoss_t(all_target_con_features, pseudo_label_t)
    # Loss Occ
    domain_similar_loss = DSH_loss(source_out, target_out)

    loss = cls_loss + 0.01 * lambd * lmmd_loss + contrastive_loss_s + contrastive_loss_t + domain_similar_loss

    return loss, dict(ce=float(cls_loss.detach()), lmmd=float(lmmd_loss.detach()), scl_s=float(contrastive_loss_s.detach()), scl_t=float(contrastive_loss_t.detach()), occupancy=float(domain_similar_loss.detach()))
