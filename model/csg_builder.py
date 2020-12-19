# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under a NVIDIA Open Source Non-commercial license.)

import torch
import torch.nn as nn
from pdb import set_trace as bp


def chunk_feature(feature, chunk):
    if chunk == 1:
        return feature
    # B x C x H x W => (B*chunk^2) x C x (H//chunk) x (W//chunk)
    _f_new = torch.chunk(feature, chunk, dim=2)
    _f_new = [torch.chunk(f, chunk, dim=3) for f in _f_new]
    f_new = []
    for f in _f_new:
        f_new += f
    f_new = torch.cat(f_new, dim=0)
    return f_new


class CSG(nn.Module):
    def __init__(self, base_encoder, get_head=None, dim=128, K=65536, m=0.999, T=0.07, mlp=True, stages=[4], num_class=12, chunks=[1], task='new',
                 base_encoder_kwargs={}, apool=True
                 ):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(CSG, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.stages = stages
        self.mlp = mlp
        self.base_encoder = base_encoder
        self.chunks = chunks  # chunk feature (segmentation)
        self.task = task  # new, new-seg
        self.attentions = [None for _ in range(len(stages))]
        self.apool = apool

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim, pretrained=True, **base_encoder_kwargs)  # q is for new task
        self.encoder_k = base_encoder(num_classes=dim, pretrained=True, **base_encoder_kwargs)  # ######
        if get_head is not None:
            num_ftrs = self.encoder_q.fc.in_features
            self.encoder_q.fc_new = get_head(num_ftrs, num_class)
        for param in self.encoder_q.fc.parameters():
            param.requires_grad = False

        if mlp:
            fc_q = {}
            fc_k = {}
            for stage in stages:
                if stage > 0:
                    try:
                        # BottleNeck
                        dim_mlp = getattr(self.encoder_q, "layer%d"%stage)[-1].conv3.weight.size()[0]
                    except torch.nn.modules.module.ModuleAttributeError:
                        # BasicBlock
                        dim_mlp = getattr(self.encoder_q, "layer%d"%stage)[-1].conv2.weight.size()[0]
                elif stage == 0:
                    dim_mlp = self.encoder_q.conv1.weight.size()[0]
                fc_q["stage%d"%(stage)] = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, dim))
                fc_k["stage%d"%(stage)] = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, dim))
            self.encoder_q.fc_csg = nn.ModuleDict(fc_q)
            self.encoder_k.fc_csg = nn.ModuleDict(fc_k)
            for param_q, param_k in zip(self.encoder_q.fc_csg.parameters(), self.encoder_k.fc_csg.parameters()):
                param_k.data.copy_(param_q.data)

        for param_k in self.encoder_k.parameters():
            param_k.requires_grad = False  # not update by gradient

        if type(self.encoder_q).__name__ == "ResNet":
            try:
                # BottleNeck
                dims = [self.encoder_q.conv1.weight.size()[0]] + [getattr(self.encoder_q, "layer%d"%stage)[-1].conv3.weight.size()[0] for stage in range(1, 5)]
            except:
                # BasicBlock
                dims = [self.encoder_q.conv1.weight.size()[0]] + [getattr(self.encoder_q, "layer%d"%stage)[-1].conv2.weight.size()[0] for stage in range(1, 5)]
        elif type(self.encoder_q).__name__ == "DigitNet":
                dims = [64 for stage in range(1, 5)]
        for stage in stages:
            self.register_buffer("queue%d"%(stage), torch.randn(dim, K))
            setattr(self, "queue%d"%(stage), nn.functional.normalize(getattr(self, "queue%d"%(stage)), dim=0))
            self.register_buffer("queue_ptr%d"%(stage), torch.zeros(1, dtype=torch.long))

    def control_q_backbone_gradient(self, control):
        for name, param in self.encoder_q.named_parameters():
            if 'fc_new' not in name:
                param.requires_grad = control
        return

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.fc_csg.parameters(), self.encoder_k.fc_csg.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, stage):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(getattr(self, "queue_ptr%d"%(stage)))

        if ptr + batch_size <= self.K:
            getattr(self, "queue%d"%(stage))[:, ptr:ptr + batch_size] = keys.T
        else:
            getattr(self, "queue%d"%(stage))[:, ptr:] = keys[:(self.K - ptr)].T
            getattr(self, "queue%d"%(stage))[:, :ptr + batch_size - self.K] = keys[:(ptr + batch_size - self.K)].T
        ptr = (ptr + batch_size) % self.K  # move pointer
        getattr(self, "queue_ptr%d"%(stage))[0] = ptr

    def adaptive_pool(self, features, attn_from, stage_idx):
        # features and attn_from are paired feature maps, of same size
        assert features.size() == attn_from.size()
        N, C, H, W = features.size()
        assert (attn_from >= 0).float().sum() == N*C*H*W
        attention = torch.einsum('nchw,nc->nhw', [attn_from, nn.functional.adaptive_avg_pool2d(attn_from, (1, 1)).view(N, C)])
        attention = attention / attention.view(N, -1).sum(1).view(N, 1, 1).repeat(1, H, W)
        attention = attention.view(N, 1, H, W)
        # output size: N, C
        return (features * attention).view(N, C, -1).sum(2)

    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        if im_k is None:
            im_k = im_q

        output, features_new = self.encoder_q(im_q, output_features=["layer%d"%stage for stage in self.stages], task=self.task)
        results = {'output': output}

        results['predictions_csg'] = []
        results['targets_csg'] = []
        # predictions: cosine b/w q and k
        # targets: zeros
        with torch.no_grad():  # no gradient to keys
            if self.mlp:
                self._momentum_update_key_encoder()  # update the key encoder
            if self.apool:
                # A-Pool: prepare attention for teacher: get feature of im_k by encoder_q
                _, features_new_k = self.encoder_q.forward_backbone(im_k, output_features=["layer%d"%stage for stage in self.stages])
        _, features_old = self.encoder_k.forward_backbone(im_k, output_features=["layer%d"%stage for stage in self.stages])
        for idx, stage in enumerate(self.stages):
            chunk = self.chunks[idx]
            # compute query features

            q_feature = chunk_feature(features_new["layer%d"%stage], chunk)
            if self.apool:
                # A-Pool prepare attention for teacher: get feature of im_k by encoder_q
                q_feature_k = chunk_feature(features_new_k["layer%d"%stage], chunk)
            if self.mlp:
                if self.apool:
                    q = self.encoder_q.fc_csg["stage%d"%(stage)](self.adaptive_pool(q_feature, q_feature, idx))  # A-Pool
                else:
                    q = self.encoder_q.fc_csg["stage%d"%(stage)](self.encoder_q.avgpool(q_feature).view(features_new["layer%d"%stage].size(0)*chunk**2, -1))
            else:
                if self.apool != 'none':
                    q = self.adaptive_pool(q_feature, q_feature, idx)  # A-Pool
                else:
                    q = self.encoder_q.avgpool(q_feature).view(features_new["layer%d"%stage].size(0)*chunk**2, -1)
            q = nn.functional.normalize(q, dim=1)

            # compute key features
            with torch.no_grad():  # no gradient to keys
                k_feature = chunk_feature(features_old["layer%d"%stage], chunk)
                # A-Pool #############
                if self.mlp:
                    if self.apool:
                        k = self.encoder_k.fc_csg["stage%d"%(stage)](self.adaptive_pool(k_feature, q_feature_k, idx))  # A-Pool
                    else:
                        k = self.encoder_k.fc_csg["stage%d"%(stage)](self.encoder_k.avgpool(k_feature).view(features_old["layer%d"%stage].size(0)*chunk**2, -1))
                else:
                    if self.apool:
                        k = self.adaptive_pool(k_feature, q_feature_k, idx)  # A-Pool
                    else:
                        k = self.encoder_k.avgpool(k_feature).view(features_old["layer%d"%stage].size(0)*chunk**2, -1)
                # #####################
                k = nn.functional.normalize(k, dim=1)

            # compute logits
            # positive logits: Nx1
            l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
            # negative logits: NxK
            l_neg = torch.einsum('nc,ck->nk', [q, getattr(self, "queue%d"%(stage)).clone().detach()])
            # logits: Nx(1+K)
            logits = torch.cat([l_pos, l_neg], dim=1)
            # apply temperature
            logits /= self.T
            # labels: positive key indicators
            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
            self._dequeue_and_enqueue(k, stage)

            results['predictions_csg'].append(logits)
            results['targets_csg'].append(labels)

        return results


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    try:
        tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    except Exception:
        tensors_gather = [tensor]

    output = torch.cat(tensors_gather, dim=0)
    return output
