# encoding: utf-8
import numpy as np

from fastreid.layers import *
from fastreid.utils.weight_init import weights_init_kaiming, weights_init_classifier
from .build import REID_HEADS_REGISTRY


@REID_HEADS_REGISTRY.register()
class HRCNHead(nn.Module):
    def __init__(self, cfg, in_feat, num_classes, pool_layer, circle_num, adj_mat=None):
        super().__init__()
        self.neck_feat = cfg.MODEL.HEADS.NECK_FEAT
        self.block_num_list = cfg.MODEL.HEADS.BLOCK_NUM_LIST
        self.gcn_channels = cfg.MODEL.HEADS.GCN_CHANNELS
        self.node_list = cfg.MODEL.HEADS.NODE_LIST
        self.in_channel_list = cfg.MODEL.HEADS.IN_CHANNEL_LIST
        self.gate = cfg.MODEL.HEADS.GATE
        self.node_num = sum(self.node_list)
        self.pool_layer = pool_layer
        self.adj_mat = adj_mat
        self.dropout = cfg.MODEL.HEADS.DROPOUT
        self.add_last_fc_layer = cfg.MODEL.HEADS.LAST_FC_LAYER
        self.circle_num = circle_num
        self.feat_channels_num = cfg.MODEL.HEADS.FINAL_CHANNEL_NUM
        self.feat_channels = cfg.MODEL.HEADS.FINAL_CHANNEL
        self.reduce_diag = cfg.MODEL.HEADS.REDUCE_DIAG

        # Align the channels of different level features
        # from the same stage in ResNet using these layers.
        self.align_multilevel_layers = nn.ModuleList()
        self.multilevel_bns = nn.ModuleList()
        for i, j in enumerate(self.node_list):
            self.align_multilevel_layers.append(nn.ModuleList(
                [nn.Sequential(nn.Conv2d(self.in_channel_list[i], self.gcn_channels[0] // pow(2, 2 - i), 1),
                               get_norm(cfg.MODEL.HEADS.NORM,
                                        self.gcn_channels[0] // pow(2, 2 - i),
                                        cfg.MODEL.HEADS.NORM_SPLIT),
                               nn.ReLU(True))
                 for _ in range(j)]))

            self.multilevel_bns.append(nn.ModuleList(
                [get_norm(cfg.MODEL.HEADS.NORM,
                          self.gcn_channels[0] // pow(2, 2 - i),
                          cfg.MODEL.HEADS.NORM_SPLIT)
                 for _ in range(j)]))

        for idx in range(len(self.multilevel_bns)):
            for i in range(len(self.multilevel_bns[idx])):
                self.multilevel_bns[idx][i].apply(weights_init_kaiming)

        # Align the lower concatenated feature channels
        # to the higher features in the next stage using these layers.
        self.align_lower_layers = nn.ModuleList()
        for i, j in enumerate(self.node_list[:-1]):
            self.align_lower_layers.append(
                nn.Sequential(nn.Conv2d(self.gcn_channels[2] // pow(2, 2 - i) * (i + j),
                                        self.gcn_channels[0] // pow(2, 1 - i), 1),
                              get_norm(cfg.MODEL.HEADS.NORM,
                                       self.gcn_channels[0] // pow(2, 1 - i),
                                       cfg.MODEL.HEADS.NORM_SPLIT),
                              nn.ReLU(True)))

        # Align channels of all part feature
        self.align_part_layers = nn.ModuleList()
        for i in range(circle_num + 1):
            self.align_part_layers.append(nn.Sequential(nn.Conv2d(self.in_channel_list[-1], self.gcn_channels[0], 1),
                                                        get_norm(cfg.MODEL.HEADS.NORM,
                                                                 self.gcn_channels[0],
                                                                 cfg.MODEL.HEADS.NORM_SPLIT),
                                                        nn.ReLU(True)))

        self.part_bns = nn.ModuleList([get_norm(cfg.MODEL.HEADS.NORM,
                                                self.gcn_channels[0],
                                                cfg.MODEL.HEADS.NORM_SPLIT)
                                       for _ in range(self.circle_num + 1)])
        for i in range(len(self.part_bns)):
            self.part_bns[i].apply(weights_init_kaiming)

        # Align the channels of the highest level feature to the part feature's.
        if self.gcn_channels[0] != self.gcn_channels[2]:
            self.align_high_layer = nn.Sequential(
                nn.Linear(self.gcn_channels[2], self.gcn_channels[0], bias=False),
                nn.BatchNorm1d(self.gcn_channels[0])
            )

        else:
            self.align_high_layer = nn.Identity()

        # The first weight matrices of GCN.
        # The number of weight matrices is len(ResNet-stages) in cross-level complementary branch,
        # and 1 in regional complementary branch.
        self.gcn_weight_1 = nn.ParameterList()
        self.gcn_weight_1.extend(
            [nn.Parameter(
                torch.zeros([self.gcn_channels[0] // pow(2, 2 - i), self.gcn_channels[1] // pow(2, 2 - i)]).normal_(
                    mean=0, std=0.01),
                requires_grad=True
            )
                for i in range(len(self.node_list))])
        self.gcn_weight_1.append(
            nn.Parameter(
                torch.zeros([self.gcn_channels[0], self.gcn_channels[1]]).normal_(mean=0, std=0.01),
                requires_grad=True
            ))

        # The second weight matrices of GCN.
        self.gcn_weight_2 = nn.ParameterList()
        self.gcn_weight_2.extend(
            [nn.Parameter(
                torch.zeros([self.gcn_channels[1] // pow(2, 2 - i), self.gcn_channels[2] // pow(2, 2 - i)]).normal_(
                    mean=0, std=0.01),
                requires_grad=True
            )
                for i in range(len(self.node_list))])
        self.gcn_weight_2.append(
            nn.Parameter(
                torch.zeros([self.gcn_channels[1], self.gcn_channels[2]]).normal_(mean=0, std=0.01),
                requires_grad=True
            ))

        # The bias of GCN
        self.gcn_bias_1 = nn.ParameterList()
        self.gcn_bias_1.extend([nn.Parameter(torch.zeros([self.gcn_channels[1] // pow(2, 2 - i)]),
                                             requires_grad=True)
                                for i in range(len(self.node_list))])
        self.gcn_bias_1.append(nn.Parameter(torch.zeros([self.gcn_channels[1]]), requires_grad=True))

        self.gcn_bias_2 = nn.ParameterList()
        self.gcn_bias_2.extend([nn.Parameter(torch.zeros([self.gcn_channels[2] // pow(2, 2 - i)]),
                                             requires_grad=True)
                                for i in range(len(self.node_list))])
        self.gcn_bias_2.append(nn.Parameter(torch.zeros([self.gcn_channels[2]]), requires_grad=True))

        # Downsampling layers
        self.gcn_downsample_1 = nn.ModuleList()
        if self.gcn_channels[0] != self.gcn_channels[1]:
            self.gcn_downsample_1.extend([
                nn.Sequential(
                    nn.Linear(self.gcn_channels[0] // pow(2, 2 - i), self.gcn_channels[1] // pow(2, 2 - i), bias=False)
                )
                for i in range(len(self.node_list))])
            self.gcn_downsample_1.append(
                nn.Sequential(
                    nn.Linear(self.gcn_channels[0], self.gcn_channels[1], bias=False)
                ))
        else:
            self.gcn_downsample_1.extend([nn.Identity() for _ in range(len(self.node_list) + 1)])

        self.gcn_downsample_2 = nn.ModuleList()
        if self.gcn_channels[1] != self.gcn_channels[2]:
            self.gcn_downsample_2.extend([
                nn.Sequential(
                    nn.Linear(self.gcn_channels[1] // pow(2, 2 - i), self.gcn_channels[2] // pow(2, 2 - i), bias=False),
                )
                for i in range(len(self.node_list))])
            self.gcn_downsample_2.append(
                nn.Sequential(
                    nn.Linear(self.gcn_channels[1], self.gcn_channels[2], bias=False),
                ))
        else:
            self.gcn_downsample_2.extend([nn.Identity() for _ in range(len(self.node_list) + 1)])

        if self.dropout:
            self.gcn_dropout_1 = nn.ModuleList()
            self.gcn_dropout_1.extend([nn.Dropout(p=0.5) for _ in range(4)])
            self.gcn_dropout_2 = nn.ModuleList()
            self.gcn_dropout_2.extend([nn.Dropout(p=0.5) for _ in range(4)])
        else:
            self.gcn_dropout_1 = nn.ModuleList()
            self.gcn_dropout_1.extend([nn.Identity() for _ in range(4)])
            self.gcn_dropout_2 = nn.ModuleList()
            self.gcn_dropout_2.extend([nn.Identity() for _ in range(4)])

        # Batch norm after Dropout.
        self.gcn_bn_1 = nn.ModuleList()
        self.gcn_bn_1.append(nn.BatchNorm1d(self.node_list[0]).apply(weights_init_kaiming))
        self.gcn_bn_1.extend([nn.BatchNorm1d(i + 1).apply(weights_init_kaiming)
                              for i in self.node_list[1:]])
        self.gcn_bn_1.append(nn.BatchNorm1d(self.circle_num + 2).apply(weights_init_kaiming))
        self.gcn_bn_2 = nn.ModuleList()
        self.gcn_bn_2.append(nn.BatchNorm1d(self.node_list[0]).apply(weights_init_kaiming))
        self.gcn_bn_2.extend([nn.BatchNorm1d(i + 1).apply(weights_init_kaiming)
                              for i in self.node_list[1:]])
        self.gcn_bn_2.append(nn.BatchNorm1d(self.circle_num + 2).apply(weights_init_kaiming))

        # identity classification layer
        cls_type = cfg.MODEL.HEADS.CLS_LAYER

        self.feat_layers = nn.ModuleList()
        for i, j in enumerate(self.feat_channels_num):
            self.feat_layers.extend([nn.Sequential(nn.Conv2d(self.gcn_channels[2], self.feat_channels[i], 1),
                                                   get_norm(cfg.MODEL.HEADS.NORM,
                                                            self.feat_channels[i],
                                                            cfg.MODEL.HEADS.NORM_SPLIT),
                                                   nn.ReLU(True))
                                     for _ in range(j)])

        for idx in range(len(self.feat_layers)):
            self.feat_layers[idx][0].apply(weights_init_kaiming)
            self.feat_layers[idx][1].apply(weights_init_kaiming)

        if self.add_last_fc_layer:
            feat_dim = np.sum(np.multiply(self.feat_channels_num, self.feat_channels)).item()
            self.last_fc_layer = nn.Linear(feat_dim, in_feat)
            self.last_fc_layer.apply(weights_init_kaiming)
        else:
            in_feat = np.sum(np.multiply(self.feat_channels_num, self.feat_channels)).item()

        self.last_bn = nn.BatchNorm1d(in_feat)

        self.last_bn.apply(weights_init_kaiming)

        if cls_type == 'linear':
            self.global_classifier = nn.Linear(in_feat, num_classes, bias=False)
        else:
            raise KeyError(f"{cls_type} is invalid, please choose from 'linear'.")

        self.global_classifier.apply(weights_init_classifier)

    def device(self):
        return self.gcn_weight_1[0].device

    def edges_mat(self, feat, reduce_diag=True, alpha=0.1):
        features = feat.clone()
        b, n, m = features.shape
        edges = features.bmm(features.permute(0, 2, 1)) / m

        edges = torch.where(edges > self.gate, edges,
                            torch.zeros([b, n, n]).to(device=self.device(), dtype=edges.dtype))
        edges = F.normalize(edges, p=1, dim=-1)

        if reduce_diag:
            diag = torch.diag(torch.ones([n])).to(self.device())
            edges = torch.where(diag > 0, torch.zeros([b, n, n]).to(self.device()), edges)

        if self.adj_mat is None:
            return edges

        edges = edges + alpha * self.adj_mat.to(self.device())

        return edges

    def matrix_norm(self, edges):
        b, m, n = edges.shape
        edges += torch.diag(torch.ones(m).to(self.device()))
        D_mat = torch.sum(edges, dim=-1).to(self.device())
        D_mat = torch.pow(D_mat, -0.5)
        D_mat = torch.diag_embed(D_mat)
        edges = D_mat.matmul(edges).matmul(D_mat).type(torch.float32)

        return edges

    def run_gcn(self, features, edges, num):
        x1 = torch.matmul(edges, features)
        x1 = torch.matmul(x1, self.gcn_weight_1[num]) + self.gcn_bias_1[num]
        x1 = self.gcn_dropout_1[num](x1)
        x1 = self.gcn_bn_1[num](x1)
        x1 = F.relu(x1)
        x1 = self.gcn_downsample_1[num](features) + x1

        x2 = torch.matmul(edges, x1)
        x2 = torch.matmul(x2, self.gcn_weight_2[num]) + self.gcn_bias_2[num]
        x2 = self.gcn_dropout_2[num](x2)
        x2 = self.gcn_bn_2[num](x2)
        x2 = F.relu(x2)
        x2 = self.gcn_downsample_2[num](x1) + x2

        return x2

    def forward(self, features_list, targets=None):
        assert isinstance(features_list, list), 'The Type of input of the gcn head should be a list.'

        target_multi_features_list = []
        multi_features_list = features_list[:13]
        cur_sum_num = 0
        for i, block_num in enumerate(self.block_num_list):
            begin = block_num - self.node_list[i] + cur_sum_num
            cur_sum_num += block_num
            if cur_sum_num >= 13:
                target_multi_features_list.append(multi_features_list[begin:])
            else:
                target_multi_features_list.append(multi_features_list[begin:cur_sum_num])

        local_feat_list = []
        lower_feat = []
        last_multi_num = 0

        # multi-level feature fusion
        for idx in range(len(target_multi_features_list)):
            bn_feat_list = []

            if idx > 0:
                bn_feat = self.align_lower_layers[idx - 1](lower_feat[-1])
                bn_feat = F.relu(bn_feat)
                bn_feat = bn_feat[..., 0, 0]
                bn_feat_list.append(bn_feat.unsqueeze(1))

            for i, features in enumerate(target_multi_features_list[idx]):
                feat = self.align_multilevel_layers[idx][i](features)
                feat = self.pool_layer(feat)
                bn_feat = self.multilevel_bns[idx][i](feat)
                bn_feat = F.relu(bn_feat)
                bn_feat = bn_feat[..., 0, 0]
                bn_feat_list.append(bn_feat.unsqueeze(1))

            multi_bn_feats = torch.cat(bn_feat_list, dim=1)

            edges = self.edges_mat(multi_bn_feats, reduce_diag=self.reduce_diag)
            edges = self.matrix_norm(edges)
            gcn_feats = self.run_gcn(multi_bn_feats, edges, idx)

            multi_feat_list = []
            for i in range(len(bn_feat_list)):
                feat = gcn_feats[:, i, :]
                multi_feat_list.append(feat)

            if idx < len(target_multi_features_list) - 1:
                lower_feat.append(torch.cat(multi_feat_list, dim=1).unsqueeze(2).unsqueeze(3))
            else:
                last_multi_num = len(bn_feat_list)
                lower_feat.append(multi_feat_list[-1])
                for i in range(len(multi_feat_list)):
                    bn_feat = multi_feat_list[i].unsqueeze(2).unsqueeze(3)
                    bn_feat = self.feat_layers[i](bn_feat)
                    bn_feat = bn_feat[..., 0, 0]
                    local_feat_list.append(bn_feat)

        # part-level feature fusion
        bn_feat_list = []
        for i, features in enumerate(features_list[13:]):
            feat = self.align_part_layers[i](features)
            feat = self.pool_layer(feat)
            bn_feat = self.part_bns[i](feat)
            bn_feat = F.relu(bn_feat)
            bn_feat = bn_feat[..., 0, 0]
            bn_feat_list.append(bn_feat.unsqueeze(1))

        bn_feat_list.append(self.align_high_layer(lower_feat[-1]).unsqueeze(1))

        part_bn_feats = torch.cat(bn_feat_list, dim=1)

        edges = self.edges_mat(part_bn_feats, reduce_diag=self.reduce_diag)
        edges = self.matrix_norm(edges)
        gcn_feats = self.run_gcn(part_bn_feats, edges, 3)

        part_feat_list = []
        for i in range(self.circle_num + 2):
            feat = gcn_feats[:, i, :]
            part_feat_list.append(feat)

        for i, j in enumerate(range(last_multi_num, last_multi_num + self.circle_num + 2)):
            bn_feat = part_feat_list[i].unsqueeze(2).unsqueeze(3)
            bn_feat = self.feat_layers[j](bn_feat)
            bn_feat = bn_feat[..., 0, 0]
            local_feat_list.append(bn_feat)

        last_feat = torch.cat(local_feat_list, dim=-1)

        if self.add_last_fc_layer:
            last_feat = self.last_fc_layer(last_feat)

        last_bn_feat = self.last_bn(last_feat)

        # Evaluation
        if not self.training:
            if self.neck_feat == 'before':
                return [last_feat]
            return [last_bn_feat]

        # Training
        cls_outputs = []
        try:
            cls_outputs.append(self.global_classifier(last_bn_feat))
        except TypeError:
            cls_outputs.append(self.global_classifier(last_bn_feat, targets))

        pred_class_logits = []

        pred_class_logits.append(F.linear(last_bn_feat, self.global_classifier.weight))

        if self.neck_feat == 'before':
            feat_output_list = [last_feat]
        else:
            feat_output_list = [last_bn_feat]

        return cls_outputs, pred_class_logits, feat_output_list
