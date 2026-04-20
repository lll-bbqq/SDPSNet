from functools import partial
import torch
import torch.nn as nn

from ...utils.spconv_utils import replace_feature, spconv


def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):

    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, bias=None, norm_fn=None, downsample=None, indice_key=None):
        super(SparseBasicBlock, self).__init__()

        assert norm_fn is not None
        if bias is None:
            bias = norm_fn is not None
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu(out.features))

        return out


class SparseDynamicParallelAttention(spconv.SparseModule):
    def __init__(self, in_channels, num_heads=4, attn_ratio=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.attn_channels = int(in_channels * attn_ratio) or 1
        self.in_channels = in_channels
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        # Dynamic weight generation
        self.q_conv = spconv.SubMConv3d(in_channels, self.attn_channels, kernel_size=1, bias=False)
        self.k_conv = spconv.SubMConv3d(in_channels, self.attn_channels, kernel_size=1, bias=False)
        self.v_conv = spconv.SubMConv3d(in_channels, self.attn_channels, kernel_size=1, bias=False)
        
        # Normalization layers for Q, K, V
        self.q_bn = norm_fn(self.attn_channels)
        self.k_bn = norm_fn(self.attn_channels)
        self.v_bn = norm_fn(self.attn_channels)

        # Parallel channel interaction branch
        self.parallel_conv = spconv.SubMConv3d(in_channels, self.attn_channels, kernel_size=1, bias=False)
        self.parallel_bn = norm_fn(self.attn_channels)

        # Output convolution
        self.out_conv = spconv.SubMConv3d(self.attn_channels, in_channels, kernel_size=1, bias=False)
        self.out_bn = norm_fn(in_channels)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        # Bypass if there are not enough points for BatchNorm to work
        if x.features.shape[0] <= 1:
            return x
            
        # Query, Key, Value feature extraction and normalization
        q_feat = self.q_bn(self.q_conv(x).features)
        k_feat = self.k_bn(self.k_conv(x).features)
        v_feat = self.v_bn(self.v_conv(x).features)

        # Attention scores with sigmoid gating
        attn_scores = torch.sigmoid(q_feat * k_feat)
        attn_out_feat = attn_scores * v_feat

        # Parallel branch
        parallel_feat = self.parallel_bn(self.parallel_conv(x).features)
        parallel_feat = self.relu(parallel_feat)

        # Combine features
        combined_feat = attn_out_feat + parallel_feat
        x_combined = replace_feature(x, combined_feat)

        # Output projection
        out_feat = self.out_bn(self.out_conv(x_combined).features)
        
        # Residual connection
        out_feat = self.relu(out_feat + x.features)
        out = replace_feature(x, out_feat)
        
        return out


class SparseStarBlock(spconv.SparseModule):
    def __init__(self, in_channels, mlp_ratio=3, norm_fn=None, indice_key=None):
        super().__init__()
        
        # Using k=3, regular conv as spconv does not support grouped convolutions
        self.dwconv_conv = spconv.SubMConv3d(in_channels, in_channels, 3, padding=1, bias=False, indice_key=indice_key)
        self.dwconv_bn = norm_fn(in_channels)
        
        hidden_dim = int(in_channels * mlp_ratio)
        self.f1 = spconv.SubMConv3d(in_channels, hidden_dim, 1, bias=False, indice_key=indice_key)
        self.f2 = spconv.SubMConv3d(in_channels, hidden_dim, 1, bias=False, indice_key=indice_key)
        
        self.g_conv = spconv.SubMConv3d(hidden_dim, in_channels, 1, bias=False, indice_key=indice_key)
        self.g_bn = norm_fn(in_channels)
        
        self.dwconv2 = spconv.SubMConv3d(in_channels, in_channels, 3, padding=1, bias=False, indice_key=indice_key)
        
        self.act = nn.ReLU6()
        self.drop_path = nn.Identity()

    def forward(self, x):
        identity = x
        
        out = self.dwconv_conv(x)
        out = replace_feature(out, self.dwconv_bn(out.features))
        
        x1 = self.f1(out)
        x2 = self.f2(out)
        
        features = self.act(x1.features) * x2.features
        out = replace_feature(out, features)
        
        out = self.g_conv(out)
        out = replace_feature(out, self.g_bn(out.features))
        
        out = self.dwconv2(out)
        
        out = replace_feature(out, identity.features + self.drop_path(out.features))
        return out


class VoxelBackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseStarBlock(32, norm_fn=norm_fn, indice_key='subm2'),
            SparseStarBlock(32, norm_fn=norm_fn, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseStarBlock(64, norm_fn=norm_fn, indice_key='subm3'),
            SparseStarBlock(64, norm_fn=norm_fn, indice_key='subm3'),
        )

        # Initialize the attention module if configured
        self.attn = None
        if self.model_cfg.get('ATTENTION', None):
            attention_cfg = self.model_cfg.ATTENTION
            self.attn = SparseDynamicParallelAttention(
                in_channels=64,
                num_heads=attention_cfg.NUM_HEADS,
                attn_ratio=attention_cfg.ATTN_RATIO
            )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            SparseStarBlock(64, norm_fn=norm_fn, indice_key='subm4'),
            SparseStarBlock(64, norm_fn=norm_fn, indice_key='subm4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 64
        }



    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        
        # Apply the attention module after conv3
        if self.attn is not None:
            x_conv3 = self.attn(x_conv3)
            
        x_conv4 = self.conv4(x_conv3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })

        return batch_dict


class VoxelResBackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        use_bias = self.model_cfg.get('USE_BIAS', None)
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(16, 16, bias=use_bias, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(16, 16, bias=use_bias, norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(32, 32, bias=use_bias, norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(32, 32, bias=use_bias, norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(64, 64, bias=use_bias, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(64, 64, bias=use_bias, norm_fn=norm_fn, indice_key='res3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(128, 128, bias=use_bias, norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(128, 128, bias=use_bias, norm_fn=norm_fn, indice_key='res4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(128, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 128
        }

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })

        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })
        
        return batch_dict