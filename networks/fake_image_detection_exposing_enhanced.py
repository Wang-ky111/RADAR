import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, resnet34
import numpy as np

class CrossDomainAttentionFusion(nn.Module):
    def __init__(self, pixel_dim=512, freq_dim=512, hidden_dim=256):
        super(CrossDomainAttentionFusion, self).__init__()
        self.pixel_dim = pixel_dim
        self.freq_dim = freq_dim
        self.pixel_to_freq_attention = nn.Sequential(
            nn.Linear(pixel_dim + freq_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, freq_dim),
            nn.Sigmoid()
        )
        self.freq_to_pixel_attention = nn.Sequential(
            nn.Linear(pixel_dim + freq_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, pixel_dim),
            nn.Sigmoid()
        )
        self.fusion_processor = nn.Sequential(
            nn.Linear(pixel_dim + freq_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1024, 512)
        )
    def forward(self, pixel_features, freq_features):
        batch_size = pixel_features.size(0)
        combined_features = torch.cat([pixel_features, freq_features], dim=1)
        pixel_attention_weights = self.pixel_to_freq_attention(combined_features)
        attended_freq_features = freq_features * pixel_attention_weights
        freq_attention_weights = self.freq_to_pixel_attention(combined_features)
        attended_pixel_features = pixel_features * freq_attention_weights
        fused_features = torch.cat([attended_pixel_features, attended_freq_features], dim=1)
        final_features = self.fusion_processor(fused_features)
        return final_features

class DualDomainAttentionFusionModule(nn.Module):
    def __init__(self, pixel_dim=512, freq_dim=512, output_dim=512):
        super(DualDomainAttentionFusionModule, self).__init__()
        self.pixel_dim = pixel_dim
        self.freq_dim = freq_dim
        self.output_dim = output_dim
        self.mix_block = nn.Sequential(
            nn.Linear(pixel_dim + freq_dim, pixel_dim + freq_dim),
            nn.ReLU(inplace=True),
            nn.Linear(pixel_dim + freq_dim, pixel_dim + freq_dim)
        )
        self.avg_pool3 = nn.AdaptiveAvgPool1d(1)
        self.avg_pool5 = nn.AdaptiveAvgPool1d(1)
        self.avg_pool7 = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.mlp = nn.Sequential(
            nn.Linear(4 * (pixel_dim + freq_dim), (pixel_dim + freq_dim) // 2),
            nn.ReLU(inplace=True),
            nn.Linear((pixel_dim + freq_dim) // 2, pixel_dim + freq_dim),
            nn.Sigmoid()
        )
        self.feature_processor = nn.Sequential(
            nn.Linear(pixel_dim + freq_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
    def forward(self, pixel_features, freq_features):
        batch_size = pixel_features.size(0)
        concat_feat = torch.cat([pixel_features, freq_features], dim=1)
        mix_feat = self.mix_block(concat_feat)
        feat_reshaped = mix_feat.unsqueeze(dim=2)
        pool3 = self.avg_pool3(feat_reshaped).squeeze(dim=2)
        pool5 = self.avg_pool5(feat_reshaped).squeeze(dim=2)
        pool7 = self.avg_pool7(feat_reshaped).squeeze(dim=2)
        pool_max = self.max_pool(feat_reshaped).squeeze(dim=2)
        pool_concat = torch.cat([pool3, pool5, pool7, pool_max], dim=1)
        channel_att = self.mlp(pool_concat)
        att_feat = mix_feat * channel_att
        final_feat = self.feature_processor(att_feat)
        return final_feat

class GatedFusion(nn.Module):
    def __init__(self, main_dim, lib_dim, output_dim):
        super(GatedFusion, self).__init__()
        self.main_dim = main_dim
        self.lib_dim = lib_dim
        self.gate = nn.Sequential(
            nn.Linear(main_dim + lib_dim, lib_dim),
            nn.ReLU(inplace=True),
            nn.Linear(lib_dim, lib_dim),
            nn.Sigmoid()
        )
        self.main_transformer = nn.Linear(main_dim, output_dim)
        self.lib_transformer = nn.Linear(lib_dim, output_dim)
    def forward(self, main_features, lib_features):
        combined = torch.cat([main_features, lib_features], dim=1)
        gate_weights = self.gate(combined)
        gated_lib_features = lib_features * gate_weights
        transformed_main = self.main_transformer(main_features)
        transformed_lib = self.lib_transformer(gated_lib_features)
        fused_features = transformed_main + transformed_lib
        return fused_features

class WeightedFeatureFusion(nn.Module):
    def __init__(self, num_libs, lib_dim, output_dim):
        super(WeightedFeatureFusion, self).__init__()
        self.num_libs = num_libs
        self.lib_dim = lib_dim
        self.attention_weights = nn.Parameter(torch.ones(num_libs) / num_libs)
        self.processor = nn.Sequential(
            nn.Linear(lib_dim * num_libs, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
    def forward(self, lib_features_list):
        weighted_features = []
        for i, features in enumerate(lib_features_list):
            weight = self.attention_weights[i]
            weighted_features.append(features * weight)
        concatenated = torch.cat(weighted_features, dim=1)
        fused_features = self.processor(concatenated)
        return fused_features

class LocalInformationBlock(nn.Module):
    def __init__(self, backbone_type='resnet34', pretrained=True, output_dim=512, input_channels=256):
        super(LocalInformationBlock, self).__init__()
        if backbone_type == 'resnet34':
            backbone = resnet34(pretrained=False)
            backbone.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            if pretrained:
                pretrained_dict = resnet34(pretrained=True).state_dict()
                model_dict = backbone.state_dict()
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in ['conv1.weight']}
                model_dict.update(pretrained_dict)
                backbone.load_state_dict(model_dict)
            self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])
            self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.feature_dim = 512
        else:
            raise ValueError(f"Unsupported backbone type: {backbone_type}")
        self.local_disentangle = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, output_dim)
        )
    def forward(self, x):
        features = self.feature_extractor(x)
        pooled = self.adaptive_pool(features)
        flattened = pooled.view(pooled.size(0), -1)
        local_features = self.local_disentangle(flattened)
        return local_features

class FrequencyDomainWithExposing(nn.Module):
    def __init__(self, num_libs=4, lib_output_dim=512, pretrained=True):
        super(FrequencyDomainWithExposing, self).__init__()
        self.num_libs = num_libs
        self.main_backbone = resnet50(pretrained=pretrained)
        self.main_backbone = nn.Sequential(*list(self.main_backbone.children())[:-2])
        self.main_adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.main_feature_dim = 2048
        self.intermediate_dim = 256
        self.channel_adapter = nn.Sequential(
            nn.Conv2d(2048, 1024, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, self.intermediate_dim, 1)
        )
        self.libs = nn.ModuleList([
            LocalInformationBlock('resnet34', pretrained, lib_output_dim, input_channels=self.intermediate_dim)
            for _ in range(num_libs)
        ])
        self.global_aggregation = WeightedFeatureFusion(
            num_libs=num_libs,
            lib_dim=lib_output_dim,
            output_dim=512
        )
        self.main_processor = nn.Sequential(
            nn.Linear(self.main_feature_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1024, 512)
        )
        self.feature_fusion = GatedFusion(
            main_dim=512,
            lib_dim=512,
            output_dim=512
        )
    def split_frequency_into_regions(self, x, grid_size=2):
        batch_size, channels, height, width = x.shape
        region_height = height // grid_size
        region_width = width // grid_size
        regions = []
        for i in range(grid_size):
            for j in range(grid_size):
                region = x[:, :,
                          i*region_height:(i+1)*region_height,
                          j*region_width:(j+1)*region_width]
                regions.append(region)
        return regions
    def forward(self, x):
        main_features = self.main_backbone(x)
        main_pooled = self.main_adaptive_pool(main_features)
        main_flattened = main_pooled.view(main_pooled.size(0), -1)
        main_processed = self.main_processor(main_flattened)
        adapted_features = self.channel_adapter(main_features)
        frequency_regions = self.split_frequency_into_regions(adapted_features, grid_size=2)
        lib_features = []
        for i, region in enumerate(frequency_regions[:self.num_libs]):
            lib_feat = self.libs[i](region)
            lib_features.append(lib_feat)
        global_features = self.global_aggregation(lib_features)
        final_frequency_features = self.feature_fusion(main_processed, global_features)
        intermediate_outputs = {
            'main_features': main_processed,
            'lib_features': lib_features,
            'global_features': global_features,
            'final_frequency_features': final_frequency_features,
            'adapted_features': adapted_features
        }
        return final_frequency_features, intermediate_outputs

class PixelDomainWithExposing(nn.Module):
    def __init__(self, num_libs=4, lib_output_dim=512, pretrained=True):
        super(PixelDomainWithExposing, self).__init__()
        self.num_libs = num_libs
        self.main_backbone = resnet50(pretrained=pretrained)
        self.main_backbone = nn.Sequential(*list(self.main_backbone.children())[:-2])
        self.main_adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.main_feature_dim = 2048
        self.libs = nn.ModuleList([
            LocalInformationBlock('resnet34', pretrained, lib_output_dim, input_channels=3)
            for _ in range(num_libs)
        ])
        self.global_aggregation = WeightedFeatureFusion(
            num_libs=num_libs,
            lib_dim=lib_output_dim,
            output_dim=512
        )
        self.feature_fusion = GatedFusion(
            main_dim=self.main_feature_dim,
            lib_dim=512,
            output_dim=512
        )
    def split_image_into_regions(self, x, grid_size=2):
        batch_size, channels, height, width = x.shape
        region_height = height // grid_size
        region_width = width // grid_size
        regions = []
        for i in range(grid_size):
            for j in range(grid_size):
                region = x[:, :,
                          i*region_height:(i+1)*region_height,
                          j*region_width:(j+1)*region_width]
                regions.append(region)
        return regions
    def forward(self, x):
        main_features = self.main_backbone(x)
        main_pooled = self.main_adaptive_pool(main_features)
        main_flattened = main_pooled.view(main_pooled.size(0), -1)
        image_regions = self.split_image_into_regions(x, grid_size=2)
        lib_features = []
        for i, region in enumerate(image_regions[:self.num_libs]):
            lib_feat = self.libs[i](region)
            lib_features.append(lib_feat)
        global_features = self.global_aggregation(lib_features)
        final_pixel_features = self.feature_fusion(main_flattened, global_features)
        intermediate_outputs = {
            'main_features': main_flattened,
            'lib_features': lib_features,
            'global_features': global_features,
            'final_pixel_features': final_pixel_features
        }
        return final_pixel_features, intermediate_outputs

class AdaptiveAFFAllBandFilter2D(nn.Module):
    def __init__(
        self,
        hidden_size=3,
        hard_thresholding_fraction=0.15,
        grid_size=14,
        hidden_size_factor=1,
        tau=1.0,
    ):
        super(AdaptiveAFFAllBandFilter2D, self).__init__()
        self.hidden_size = hidden_size
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.grid_size = grid_size
        self.hidden_size_factor = hidden_size_factor
        self.tau = max(float(tau), 1e-6)
        hidden_dim = max(int(hidden_size * hidden_size_factor), hidden_size)
        self.score_net = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_dim, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=1, bias=True)
        )
    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        batch_size, channels, height, width = x.shape
        x_fft = torch.fft.rfft2(x, dim=(2, 3), norm="ortho")
        origin_ffted = x_fft
        freq_h, freq_w = origin_ffted.shape[2], origin_ffted.shape[3]
        magnitude = torch.log1p(torch.abs(origin_ffted))
        pooled = F.adaptive_avg_pool2d(magnitude, (self.grid_size, self.grid_size))
        scores = self.score_net(pooled)
        soft_keep_mask = torch.sigmoid(scores / self.tau)
        flat_scores = scores.view(batch_size, -1)
        total_regions = flat_scores.size(1)
        num_drop = int(round(total_regions * self.hard_thresholding_fraction))
        num_drop = max(0, min(num_drop, total_regions))
        hard_keep_mask = torch.ones_like(flat_scores, dtype=x.dtype)
        if num_drop > 0:
            drop_indices = torch.topk(flat_scores, k=num_drop, dim=1, largest=False).indices
            hard_keep_mask.scatter_(1, drop_indices, 0.0)
        hard_keep_mask = hard_keep_mask.view(batch_size, 1, self.grid_size, self.grid_size)
        if self.training:
            keep_mask = hard_keep_mask + soft_keep_mask - soft_keep_mask.detach()
        else:
            keep_mask = hard_keep_mask
        keep_mask = F.interpolate(keep_mask, size=(freq_h, freq_w), mode='nearest')
        keep_mask = keep_mask.expand(-1, channels, -1, -1)
        x_fft = origin_ffted * keep_mask.to(origin_ffted.dtype)
        filtered_x = torch.fft.irfft2(x_fft, s=(height, width), dim=(2, 3), norm="ortho")
        filtered_x = filtered_x.type(dtype)
        mask_info = {
            'region_scores': scores.detach(),
            'hard_keep_mask': hard_keep_mask.detach(),
            'drop_ratio': (1.0 - hard_keep_mask.mean(dim=(1, 2, 3))).detach(),
            'freq_keep_mask': keep_mask[:, :1].detach()
        }
        return filtered_x, mask_info

class MultiTaskLoss(nn.Module):
    def __init__(self, num_tasks=6):
        super(MultiTaskLoss, self).__init__()
        self.num_tasks = num_tasks
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
    def forward(self, losses):
        total_loss = 0
        precision_list = []
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total_loss += precision * loss + self.log_vars[i]
            precision_list.append(precision)
        return total_loss, precision_list

class FakeImageDetectionExposingEnhanced(nn.Module):
    def __init__(
        self,
        num_classes=2,
        pretrained=True,
        use_ddafm=True,
        use_adaptive_mask=False,
        adaptive_mask_ratio=0.15,
        adaptive_mask_grid=14,
        adaptive_mask_hidden_factor=1,
        adaptive_mask_tau=1.0,
    ):
        super(FakeImageDetectionExposingEnhanced, self).__init__()
        self.use_ddafm = use_ddafm
        self.use_adaptive_mask = use_adaptive_mask
        self.pixel_domain = PixelDomainWithExposing(pretrained=pretrained)
        self.frequency_domain = FrequencyDomainWithExposing(pretrained=pretrained)
        if self.use_ddafm:
            self.cross_domain_fusion = DualDomainAttentionFusionModule(
                pixel_dim=512,
                freq_dim=512,
                output_dim=512
            )
        else:
            self.cross_domain_fusion = CrossDomainAttentionFusion(
                pixel_dim=512,
                freq_dim=512,
                hidden_dim=256
            )
        if self.use_adaptive_mask:
            self.adaptive_masker = AdaptiveAFFAllBandFilter2D(
                hidden_size=3,
                hard_thresholding_fraction=adaptive_mask_ratio,
                grid_size=adaptive_mask_grid,
                hidden_size_factor=adaptive_mask_hidden_factor,
                tau=adaptive_mask_tau,
            )
        else:
            self.adaptive_masker = None
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        self.multi_task_loss = MultiTaskLoss(num_tasks=6)
        self.aux_classifiers = nn.ModuleList([
            nn.Linear(512, num_classes) for _ in range(8)
        ])
    def compute_frequency_spectrum(self, x):
        x_fft = torch.fft.rfft2(x, dim=(2, 3), norm="ortho")
        magnitude = torch.abs(x_fft)
        log_magnitude = torch.log1p(magnitude)
        spectrum = (log_magnitude - log_magnitude.min()) / (log_magnitude.max() - log_magnitude.min() + 1e-8)
        return spectrum
    def forward(self, pixel_input, freq_input):
        adaptive_mask_info = None
        if self.use_adaptive_mask and self.adaptive_masker is not None:
            shared_input = pixel_input
            masked_input, adaptive_mask_info = self.adaptive_masker(shared_input)
            pixel_input = masked_input
            freq_input = masked_input
        freq_spectrum = self.compute_frequency_spectrum(freq_input)
        pixel_features, pixel_intermediate = self.pixel_domain(pixel_input)
        freq_features, freq_intermediate = self.frequency_domain(freq_spectrum)
        combined_features = self.cross_domain_fusion(pixel_features, freq_features)
        output = self.classifier(combined_features)
        aux_outputs = []
        pixel_lib_features = pixel_intermediate['lib_features']
        for i, lib_feat in enumerate(pixel_lib_features):
            aux_output = self.aux_classifiers[i](lib_feat)
            aux_outputs.append(aux_output)
        freq_lib_features = freq_intermediate['lib_features']
        for i, lib_feat in enumerate(freq_lib_features):
            aux_output = self.aux_classifiers[i+4](lib_feat)
            aux_outputs.append(aux_output)
        return {
            'main_output': output,
            'aux_outputs': aux_outputs,
            'pixel_features': pixel_features,
            'freq_features': freq_features,
            'pixel_intermediate': pixel_intermediate,
            'freq_intermediate': freq_intermediate,
            'adaptive_mask_info': adaptive_mask_info
        }
    def compute_loss(self, outputs, labels):
        main_output = outputs['main_output']
        aux_outputs = outputs['aux_outputs']
        main_loss = F.cross_entropy(main_output, labels)
        pixel_local_losses = []
        for i, aux_output in enumerate(aux_outputs[:4]):
            loss = F.cross_entropy(aux_output, labels)
            pixel_local_losses.append(loss)
        pixel_local_loss = sum(pixel_local_losses) / len(pixel_local_losses)
        freq_local_losses = []
        for i, aux_output in enumerate(aux_outputs[4:8]):
            loss = F.cross_entropy(aux_output, labels)
            freq_local_losses.append(loss)
        freq_local_loss = sum(freq_local_losses) / len(freq_local_losses)
        pixel_lib_features = outputs['pixel_intermediate']['lib_features']
        freq_lib_features = outputs['freq_intermediate']['lib_features']
        pixel_diversity_loss = self.compute_diversity_loss(pixel_lib_features)
        freq_diversity_loss = self.compute_diversity_loss(freq_lib_features)
        cross_domain_loss = self.compute_cross_domain_loss(
            outputs['pixel_features'],
            outputs['freq_features']
        )
        losses = [
            main_loss,
            pixel_local_loss,
            freq_local_loss,
            pixel_diversity_loss,
            freq_diversity_loss,
            cross_domain_loss
        ]
        total_loss, precision_list = self.multi_task_loss(losses)
        loss_dict = {
            'total_loss': total_loss,
            'main_loss': main_loss,
            'pixel_local_loss': pixel_local_loss,
            'freq_local_loss': freq_local_loss,
            'pixel_diversity_loss': pixel_diversity_loss,
            'freq_diversity_loss': freq_diversity_loss,
            'cross_domain_loss': cross_domain_loss,
            'precisions': precision_list
        }
        return total_loss, loss_dict
    def compute_diversity_loss(self, lib_features):
        if len(lib_features) < 2:
            return torch.tensor(0.0).to(lib_features[0].device)
        diversity_loss = 0
        count = 0
        for i in range(len(lib_features)):
            for j in range(i+1, len(lib_features)):
                similarity = F.cosine_similarity(lib_features[i], lib_features[j])
                diversity_loss += torch.mean(similarity)
                count += 1
        return diversity_loss / count if count > 0 else torch.tensor(0.0).to(lib_features[0].device)
    def compute_cross_domain_loss(self, pixel_features, freq_features):
        similarity = F.cosine_similarity(pixel_features, freq_features)
        return torch.mean(1 - similarity)
