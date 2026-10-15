import torch
import torch.nn as nn
from torch.autograd import Function


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        # store alpha for use in backward pass
        ctx.alpha = alpha
        # forward pass is identity, no change to input
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # reverse gradient by multiplying by negative alpha
        # this is what makes the feature extractor adversarial
        return grad_output.neg() * ctx.alpha, None


class GradientReversalLayer(nn.Module):
    def __init__(self, alpha=1.0):
        super(GradientReversalLayer, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)


class DomainDiscriminator(nn.Module):
    def __init__(self, embed_dim=128, dropout_rate=0.5):
        super(DomainDiscriminator, self).__init__()

        self.grl = GradientReversalLayer()

        self.network = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1),
        )

    def forward(self, x, alpha=1.0):
        # update alpha for scheduled training
        self.grl.alpha = alpha
        x = self.grl(x)
        return self.network(x)


class ConditionalDomainDiscriminator(nn.Module):
    def __init__(self, embed_dim=128, n_classes=2, dropout_rate=0.5):
        super(ConditionalDomainDiscriminator, self).__init__()

        # input dimension is embed_dim * n_classes due to multilinear map
        # 128 * 2 = 256 in our case
        input_dim = embed_dim * n_classes

        self.grl = GradientReversalLayer()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1),
        )

    def forward(self, x, alpha=1.0):
        # x is the multilinear map output, shape (batch, embed_dim * n_classes)
        self.grl.alpha = alpha
        x = self.grl(x)
        return self.network(x)


class Classifier(nn.Module):
    def __init__(self, embed_dim=128, n_classes=2, dropout_rate=0.5):
        super(Classifier, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.network(x)


class EEGNet(nn.Module):
    def __init__(
        self,
        n_channels=8,
        n_timepoints=480,
        embed_dim=128,
        dropout_rate=0.5,
        f1=8,
        d=2,
        f2=16,
    ):
        super(EEGNet, self).__init__()

        # block 1: temporal convolution which learns frequency specific patterns across time
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, f1, kernel_size=(1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(f1),
        )

        # block 2: depthwise spatial convolution
        # learns spatial patterns across channels
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(f1, f1 * d, kernel_size=(n_channels, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f1 * d),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout_rate),
        )

        # block 3: separable convolution which learns temporal summaries of each feature map
        self.separable_conv = nn.Sequential(
            nn.Conv2d(
                f1 * d,
                f1 * d,
                kernel_size=(1, 16),
                padding=(0, 8),
                groups=f1 * d,
                bias=False,
            ),
            nn.Conv2d(f1 * d, f2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout_rate),
        )

        # calculate flattened size after convolutions
        self._calculate_flat_size(n_channels, n_timepoints, f1, d, f2)

        # projection to embedding dimension
        self.projection = nn.Linear(self.flat_size, embed_dim)

    def _calculate_flat_size(self, n_channels, n_timepoints, f1, d, f2):
        # a dummy forward pass to get the output size
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_timepoints)
            x = self.temporal_conv(dummy)
            x = self.depthwise_conv(x)
            x = self.separable_conv(x)
            self.flat_size = x.view(1, -1).shape[1]

    def forward(self, x):
        # x shape: (batch, 1, n_channels, n_timepoints)
        x = self.temporal_conv(x)
        x = self.depthwise_conv(x)
        x = self.separable_conv(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.projection(x)
        return x


class InceptionBranch(nn.Module):
    """
    processes one group of channels (left, right, or midline hemisphere)
    inception-style temporal convolution captures multiple time scales
    followed by spatial convolution collapsing the channel dimension
    adapted from EEGSym concept, not identical to original paper
    """

    def __init__(self, n_branch_channels, n_timepoints, n_filters=8, dropout_rate=0.4):
        super(InceptionBranch, self).__init__()

        self.n_branch_channels = n_branch_channels

        # three parallel temporal kernels, different time scales
        # roughly 500ms, 250ms, 125ms windows at 160Hz
        self.temporal_64 = nn.Sequential(
            nn.Conv2d(1, n_filters, kernel_size=(1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(n_filters),
        )
        self.temporal_32 = nn.Sequential(
            nn.Conv2d(1, n_filters, kernel_size=(1, 32), padding=(0, 16), bias=False),
            nn.BatchNorm2d(n_filters),
        )
        self.temporal_16 = nn.Sequential(
            nn.Conv2d(1, n_filters, kernel_size=(1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(n_filters),
        )

        self.elu = nn.ELU()
        self.dropout1 = nn.Dropout(dropout_rate)

        # spatial convolution collapses the channel dimension to 1
        # only valid if n_branch_channels > 1
        combined_filters = n_filters * 3
        if n_branch_channels > 1:
            self.spatial_conv = nn.Sequential(
                nn.Conv2d(
                    combined_filters,
                    combined_filters,
                    kernel_size=(n_branch_channels, 1),
                    groups=combined_filters,
                    bias=False,
                ),
                nn.BatchNorm2d(combined_filters),
            )
        else:
            # single channel branch, no spatial collapse needed
            self.spatial_conv = None

        self.pool = nn.AvgPool2d(kernel_size=(1, 4))
        self.dropout2 = nn.Dropout(dropout_rate)

        self.out_filters = combined_filters

    def forward(self, x):
        # x shape: (batch, 1, n_branch_channels, n_timepoints)
        t64 = self.temporal_64(x)
        t32 = self.temporal_32(x)
        t16 = self.temporal_16(x)

        # concatenate along filter dimension
        combined = torch.cat([t64, t32, t16], dim=1)
        combined = self.elu(combined)
        combined = self.dropout1(combined)

        if self.spatial_conv is not None:
            combined = self.spatial_conv(combined)
            combined = self.elu(combined)

        out = self.pool(combined)
        out = self.dropout2(out)
        return out


class EEGSymAdapted(nn.Module):
    """
    adapted EEGSym feature extractor
    inspired by Perez-Velasco et al. 2022 EEGSym architecture
    changed for our 8 channel asymmetric hemisphere setup
    left: C3, FC3, CP3 (channels 0, 3, 5)
    right: C4, CP4 (channels 2, 7)
    midline: Cz, FCz, CPz (channels 1, 4, 6)
    """

    def __init__(
        self,
        n_channels=8,
        n_timepoints=480,
        embed_dim=128,
        dropout_rate=0.4,
        n_filters=8,
    ):
        super(EEGSymAdapted, self).__init__()

        # channel indices based on our fixed channel order:
        # C3, Cz, C4, FC3, FCz, CP3, CPz, CP4
        self.left_idx = [0, 3, 5]  # C3, FC3, CP3
        self.right_idx = [2, 7]  # C4, CP4
        self.mid_idx = [1, 4, 6]  # Cz, FCz, CPz

        self.left_branch = InceptionBranch(
            n_branch_channels=3,
            n_timepoints=n_timepoints,
            n_filters=n_filters,
            dropout_rate=dropout_rate,
        )
        self.right_branch = InceptionBranch(
            n_branch_channels=2,
            n_timepoints=n_timepoints,
            n_filters=n_filters,
            dropout_rate=dropout_rate,
        )
        self.mid_branch = InceptionBranch(
            n_branch_channels=3,
            n_timepoints=n_timepoints,
            n_filters=n_filters,
            dropout_rate=dropout_rate,
        )

        branch_out_filters = n_filters * 3  # 24
        merged_filters = branch_out_filters * 3  # 72, three branches concatenated

        # channel merging stage
        # combines left, right, midline information
        self.merge_conv = nn.Sequential(
            nn.Conv2d(
                merged_filters,
                merged_filters,
                kernel_size=(1, 8),
                padding=(0, 4),
                groups=merged_filters,
                bias=False,
            ),
            nn.BatchNorm2d(merged_filters),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout_rate),
        )

        # calculate flattened size dynamically
        self._calculate_flat_size(n_channels, n_timepoints)

        self.projection = nn.Linear(self.flat_size, embed_dim)

    def _calculate_flat_size(self, n_channels, n_timepoints):
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_timepoints)
            out = self._forward_features(dummy)
            self.flat_size = out.view(1, -1).shape[1]

    def _forward_features(self, x):
        # x shape: (batch, 1, n_channels, n_timepoints)
        x_left = x[:, :, self.left_idx, :]
        x_right = x[:, :, self.right_idx, :]
        x_mid = x[:, :, self.mid_idx, :]

        out_left = self.left_branch(x_left)
        out_right = self.right_branch(x_right)
        out_mid = self.mid_branch(x_mid)

        # concatenate along filter dimension
        # all three have spatial dim 1, same temporal dim
        merged = torch.cat([out_left, out_right, out_mid], dim=1)
        merged = self.merge_conv(merged)
        return merged

    def forward(self, x):
        out = self._forward_features(x)
        out = out.view(out.size(0), -1)
        out = self.projection(out)
        return out


class DANN(nn.Module):
    def __init__(
        self,
        n_channels=8,
        n_timepoints=480,
        n_classes=2,
        embed_dim=128,
        dropout_rate=0.5,
        use_conditional_discriminator=False,
        feature_extractor_type="eegnet",
    ):
        super(DANN, self).__init__()

        if feature_extractor_type == "eegnet":
            self.feature_extractor = EEGNet(
                n_channels=n_channels,
                n_timepoints=n_timepoints,
                embed_dim=embed_dim,
                dropout_rate=dropout_rate,
            )
        elif feature_extractor_type == "eegsym":
            self.feature_extractor = EEGSymAdapted(
                n_channels=n_channels,
                n_timepoints=n_timepoints,
                embed_dim=embed_dim,
                dropout_rate=dropout_rate,
            )
        else:
            raise ValueError(
                f"Unknown feature_extractor_type: {feature_extractor_type}"
            )

        self.classifier = Classifier(
            embed_dim=embed_dim, n_classes=n_classes, dropout_rate=dropout_rate
        )

        if use_conditional_discriminator:
            self.domain_discriminator = ConditionalDomainDiscriminator(
                embed_dim=embed_dim, n_classes=n_classes, dropout_rate=dropout_rate
            )
        else:
            self.domain_discriminator = DomainDiscriminator(
                embed_dim=embed_dim, dropout_rate=dropout_rate
            )

        self.use_conditional_discriminator = use_conditional_discriminator

    def forward(self, x, alpha=1.0):
        # extract features
        features = self.feature_extractor(x)

        # classify
        class_output = self.classifier(features)

        # domain discrimination
        if self.use_conditional_discriminator:
            # get softmax probabilities from classifier output
            class_probs = torch.softmax(class_output, dim=1)

            # multilinear map: outer product of features and class probabilities
            # features shape: (batch, embed_dim)
            # class_probs shape: (batch, n_classes)
            # result shape: (batch, embed_dim * n_classes)
            multilinear = torch.bmm(
                features.unsqueeze(2), class_probs.unsqueeze(1)
            ).view(features.size(0), -1)

            domain_output = self.domain_discriminator(multilinear, alpha)
        else:
            domain_output = self.domain_discriminator(features, alpha)

        return class_output, domain_output, features
