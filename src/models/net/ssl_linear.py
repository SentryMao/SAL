import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.net.ssl_encoder import get_ssl_encoder

class PoolHead(nn.Module):
    """
    Pooling head for temporal feature aggregation.
    
    Supports both average pooling and attention-based pooling.
    """
    
    def __init__(
        self,
        hid_dim: int,
        pool: str = 'att',
        resolution_train: float = 0.16,
        resolution_test: float = 0.16,
        num_head: int = 1,
        **kwargs
    ):
        super(PoolHead, self).__init__()
        self.hid_dim = hid_dim
        self.scale_train = int(resolution_train // 0.02)
        self.scale_test = int(resolution_test // 0.02)
        
        assert pool in ['avg', 'att'], f"Unsupported pool: {pool}"
        self.pool = pool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through pooling head."""
        scale = self.scale_train if self.training else self.scale_test

        if self.pool == 'avg':
            x = x.transpose(1, 2)
            x = F.avg_pool1d(x, kernel_size=scale, stride=scale)
            x = x.transpose(1, 2)
        else:
            # attention pooling
            b, f, d = x.shape
            x = x.reshape(-1, scale, d)
            x = self.att_pool(x)
            x = x.reshape(b, -1, d * self.att_pool.num_head)
        return x

class SSL_Linear(nn.Module):
    def __init__(
            self,
            ssl_name="xlsr",
            mode="s3prl",
            ckpt=None,
            ssl_frozen=False,
            **kwargs
    ):
        super(SSL_Linear, self).__init__()
        self.ssl_encoder = get_ssl_encoder(
            name=ssl_name,
            mode=mode,
            ckpt=ckpt,
            **kwargs
        )
        self.pool_head = PoolHead(
            hid_dim=self.ssl_encoder.out_dim,
            pool='avg',
            resolution_train=0.02,
            resolution_test=0.02,
            num_head=1
        )
        if ssl_frozen:
            for param in self.ssl_encoder.parameters():
                param.requires_grad = False
        self.projection = nn.ModuleList([
            nn.Linear(self.ssl_encoder.out_dim, 256),
            nn.BatchNorm1d(256),
            # nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            # nn.LayerNorm(128),
            nn.ReLU(),
        ])
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        x = F.pad(x, (0, 256), mode='constant', value=0)
        # print(x.shape)
        x_ssl_feat = self.ssl_encoder(x)
        # print(x_ssl_feat.shape)
        # x = x_ssl_feat.mean(dim=1)  # mean pooling, (bs, dim)
        x = self.pool_head(x_ssl_feat)  # avg pooling, (bs, dim)
        # print(x.shape)
        # last_hidden = self.projection(x)
        # (层 0) Linear: [B, 13, 256] -> [B, 13, 256]
        last_hidden = self.projection[0](x)

        # (层 1) BatchNorm1d:
        # (N, L, C) -> (N, C, L)
        last_hidden = last_hidden.permute(0, 2, 1)  # 形状变为 [B, 256, 13]
        last_hidden = self.projection[1](last_hidden)
        # (N, C, L) -> (N, L, C)
        last_hidden = last_hidden.permute(0, 2, 1)  # 形状变回 [B, 13, 256]

        # (层 2) ReLU
        last_hidden = self.projection[2](last_hidden)

        # (层 3) Linear: [B, 13, 256] -> [B, 13, 128]
        last_hidden = self.projection[3](last_hidden)

        # (层 4) BatchNorm1d:
        # (N, L, C) -> (N, C, L)
        last_hidden = last_hidden.permute(0, 2, 1)  # 形状变为 [B, 128, 13]
        last_hidden = self.projection[4](last_hidden)
        # (N, C, L) -> (N, L, C)
        last_hidden = last_hidden.permute(0, 2, 1)  # 形状变回 [B, 13, 128]

        # (层 5) ReLU
        last_hidden = self.projection[5](last_hidden)
        # print(last_hidden.shape)
        output = self.classifier(last_hidden)
        # print(output.shape)
        # return output, last_hidden
        return output

    def forward_ssl(self, x):
        outputs = self.ssl_encoder.forward_ssl(x)
        embedding = outputs['embedding'].mean(dim=1)  # mean pooling, (bs, dim)
        embedding = self.projection(embedding)
        prediction = self.classifier(embedding)
        return {
            'prediction': prediction,
            'embedding': embedding,
            'logits': outputs['logits'],
            'targets': outputs['targets'],
            'features_pen': outputs['features_pen'],
            'diversity_loss': outputs['diversity_loss']
        }
