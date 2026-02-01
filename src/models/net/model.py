"""
Reorganized SSL-based Deepfake Detection Models

This module contains various SSL (Self-Supervised Learning) based models for deepfake detection,
organized with proper inheritance hierarchy and reduced code duplication.

Terminology note:
- Position Loss: 8-label head (segment positional classification)
- Binary Loss: 2-label head (binary classification)
- Transition Loss + Binary Loss: dual 2-label heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

from src.models.modules.attention import SelfWeightedPooling
from src.models.modules.transformer import PositionalEncoding
from src.models.net.ssl_encoder import get_ssl_encoder
from src.models.modules.conformer import ConformerEncoder


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
        
        if self.pool == 'att':
            self.att_pool = SelfWeightedPooling(
                self.hid_dim,
                num_head=num_head,
                mean_only=True
            )

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


class BaseSSLModel(nn.Module):
    """
    Base class for SSL-based models with common functionality.
    """
    
    def __init__(
        self,
        ssl_encoder: str = "xlsr",
        ckpt: Optional[str] = None,
        mode: str = "s3prl",
        hid_dim: int = 1024,
        resolution_train: float = 0.16,
        resolution_test: float = 0.16,
        pool: str = "att",
        pool_head_num: int = 1,
        num_outputs: int = 2,
        create_out_layer: bool = True,
        **kwargs
    ):
        super(BaseSSLModel, self).__init__()
        
        # SSL Encoder
        self.ssl_encoder = get_ssl_encoder(
            ssl_encoder, mode=mode, ckpt=ckpt, hid_dim=hid_dim
        )
        self.hid_dim = self.ssl_encoder.out_dim
        self.pool_head_num = pool_head_num
        
        # Pooling head
        self.pool_head = PoolHead(
            hid_dim=hid_dim,
            pool=pool,
            resolution_train=resolution_train,
            resolution_test=resolution_test,
            num_head=pool_head_num
        )
        
        # Weighted mode support
        if mode == "s3prl_weighted":
            self.weight_layer = nn.Parameter(torch.ones(25) / 25)
        
        # Output layer (optional, subclasses may define their own)
        if create_out_layer:
            emb_dim = self.hid_dim * self.pool_head_num
            self.out_layer = nn.Sequential(
                nn.SELU(),
                nn.Linear(emb_dim, 256),
                nn.SELU(),
                nn.Linear(256, num_outputs)
            )

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract SSL features with optional weighted combination."""
        x = F.pad(x, (0, 256), mode='constant', value=0)
        out = self.ssl_encoder.extract_feat(x)
        
        if hasattr(self, "weight_layer"):
            out = torch.stack(out, dim=0)
            out = out * self.weight_layer.view(25, 1, 1, 1)
            out = out.sum(dim=0)
        
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Base forward pass."""
        out = self._extract_features(x)
        out = self.pool_head(out)
        out = self.out_layer(out)
        return out


class BaseSSLSeqModel(BaseSSLModel):
    """
    Base class for SSL models with sequence modeling capabilities.
    """
    
    def __init__(
        self,
        seq_model: str = "lstm",
        num_layers: int = 2,
        num_heads: int = 4,
        **kwargs
    ):
        super(BaseSSLSeqModel, self).__init__(**kwargs)
        
        # Sequence model
        if seq_model == "lstm":
            self.seq_model = nn.LSTM(
                input_size=self.hid_dim,
                hidden_size=self.hid_dim // 2,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True
            )
        elif seq_model == "rnn":
            self.seq_model = nn.GRU(
                input_size=self.hid_dim,
                hidden_size=self.hid_dim // 2,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True
            )
        elif seq_model == "tf":
            self.seq_model = nn.Sequential(
                PositionalEncoding(self.hid_dim),
                nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=self.hid_dim,
                        nhead=num_heads,
                        dim_feedforward=self.hid_dim // 2,
                        dropout=0.1
                    ),
                    num_layers=num_layers
                )
            )
        elif seq_model == "cf":  # Conformer
            self.seq_model = ConformerEncoder(
                attention_in=self.hid_dim,
                ffn_hidden=self.hid_dim,
                num_head=num_heads,
                num_layer=num_layers,
            )
        else:
            raise ValueError(f"Unsupported seq_model: {seq_model}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with sequence modeling."""
        out = self._extract_features(x)
        out = self.seq_model(out)
        if isinstance(out, tuple):
            out = out[0]
        out = self.pool_head(out)
        out = self.out_layer(out)
        return out


# ============================================================================
# Basic SSL Models
# ============================================================================

class SSL_Pool(BaseSSLModel):
    """Basic SSL model with Binary loss head (2 labels)."""
    pass


# ============================================================================
# Sequence Models
# ============================================================================

class SSLSeq(BaseSSLSeqModel):
    """SSL sequence model with Binary loss head (2 labels)."""
    pass


class SSLSeqBin2(BaseSSLSeqModel):
    """SSL sequence model with Transition Loss + Binary Loss heads (2 labels + 2 labels)."""
    
    def __init__(self, **kwargs):
        super(SSLSeqBin2, self).__init__(create_out_layer=False, **kwargs)
        # Transition Loss head (2 labels)
        self.out_layer1 = nn.Sequential(
            nn.SELU(),
            nn.Linear(self.hid_dim, 256),
            nn.SELU(),
            nn.Linear(256, 2)
        )
        # Binary Loss head (2 labels)
        self.out_layer2 = nn.Sequential(
            nn.SELU(),
            nn.Linear(self.hid_dim, 256),
            nn.SELU(),
            nn.Linear(256, 2)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with dual 2-label outputs."""
        out = self._extract_features(x)
        out = self.seq_model(out)
        if isinstance(out, tuple):
            out = out[0]
        out = self.pool_head(out)
        out1 = self.out_layer1(out)
        out2 = self.out_layer2(out)
        return out1, out2


class SSLSeq8Bin(BaseSSLSeqModel):
    """SSL sequence model (Position + Binary) with learnable layer weighting."""
    
    def __init__(self, **kwargs):
        super(SSLSeq8Bin, self).__init__(create_out_layer=False, **kwargs)
        # Position Loss head (8 labels)
        self.out_layer1 = nn.Sequential(
            nn.SELU(),
            nn.Linear(self.hid_dim, 256),
            nn.SELU(),
            nn.Linear(256, 8)
        )
        # Binary Loss head (2 labels)
        self.out_layer2 = nn.Sequential(
            nn.SELU(),
            nn.Linear(self.hid_dim, 256),
            nn.SELU(),
            nn.Linear(256, 2)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with learnable weighted combination."""
        out = self._extract_features(x)
        out = self.seq_model(out)
        if isinstance(out, tuple):
            out = out[0]
        out = self.pool_head(out)
        out1 = self.out_layer1(out)
        out2 = self.out_layer2(out)
        return out1, out2


# ============================================================================
# Model Registry and Factory Functions
# ============================================================================

MODEL_REGISTRY = {
    # Basic SSL Models
    'ssl_pool': SSL_Pool,
    
    # Sequence Models
    'ssl_seq': SSLSeq,
    'ssl_seq_8labels_2loss': SSLSeq8Bin,
    'ssl_seq_2labels_2loss': SSLSeqBin2,
}

# Backward compatibility aliases
SSL_SeqModel = SSLSeq
SSL_SeqModel_2Labels_2Loss = SSLSeqBin2
SSL_SeqModel_8Labels_2Loss = SSLSeq8Bin


def create_model(model_name: str, **kwargs) -> nn.Module:
    """
    Factory function to create models by name.
    
    Args:
        model_name: Name of the model to create
        **kwargs: Additional arguments to pass to the model constructor
        
    Returns:
        Instantiated model
        
    Raises:
        ValueError: If model_name is not in the registry
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(MODEL_REGISTRY.keys())}")
    
    return MODEL_REGISTRY[model_name](**kwargs)


def list_models() -> List[str]:
    """Return list of available model names."""
    return list(MODEL_REGISTRY.keys())


# ============================================================================
# Testing and Example Usage
# ============================================================================

if __name__ == '__main__':
    # Example: Test model creation
    test_models = ['ssl_pool', 'ssl_seq', 'ssl_seq_8labels_2loss', 'ssl_seq_2labels_2loss']
    
    for model_name in test_models:
        print(f"\nTesting {model_name}:")
        try:
            model = create_model(model_name)
            print(f"✓ Successfully created {model_name}")
            print(f"  Model type: {type(model).__name__}")
        except Exception as e:
            print(f"✗ Error with {model_name}: {e}")
