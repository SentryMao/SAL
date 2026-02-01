import s3prl.hub as hub
import torch.nn as nn
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

def get_ssl_encoder(name="xlsr", mode="s3prl", ckpt=None, hid_dim=1024):
    if name == "xlsr":
        return Wav2vec2(mode=mode, ckpt=ckpt, hid_dim=hid_dim)
    elif name == "large":
        return WavLM(mode=mode, ckpt=ckpt, hid_dim=hid_dim)
    else:
        raise ValueError(f"Unsupported SSL encoder: {name}")


class Wav2vec2(nn.Module):
    def __init__(self, mode="s3prl", ckpt=None, hid_dim=1024):
        super(Wav2vec2, self).__init__()
        self.mode = mode
        if self.mode == "s3prl":
            if ckpt is None:
                raise ValueError("Please provide a valid checkpoint.")
            self.ssl_encoder = hub.wav2vec2_local(ckpt=ckpt, fairseq=True)
            self.out_dim = hid_dim
        elif self.mode == "s3prl_weighted":
            if ckpt is None:
                raise ValueError("Please provide a valid checkpoint.")
            self.ssl_encoder = hub.wav2vec2_local(ckpt=ckpt, fairseq=True)
            self.out_dim = hid_dim
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def extract_feat(self, input_data):
        """
        Extract features using Hugging Face Wav2Vec2 model.
        Args:
            input_data (torch.Tensor): Audio input tensor of shape (batch_size, seq_len).
                                       The input is expected to be raw audio waveforms.
        Returns:
            torch.Tensor: Extracted features of shape (batch_size, seq_len, hidden_size).
        """
        if input_data.ndim == 3:  # Remove channel dimension if present
            input_data = input_data[:, :, 0]

        if self.mode == "s3prl":
            return self.ssl_encoder(input_data)["hidden_states"][-1]
        elif self.mode == "s3prl_weighted":
            return self.ssl_encoder(input_data)["hidden_states"]
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

class WavLM(nn.Module):
    def __init__(self, mode="s3prl", ckpt=None, hid_dim=1024):
        super(WavLM, self).__init__()
        self.mode = mode
        if self.mode == "s3prl":
            if ckpt is None:
                raise ValueError("Please provide a valid checkpoint.")
            self.ssl_encoder = hub.wavlm_local(ckpt=ckpt)
            self.out_dim = hid_dim
        elif self.mode == "s3prl_weighted":
            if ckpt is None:
                raise ValueError("Please provide a valid checkpoint.")
            self.ssl_encoder = hub.wavlm_local(ckpt=ckpt)
            self.out_dim = hid_dim
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def extract_feat(self, input_data):
        """
        Extract features using Hugging Face Wav2Vec2 model.
        Args:
            input_data (torch.Tensor): Audio input tensor of shape (batch_size, seq_len).
                                       The input is expected to be raw audio waveforms.
        Returns:
            torch.Tensor: Extracted features of shape (batch_size, seq_len, hidden_size).
        """
        if input_data.ndim == 3:  # Remove channel dimension if present
            input_data = input_data[:, :, 0]

        if self.mode == "s3prl":
            return self.ssl_encoder(input_data)["hidden_states"][-1]
        elif self.mode == "s3prl_weighted":
            return self.ssl_encoder(input_data)["hidden_states"]
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")
