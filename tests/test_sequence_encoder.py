
import torch

from alphagenome.sequence_encoder import RMSBatchNorm1D


class TestRMSBatchNorm1D:
    def setup_method(self):
        self.batch_size = 4
        self.seq_len = 100
        self.num_features = 128
        self.eps = 1e-6
        self.momentum = 0.9
    
    def test_3d_input(self):
        rms_norm = RMSBatchNorm1D(self.num_features)
        x = torch.randn(self.batch_size, self.seq_len, self.num_features)
        output = rms_norm(x)

        assert output.shape == x.shape
