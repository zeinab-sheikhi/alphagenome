
import torch

from alphagenome.sequence_encoder import RMSBatchNorm1D, StandardizedConv1D


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


class TestStandardizedConv1D:
    def setup_method(self):
        self.batch_size = 4
        self.seq_len = 100
        self.in_channels = 64
        self.out_channels = 128
        self.kernel_size = 5
        self.eps = 1e-6
    
    def test_basic_forward(self):
        conv = StandardizedConv1D(
            in_channels=self.in_channels, 
            out_channels=self.out_channels, 
            kernel_size=self.kernel_size,
        )

        x = torch.randn(self.batch_size, self.seq_len, self.in_channels)
        output = conv(x)

        expected_output_shape = (self.batch_size, self.seq_len, self.out_channels)
        assert output.shape == expected_output_shape
    

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
