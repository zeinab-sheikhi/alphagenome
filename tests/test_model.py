
import torch
import pytest

from alphagenome.model import SequenceEncoder


class TestSequenceEncoder:
    def setup_method(self):
        self.batch_size = 4
        self.input_seq_len = 2 ** 12
        self.input_channels = 4

        self.encoder = SequenceEncoder(in_channels=4)

        self.x = torch.randn(self.batch_size, self.input_seq_len, self.input_channels)
    
    def test_final_output_shape(self):
        final_output, _ = self.encoder(self.x)

        expected_seq_len = self.input_seq_len // (2**7)  # 32
        expected_channels = 1536  # 768 + 6 * 128
        expected_shape = (self.batch_size, expected_seq_len, expected_channels)

        actual_shape = final_output.shape
        
        assert actual_shape == expected_shape, (
            f"Expected final output shape {expected_shape}, got {actual_shape}"
        )
        

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
