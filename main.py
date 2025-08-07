import torch
from sequence_encoder import SequenceEncoder


def test_sequence_encoder():
    # Initialize encoder
    encoder = SequenceEncoder(in_channels=4)  # 4 for DNA (A, C, G, T)
    
    # Test input: 1Mbp sequence (2^20 = 1,048,576)
    batch_size = 2
    seq_len = 2**20  # 1Mbp
    channels = 4
    
    x = torch.randn(batch_size, seq_len, channels)
    
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    output, intermediates = encoder(x)
    
    print(f"Output shape: {output.shape}")
    print("Expected output shape: (batch_size, 8192, 1536)")
    print(f"Actual output shape: {output.shape}")
    
    # Check intermediate shapes
    expected_shapes = {
        'bin_size_1': (batch_size, seq_len, 768),
        'bin_size_2': (batch_size, seq_len // 2, 896),
        'bin_size_4': (batch_size, seq_len // 4, 1024),
        'bin_size_8': (batch_size, seq_len // 8, 1152),
        'bin_size_16': (batch_size, seq_len // 16, 1280),
        'bin_size_32': (batch_size, seq_len // 32, 1408),
        'bin_size_64': (batch_size, seq_len // 64, 1536),
    }
    
    for key, expected_shape in expected_shapes.items():
        if key in intermediates:
            actual_shape = intermediates[key].shape
            print(f"{key}: expected {expected_shape}, got {actual_shape}")
            assert actual_shape == expected_shape, f"Shape mismatch for {key}"
    
    print("✅ All tests passed!")


if __name__ == "__main__":
    test_sequence_encoder()
