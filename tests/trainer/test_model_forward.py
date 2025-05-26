import torch
from model import CavitationModel


def test_forward():
    sample_rate = 44100
    duration = 2.0
    input_length = int(sample_rate * duration)

    model = CavitationModel(input_length=input_length, num_classes=7, model_type="msc_cnn")
    model.eval()

    dummy_input = torch.randn(4, 1, input_length)  # batch size 4
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (4, 7), f"Expected (4,7), got {output.shape}"
    print("✅ Forward pass OK")


if __name__ == '__main__':
    test_forward()
