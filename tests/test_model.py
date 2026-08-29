import torch

from src.models.cnn_lstm import VQAModel


def test_forward_pass_output_shape():
    model = VQAModel(
        vocab_size=50,
        n_classes=2,
        embedding_dim=16,
        n_layers=1,
        hidden_size=32,
        drop_p=0.0,
        pretrained_backbone=False,
    )
    model.eval()

    img = torch.randn(4, 3, 64, 64)
    question = torch.randint(0, 50, (4, 10))

    with torch.no_grad():
        out = model(img, question)

    assert out.shape == (4, 2)
