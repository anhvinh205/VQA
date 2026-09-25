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
    output = model(img, question)
    assert output.shape == (4, 2)


def test_padding_does_not_change_question_representation():
    model = VQAModel(
        vocab_size=50,
        n_classes=2,
        embedding_dim=16,
        n_layers=1,
        hidden_size=32,
        drop_p=0.0,
        pad_idx=0,
        pretrained_backbone=False,
    )
    model.eval()
    image = torch.randn(1, 3, 64, 64)
    question = torch.tensor([[4, 5, 6, 0, 0]])
    with torch.no_grad():
        padded_output = model(image, question)
        shorter_output = model(image, torch.tensor([[4, 5, 6]]))
    assert torch.allclose(padded_output, shorter_output)