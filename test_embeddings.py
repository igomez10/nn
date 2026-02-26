import pytest
import torch
import torch.nn as nn


def test_output_shape():
    embedding = nn.Embedding(10, 2)
    indices = torch.tensor([1, 0, 2, 4])
    out = embedding(indices)
    assert out.shape == (4, 2)


def test_output_shape_single_index():
    embedding = nn.Embedding(10, 5)
    indices = torch.tensor([3])
    out = embedding(indices)
    assert out.shape == (1, 5)


def test_output_shape_2d_input():
    embedding = nn.Embedding(10, 4)
    indices = torch.tensor([[1, 2], [3, 4]])
    out = embedding(indices)
    assert out.shape == (2, 2, 4)


def test_weight_shape():
    num_embeddings, embedding_dim = 10, 2
    embedding = nn.Embedding(num_embeddings, embedding_dim)
    assert embedding.weight.shape == (num_embeddings, embedding_dim)


def test_same_index_same_vector():
    embedding = nn.Embedding(10, 2)
    idx = torch.tensor([3])
    out1 = embedding(idx)
    out2 = embedding(idx)
    assert torch.equal(out1, out2)


def test_different_indices_different_vectors():
    torch.manual_seed(0)
    embedding = nn.Embedding(10, 8)
    out0 = embedding(torch.tensor([0]))
    out1 = embedding(torch.tensor([1]))
    assert not torch.equal(out0, out1)


def test_output_has_grad_fn():
    embedding = nn.Embedding(10, 2)
    out = embedding(torch.tensor([1, 2]))
    assert out.grad_fn is not None


def test_output_dtype_is_float():
    embedding = nn.Embedding(10, 2)
    out = embedding(torch.tensor([0]))
    assert out.dtype == torch.float32


def test_out_of_bounds_index_raises():
    embedding = nn.Embedding(10, 2)
    with pytest.raises(IndexError):
        embedding(torch.tensor([10]))


def test_padding_idx_produces_zero_vector():
    embedding = nn.Embedding(10, 4, padding_idx=0)
    out = embedding(torch.tensor([0]))
    assert torch.all(out == 0)
