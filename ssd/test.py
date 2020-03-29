import torch
import numpy as np

from utils import sigmoid, softmax, addcmul, topk


if __name__ == '__main__':
    ndarray = np.random.rand(3, 3)
    tensor = torch.tensor(ndarray)
    assert(ndarray.shape[-2:] == tensor.size()[-2:])
    assert(np.allclose(sigmoid(ndarray), tensor.sigmoid().numpy()))
    assert(np.allclose(softmax(ndarray), tensor.softmax(-1).numpy()))

    px = np.random.rand(3, 3)
    pw = np.random.rand(3, 3)
    dx = np.random.rand(3, 3)
    px_t = torch.tensor(px)
    pw_t = torch.tensor(pw)
    dx_t = torch.tensor(dx)

    assert(np.allclose(addcmul(px, pw, dx), torch.addcmul(px_t, 1, pw_t, dx_t)))

    ndarray = np.random.rand(12, 15, 18)
    tensor = torch.tensor(ndarray)
    assert(np.array_equal(np.transpose(ndarray, (1, 0, 2)).reshape((-1, 12)), tensor.permute(1, 0, 2).reshape(-1, 12)))

    assert(np.allclose(np.max(ndarray, axis=1), tensor.max(dim=1)[0].numpy()))

    dists = np.random.permutation(np.arange(30)).reshape(6, 5)
    print(dists)
    tensor = torch.tensor(dists)
    _, idx = tensor.topk(2)
    print(dists[topk(dists, 2, axis=1)[0][0]])
    print(dists[topk(dists, 2, axis=1)[0][1]])
    print(tensor[idx].numpy())
