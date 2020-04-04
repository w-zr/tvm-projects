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
    tensor = torch.tensor(dists)
    _, idx = tensor.topk(2)
    assert(np.array_equal(dists[topk(dists, 2, axis=1)], tensor[idx].numpy()))

    ndarray = np.random.rand(12, 15, 18, 21)
    tensor = torch.tensor(ndarray)

    assert(np.array_equal(tensor.new_tensor([0, 0, 0, 0]).repeat(1, tensor.size(1)//4).numpy(),
                          np.tile(np.array([0, 0, 0, 0], dtype=ndarray.dtype), (1, tensor.size(1)//4))))
    assert(np.array_equal(tensor.unsqueeze(1).numpy(), np.expand_dims(ndarray, axis=1)))

    assert(np.array_equal(ndarray.reshape(ndarray.shape), tensor.expand_as(tensor).numpy()))

    x = torch.randn(3, 4)
    indices = torch.tensor([0, 2])
    assert(np.allclose(torch.index_select(x, 0, indices).numpy(), x.numpy()[indices.numpy()]))

    import torch
    rois = np.array([[0., 0., 1., 1.], [0., 0., 1., 1.], [0., 0., 1., 1.], [5., 5., 5., 5.]])
    deltas = np.array([[0., 0., 0., 0.], [1., 1., 1., 1.], [0., 0., 2., -1.], [0.7, -1.9, -0.5, 0.3]])

    import bbox_utils
    #print(bbox_utils.delta2bbox(rois, deltas, max_shape=(32, 32)))

    y = np.zeros((1, 2, 3, 4, 5))

    print(y.size)
    print(x.numel())
    assert(y.size == x.numel())