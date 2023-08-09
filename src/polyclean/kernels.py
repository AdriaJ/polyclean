"""
Gaussian convolution kernels for the reconstruction of extended sources.
"""

import time
import numpy as np
import typing as typ
import collections.abc as cabc

import pycsou.operator as pycop
import pycsou.util.ptype as pyct

__all__ = [
    "kernelGen",
    "stackedKernels"
]

def gauss1D(x, scale, norm=2):
    res = np.exp(-x ** 2 / scale ** 2)
    return res / np.linalg.norm(res, norm)


def gauss2D(x, scale):
    """
    x should be a tuple of coordinates ranges (x_range, y_range)
    """
    radius = x[0].reshape((-1, 1)) ** 2 + x[1].reshape((1, -1)) ** 2
    res = np.exp(-radius / scale ** 2)
    return res / np.linalg.norm(res, 'fro')


def kernelGen(arg_shape, scale, n_supp: int = 2, norm: int = 2) -> pyct.OpT:
    """

    Parameters
    ----------
    arg_shape:
        Shape of the image
    kernel: np.ndarray
        1D kernel, so that will form the steerable filter
    """
    assert arg_shape[0] == arg_shape[1]
    if scale == 0:
        return pycop.IdentityOp(dim=arg_shape[0]**2)
    half_support = np.ceil(n_supp * scale).astype(int)
    kernel_center = half_support
    kernek1d = gauss1D(np.arange(2 * half_support + 1) - half_support, scale, norm=norm)
    convOp = pycop.Convolve(arg_shape, (kernek1d, kernek1d), (half_support,) * 2, mode='constant')
    injection = pycop.SubSample(arg_shape,
                                slice(half_support, arg_shape[0] - half_support),
                                slice(half_support, arg_shape[0] - half_support)).T
    res = convOp * injection
    res.lipschitz(tight=False)  # we use the upper bound from the Stencil operator
    return res


def stackedKernels(
        input_shape: typ.Union[pyct.NDArrayShape, cabc.Sequence[pyct.NDArrayShape]],
        scale_list: typ.Union[int, cabc.Sequence[int]],
        n_supp: int = 2,
        norm: int = 2,
        bias_list: list = None,
        tight_lipschitz: bool = True,
        verbose: bool = False,
) -> pyct.OpT:
    if not isinstance(scale_list, list):
        op = kernelGen(input_shape, scale_list)
        if bias_list is not None:
            if isinstance(bias_list, list):
                bias = bias_list[0]
            else:
                bias = bias_list
            return pycop.HomothetyOp(op.shape[0], bias) * op
        else:
            return op
    if verbose:
        start = time.time()
    op_list = [kernelGen(arg_shape=input_shape, scale=s, n_supp=n_supp, norm=norm) for s in scale_list]
    if bias_list is not None:
        assert len(bias_list) == len(scale_list)
        op_list = [pycop.HomothetyOp(op.shape[0], bias) * op for op, bias in zip(op_list, bias_list)]
    res = pycop.stack(op_list, axis=1)
    if verbose:
        print("Instantiation time: {:.2f}".format(time.time() - start))
        start = time.time()
    res.lipschitz(tight=tight_lipschitz)
    if verbose:
        print("Lipschitz time: {:.2f}".format(time.time() - start))
    return res

def stack_list_coeffs(coeffs, stackOp, scale_list):
    offsets = [p[1] for p in stackOp._block_offset.values()] + [stackOp.shape[1], ]
    cores = []
    for i in range(len(scale_list)):
        s = slice(offsets[i], offsets[i+1])
        cores.append(coeffs[s])
    return cores

def stack_list_sources(coeffs, stackOp, scale_list, n_supp):
    cores = stack_list_coeffs(coeffs, stackOp, scale_list)
    sources = []
    for core, s in zip(cores, scale_list):
        side_length = np.round(core.shape[0]**.5).astype(int)
        margin = np.ceil(n_supp * s).astype(int)
        sources.append(np.pad(core.reshape((side_length, ) * 2),
                              (margin, margin))
                       )
    return sources

def stack_list_components(coeffs, stackOp, scale_list, arg_shape):
    cores = stack_list_coeffs(coeffs, stackOp, scale_list)
    components = []
    ops = stackOp._block.values()
    for core, op in zip(cores, ops):
        components.append(op.apply(core).reshape(arg_shape))
    return components

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib import use
    use("Qt5Agg")

    scales = [1, 3, 5, 0]
    n_sources = [10, 5, 3, 10]
    arg_shape = (200,) * 2
    n_supp = 2


    # ops = [kernelGen(arg_shape, s) for s in scales]
    # print([op.lipschitz() for op in ops])
    # print([op.lipschitz(tight=True) for op in ops])
    # print([op.shape for op in ops])
    # res = pycop.stack(ops, axis=1)
    # print(res.lipschitz())
    # print(res.lipschitz(tight=True))
    # offsets = res._block_offset

    kOp = stackedKernels(arg_shape, scales, n_supp=2, tight_lipschitz=False, verbose=True)
    offsets = [p[1] for p in kOp._block_offset.values()] + [kOp.shape[1], ]
    n_components = len(scales)

    # set some coefficients
    coeffs = np.zeros(kOp.shape[1])
    for i in range(n_components):
        s = slice(offsets[i], offsets[i+1])
        c = coeffs[s]
        c[np.random.choice(np.arange(c.shape[0]), size=n_sources[i])] = 1.
        coeffs[s] = c

    # compute the image
    output_im = kOp(coeffs).reshape(arg_shape)

    # extract component sub images information
    # segment coefficients
    cores = []
    for i in range(n_components):
        s = slice(offsets[i], offsets[i+1])
        cores.append(coeffs[s])

    # extract the sources
    sources = []
    for core, s in zip(cores, scales):
        side_length = np.round(core.shape[0]**.5).astype(int)
        margin = np.ceil(n_supp * s).astype(int)
        sources.append(np.pad(core.reshape((side_length, ) * 2),
                              (margin, margin))
                       )

    # compute the sub-images
    components = []
    ops = kOp._block.values()
    for i, op in enumerate(ops):
        components.append(op.apply(cores[i]).reshape(arg_shape))

    ## plot the informations

    # # plot the sources as pixels
    # fig = plt.figure(figsize=(4 * len(scales) + 2, 5))
    # axes = fig.subplots(1, len(scales), sharex=True, sharey=True)
    # for i in range(len(scales)):
    #     ax = axes[i]
    #     ax.imshow(sources[i].T, origin='lower', interpolation='none')
    # plt.show()

    # place the locations of the centers
    n_components = len(kOp._block)
    fig = plt.figure(figsize=(4 * n_components + 2, 5))
    axes = fig.subplots(1, n_components, sharex=True, sharey=True)
    vmin, vmax = output_im.min(), output_im.max()
    for i in range(len(scales)):
        ax = axes[i]
        ims = ax.imshow(components[i].T, origin='lower', interpolation='none', vmin=vmin, vmax=vmax)
        ax.scatter(*np.where(sources[i] != 0), marker='.', color='r', alpha=.5)
    plt.show()


    # plot the output image
    plt.figure(figsize=(10, 10))
    plt.imshow(output_im.T, origin='lower')
    for s in sources:
        plt.scatter(*np.where(s != 0), marker='.', color='r', alpha=.5)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    for s in [1, 2, 4, 8]:
        k = kernelGen(arg_shape, s, norm=2)
        co = np.zeros(k.shape[1])
        co[20100] = 1.
        print("scale ", s)
        print(np.linalg.norm(k(co), 2), np.linalg.norm(k(co), 1))

    for s in [1, 2, 4, 8]:
        k = kernelGen(arg_shape, s, norm=1)
        co = np.zeros(k.shape[1])
        co[20100] = 1.
        print("scale ", s)
        print(np.linalg.norm(k(co), 2), np.linalg.norm(k(co), 1))

    # mutual coherence:
    scales = [0, 1, 3, 8]
    ops = [kernelGen(arg_shape, s, norm=2) for s in scales]
    mutual_coherence = np.zeros((4, 4))
    for i in range(4):
        k1 = ops[i]
        side1 = int(np.sqrt(k1.shape[1]))
        c1 = np.zeros(k1.shape[1]).reshape((side1, side1))
        c1[side1//2, side1//2] = 1.
        r1 = k1(c1.flatten()).reshape(arg_shape)
        # print(r1.shape, np.linalg.norm(r1))
        for j in range(i, 4):
            k2 = ops[j]
            side2 = int(np.sqrt(k2.shape[1]))
            c2 = np.zeros(k2.shape[1]).reshape((side2, side2))
            c2[side2//2, side2//2] = 1.
            r2 = k2(c2.flatten()).reshape(arg_shape)
            # print(r2.shape, np.linalg.norm(r2))

            mutual_coherence[i, j] = np.sum(r1 * r2) / np.sqrt(np.sum(r1**2) * np.sum(r2**2))
    print(mutual_coherence)


    # import numpy as np
    # import polyclean.kernels as pck
    # from matplotlib import use
    # import matplotlib.pyplot as plt
    # use("Qt5Agg")
    # op = pck.kernelGen((50, 50), 2)
    # a = np.zeros(op.shape[0])
    # a[50*25 + 25] = 1
    # res = op.adjoint(a)
    # plt.figure()
    # plt.imshow(res.reshape((42, 42)))
    # plt.show()