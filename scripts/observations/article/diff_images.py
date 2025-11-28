# load the 6 reconstruction images from the pkl files and the ground truth convolved,
# then make the difference and produce the plots for section 6.

import pickle
import os

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as mplc

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from scripts.observations.article.plot_reconstructions import zoomed_in

nantennas = 50
modes = ["peak",]  # "total"]

def truncate_colormap(cmap, minval, maxval, n=100):
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    new_cmap = mplc.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def plot_1_image(image, title="", cmaps=['hot', 'Greys'], alpha=.95, offset_cm=0., symm=True, ticks=None, vlim=None):
    if ticks is None:
        ticks = [1, 500, 1000, 2000, 3000, 4000]
    arr = image.pixels.data[0, 0]

    fig = plt.figure(figsize=(12, 10))
    ax = fig.subplots(1, 1, subplot_kw={'projection': image.image_acc.wcs.sub([1, 2]), 'frameon': False})
    ax.set_xlabel(image.image_acc.wcs.wcs.ctype[0])
    ax.set_ylabel(image.image_acc.wcs.wcs.ctype[1])
    if vlim is None:
        vlim = -arr.min() if symm else 0.
    mask_comp = np.ma.masked_array(arr, arr < vlim, fill_value=vlim)
    mask_res = np.ma.masked_array(arr, arr > vlim, fill_value=vlim)
    cmapc = truncate_colormap(cmaps[0], offset_cm, 1.)
    aximc = ax.imshow(mask_comp, origin="lower", cmap=cmapc, interpolation='none', alpha=alpha,
                      norm=mplc.PowerNorm(gamma=0.5, vmin=vlim, vmax=1. * mask_comp.max()))
    cmapr = truncate_colormap(cmaps[1], 0., 1 - offset_cm)
    aximr = ax.imshow(mask_res, origin="lower", interpolation='none', alpha=alpha,
                      cmap=cmapr, norm='linear', vmin=-vlim, vmax=vlim)
    # norm=symm_sqrt_norm(-vlim, vlim))
    axinsc = inset_axes(ax, width="3%", height="100%", loc='center right', borderpad=-3)
    cbc = fig.colorbar(aximc, cax=axinsc,
                       orientation="vertical", ticks=[round(vlim)] + ticks)
    axinsr = inset_axes(axinsc, width="100%", height="100%", loc='center right', borderpad=-6)
    cbr = fig.colorbar(aximr, cax=axinsr, orientation="vertical")
    fig.suptitle(title)
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.0, right=0.93, hspace=0.15, wspace=0.15)
    fig.show()

sharp = False

if __name__ == "__main__":
    pkl_dir = "/home/jarret/PycharmProjects/polyclean/scripts/observations/article/reco_pkl"
    pkl_dir += f"/{nantennas}antennas"

    # files_dir = [f for f in os.listdir('.') if os.path.isdir(os.path.join('.', f))]
    files_dir = ['0.05', 'autothresh3', '0.02', 'autothresh2', '0.005', 'autothresh1']
    images = []
    for d in files_dir:
        # restored, restored_sharps, comp_restored, comp_restored_sharp, model, residual
        with open(os.path.join(pkl_dir, d, 'comp_restored' + sharp * '_sharp' + '.pkl'), 'rb') as file:
            images.append(pickle.load(file))
    with open(os.path.join(pkl_dir, 'gt', 'gt_peak_' + (1-sharp) * 'cb' + sharp * 'sharp' + '.pkl'), 'rb') as file:
        gt_peak_im = pickle.load(file)
    with open(os.path.join(pkl_dir, 'gt', 'gt_total_' + (1-sharp) * 'cb' + sharp * 'sharp' + '.pkl'), 'rb') as file:
        gt_total_im = pickle.load(file)

    # Plot GT peak
    arr = gt_peak_im.pixels.data[0, 0]
    # arr = np.sqrt(np.abs(arr))*np.sign(arr)
    # vlim = np.abs(arr).max()
    plt.figure(figsize=(12, 10))
    plt.imshow(arr, origin="lower", cmap='hot', interpolation='none')
    plt.colorbar()
    plt.title("GT peak")
    plt.show()

    plot_1_image(gt_peak_im, title="GT peak", cmaps=['hot', 'Greys'], alpha=.95, offset_cm=0., symm=True, ticks=None, vlim=170)

    meanl2_peak = []
    meanl2_total = []
    all_diff = []

    for reco_im, name in zip(images, files_dir):
        print(f"Reconstruction: {name}")
        differences = []
        for gt_im in [gt_peak_im, gt_total_im]:
            arr = reco_im.pixels.data[0, 0] - gt_im.pixels.data[0, 0]
            differences.append(arr)

        ## Plot
        # for arr, mode in zip(differences, modes):
            # arr = np.sqrt(np.abs(arr))*np.sign(arr)
            # vlim = np.abs(arr).max()
            # plt.figure(figsize=(12, 10))
            # plt.imshow(arr, origin="lower", cmap='bwr', interpolation='none', vmin=-vlim, vmax=vlim)
            # plt.colorbar()
            # plt.title(name + " " + mode)
            # plt.show()

            # plt.figure()
            # plt.hist(arr.flatten(), bins=100)
            # plt.title(mode)
            # plt.show()

            # print the mean, median and std of arr, and min and max
        for arr, mode in zip(differences, modes):
            print("Mode: ", mode)
            print(f"\tMean: {arr.mean():.3f}, Median: {np.median(arr):.3f}, Std: {arr.std():.3f}")
            print(f"\tMean error: {np.abs(arr).mean():.3f}")
            print(f"\tMin: {arr.min():.3f}, Max: {arr.max():.3f}")

        meanl2_peak.append(np.abs(differences[0]).mean())
        meanl2_total.append(np.abs(differences[1]).mean())

    print("Mean L2 peak: ", meanl2_peak)
    print("Mean L2 total: ", meanl2_total)

    plot_1_image(images[2], title="Reconstruction 0.02", cmaps=['hot', 'Greys'], alpha=.95, offset_cm=0., symm=True, ticks=None, vlim=170)

    all_diff = [im.pixels.data[0, 0] - gt_peak_im.pixels.data[0, 0] for im in images]

    sqrt = True
    zoomed_in = False
    vlim = np.abs(np.array(all_diff)).max()
    # vlim = 800
    if sqrt:
        vlim = np.sqrt(vlim)

    fig = plt.figure(figsize=(10, 17))
    axes = fig.subplots(3, 2, sharex=True, sharey=True,
                        subplot_kw={'projection': images[0].image_acc.wcs.sub([1, 2]), 'frameon': False})
    for i in range(6):
        ax = axes.ravel()[i]
        arr = all_diff[i]
        name = files_dir[i]
        if sqrt:
            arr = np.sqrt(np.abs(arr)) * np.sign(arr)
        axim = ax.imshow(arr, origin="lower", cmap='bwr', interpolation='none', vmin=-vlim, vmax=vlim)
        ax.set_title(name)
        if i in [1, 3, 5]:
            ax.coords[1].set_ticklabel_visible(False)
            ax.coords[1].set_axislabel('')
        else:
            ax.set_ylabel(images[0].image_acc.wcs.wcs.ctype[1])

        if i in [4, 5]:
            ax.set_xlabel(images[0].image_acc.wcs.wcs.ctype[0])
        else:
            ax.coords[0].set_ticklabel_visible(False)
            ax.coords[0].set_axislabel('')

        if zoomed_in:
            x1, x2, y1, y2 = 390, 750, 1890, 2250 # 360, 360
            axins = ax.inset_axes(
                [0.5, 0.5, 0.47, 0.47],
                xlim=(x1, x2), ylim=(y1, y2), xticklabels=[], yticklabels=[], xticks=[], yticks=[], )
            axins.imshow(arr, origin="lower", cmap='bwr', interpolation='none', vmin=-vlim,
                         vmax=vlim)  # , extent=extent)
            ax.indicate_inset_zoom(axins, edgecolor="black")

    cbar_ax = inset_axes(axes[0, 0], width="250%", height="6%", loc='upper left', borderpad=-5)
    if sqrt:
        ticks = [10, 50, 200, 500, 1000]
        ticks = [-s for s in ticks[::-1]] + [0] + ticks
        ticks = np.array(ticks)
    cbar = fig.colorbar(axim, cax=cbar_ax, orientation="horizontal", ticks=np.sqrt(np.abs(ticks)) * np.sign(ticks))
    cbar.ax.set_xticklabels([f"{t:.0f}" for t in ticks])
    plt.subplots_adjust(top=0.9, bottom=0.05, left=0.1, right=0.9, hspace=0.09, wspace=0.)
    fig.show()

    with np.printoptions(precision=2):
        print("MSE peak:")
        print(np.sum(np.array(all_diff)**2, axis=(1, 2))/(images[0].dims['x'] * images[0].dims['y']))
        print("MAD peak:")
        print(np.sum(np.abs(np.array(all_diff)), axis=(1, 2))/(images[0].dims['x'] * images[0].dims['y']))

    for im in images:
        print(im.dims)

    # arr = images[2].pixels.data[0, 0] - images[1].pixels.data[0, 0]
    # vlim = np.abs(arr).max()
    # plt.figure(figsize=(12, 10))
    # plt.imshow(arr, origin="lower", cmap='bwr', interpolation='none', vmin=-vlim, vmax=vlim)
    # plt.colorbar()
    # plt.show()