import os

import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as mplc
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def truncate_colormap(cmap, minval, maxval, n=100):
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    new_cmap = mplc.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def plot_3_images(im_list, title_list, suptitle="", normalize=True, offset_cm=0., vlim=None, alpha=.8, save=None,
                  windowx=None, windowy=None):
    chan, pol = 0, 0
    set_vlim = vlim is None
    if normalize:
        vmax = max([np.abs(im.pixels.data).max() for im in im_list])
        if set_vlim:
            vlim = max(0, -min([im.pixels.data.min() for im in im_list]))
    else:
        vmax = None
    any_lim = False
    if windowx is None:
        windowx = slice(None)
    if windowy is None:
        windowy = slice(None)

    fig = plt.figure(figsize=(15, 5))
    axes = fig.subplots(1, 3, sharex=True, sharey=True,
                        subplot_kw={'projection': im_list[0].image_acc.wcs.sub([1, 2]), 'frameon': False})
    for i in range(3):
        ax = axes[i]
        arr = np.real(im_list[i]["pixels"].data[chan, pol, windowy, windowx])
        if not normalize:
            if i == 0:
                vmax = np.abs(arr).max()
                if set_vlim:
                    vlim = max(0, -arr.min())
            else:
                vmax = max([np.abs(im.pixels.data).max() for im in im_list[1:]])
                if set_vlim:
                    vlim = max(0, -min([im.pixels.data.min() for im in im_list[1:]]))
        # ims = ax.imshow(arr, origin="lower", cmap=cm, vmin=vmin, vmax=vmax)
        im_pos = np.ma.masked_array(arr, arr < vlim, fill_value=vlim)
        if (arr < vlim).any():
            im_neg = np.ma.masked_array(arr, arr > vlim, fill_value=vlim)
            cmapn = truncate_colormap('Greys', offset_cm, 1. - offset_cm)
            aximn = ax.imshow(im_neg, origin="lower", cmap=cmapn, interpolation='none', alpha=alpha,
                              vmin=-vlim, vmax=vlim)
            any_lim = True
        cmapp = truncate_colormap('hot', offset_cm, 1.)
        aximp = ax.imshow(im_pos, origin="lower", cmap=cmapp, interpolation='none', vmax=vmax, vmin=vlim)
        if i == 0:
            ax.set_ylabel(im_list[i].image_acc.wcs.wcs.ctype[1])
        else:
            ax.coords[1].set_ticklabel_visible(False)
            ax.coords[1].set_axislabel('')
        ax.set_xlabel(im_list[i].image_acc.wcs.wcs.ctype[0])
        ax.set_title(title_list[i])
        # fig.colorbar(ims, orientation="vertical", shrink=0.5, ax=ax)
        if i == 2:
            axinsp = inset_axes(ax, width="3%", height="100%", loc='center right', borderpad=-3)
            cbp = fig.colorbar(aximp, cax=axinsp, orientation="vertical")
            cbp.ax.tick_params(labelsize=14)
            if any_lim:
                axinsn = inset_axes(axinsp, width="100%", height="100%", loc='center right', borderpad=-4)
                cbn = fig.colorbar(aximn, cax=axinsn, orientation="vertical")
                cbn.ax.tick_params(labelsize=14)
        if not normalize and i == 0:
            axinsp = inset_axes(ax, width="3%", height="100%", loc='center left', borderpad=-8)
            cbp = fig.colorbar(aximp, cax=axinsp, orientation="vertical")
            cbp.ax.tick_params(labelsize=14)
            if any_lim:
                axinsn = inset_axes(axinsp, width="100%", height="100%", loc='center left', borderpad=-4)
                cbn = fig.colorbar(aximn, cax=axinsn, orientation="vertical")
                cbn.ax.tick_params(labelsize=14)

    fig.suptitle(suptitle)
    # plt.subplots_adjust(
    #     top=0.918,
    #     bottom=0.027,
    #     left=0.047,
    #     right=0.991,
    #     hspace=0.2,
    #     wspace=0.023)
    if save is not None:
        plt.savefig(save, bbox_inches='tight')
    plt.show()

if __name__=="__main__":
    # load the images from the pkl files
    folders = ['pclean', 'rsclean', 'wsclean']
    names = ['comp.pkl', 'comp_conv.pkl', 'comp_conv_res.pkl']
    images = [[], [], []]
    for i, name in enumerate(names):
        for folder in folders:
            with open(os.path.join(os.getcwd(), 'reco_pkl', folder, name), 'rb') as handle:
                images[i].append(pickle.load(handle))

    with open(os.path.join(os.getcwd(), 'reco_pkl', 'dirty.pkl'), 'rb') as handle:
        dirty = pickle.load(handle)
    with open(os.path.join(os.getcwd(), 'reco_pkl', 'source.pkl'), 'rb') as handle:
        source = pickle.load(handle)
    with open(os.path.join(os.getcwd(), 'reco_pkl', 'source_conv.pkl'), 'rb') as handle:
        source_conv = pickle.load(handle)

    # RASCIL CLEAN
    plot_3_images([source, ] + images[0][:-1],
                  ['Source', 'PolyCLEAN', 'CLEAN'],
                  suptitle="Comparison components",
                  normalize=True, vlim=0.3, alpha=1)
    plot_3_images([source_conv, ] + images[1][:-1],
                  ['Source', 'PolyCLEAN', 'CLEAN'],
                  suptitle="Comparison components convolved sharp",
                  normalize=True)
    plot_3_images([source_conv, ] + images[2][:-1],
                  ['Source', 'PolyCLEAN', 'CLEAN'],
                  suptitle="Comparison restored sharp (components + residual)",
                  normalize=True, offset_cm=0.05)

    save_dir = 'figures'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # WS CLEAN
    plot_3_images([source, ] + images[0][::2],
                  ['Source', 'PolyCLEAN', 'WS-CLEAN'],
                  suptitle="Comparison components",
                  normalize=True, vlim=0.3, alpha=1,
                  save=os.path.join(os.getcwd(), save_dir, 'comparison_components.png'))
    plot_3_images([source_conv, ] + images[1][::2],
                  ['Source', 'PolyCLEAN', 'WS-CLEAN'],
                  suptitle="Comparison components convolved sharp",
                  normalize=True,
                  save=os.path.join(os.getcwd(), save_dir, 'comparison_components_convolved.png'))
    plot_3_images([source_conv, ] + images[2][::2],
                  ['Source', 'PolyCLEAN', 'WS-CLEAN'],
                  suptitle="Comparison restored sharp (components + residual)",
                  normalize=True, offset_cm=0.05,
                  save=os.path.join(os.getcwd(), save_dir, 'comparison_restored.png'))

    # Zoom in the centre of the image, compare without and with convolution
    # chan, pol = 0, 0
    # vlim1 = 0.3
    # vmax1 = max([np.abs(im.pixels.data).max() for im in images[0][::2]])
    # vlim2 = max(0, -min([im.pixels.data.min() for im in images[1][::2]]))
    # vmax2 = max([np.abs(im.pixels.data).max() for im in images[1][::2]])

    windowx = slice(100, 175)
    windowy = slice(120, 195)
    plot_3_images([source, ] + images[0][::2],
                  ['Source', 'PolyCLEAN', 'WS-CLEAN'],
                  suptitle="Comparison components",
                  normalize=True, vlim=0.3, alpha=1,
                  windowx=windowx, windowy=windowy,
                  save=os.path.join(os.getcwd(), save_dir, 'zoom_components.png'))
    plot_3_images([source_conv, ] + images[1][::2],
                  ['Source', 'PolyCLEAN', 'WS-CLEAN'],
                  suptitle="Comparison components",
                  normalize=True, vlim=0.3, alpha=1,
                  windowx=windowx, windowy=windowy,
                  save=os.path.join(os.getcwd(), save_dir, 'zoom_components_convolved.png'))
    plot_3_images([source_conv, ] + images[2][::2],
                  ['Source', 'PolyCLEAN', 'WS-CLEAN'],
                  suptitle="Comparison components",
                  normalize=True, vlim=0.3, alpha=1,
                  windowx=windowx, windowy=windowy,
                  save=os.path.join(os.getcwd(), save_dir, 'zoom_restored.png'))

    # Difference imaging
    # import matplotlib as mpl
    # mpl.rcParams['text.usetex'] = False
    from mpl_toolkits.axes_grid1 import ImageGrid
    import matplotlib.ticker as ticker
    diff_comp = [[(images[i][0]["pixels"].data - s["pixels"].data)/np.abs(s["pixels"].data.max()),
                  (images[i][2]["pixels"].data - s["pixels"].data)/np.abs(s["pixels"].data.max()),
                  (images[i][0]["pixels"].data - images[i][2]["pixels"].data)/np.abs(s["pixels"].data.max())]
                 for i, s in enumerate([source, source_conv, source_conv])]

    # Components
    vlim = max([np.abs(im).max() for im in diff_comp[0]])
    vlim = np.sqrt(vlim)
    fig = plt.figure(figsize=(15, 5))
    grid = ImageGrid(fig, 111, nrows_ncols=(1, 3), axes_pad=0.1,
                     share_all=True, cbar_location="right", cbar_mode="single")
    for ax, im, t in zip(grid, diff_comp[0], ['PolyCLEAN minus source', "WS-CLEAN minus source", "PolyCLEAN minus WS-CLEAN"]):
        arr = im[0, 0]
        arr = np.sqrt(np.abs(arr)) * np.sign(arr)
        im = ax.imshow(arr, interpolation="none", origin="lower", cmap='seismic', vmax=vlim, vmin=-vlim)
        # ax.set_title(t)
        ax.set_xticks([])
        ax.set_yticks([])
    cb = grid.cbar_axes[0].colorbar(im)
    new_tick_loc = np.array([-4, -2, -1, -.5, 0, .5, 1, 2, 4])
    cb.ax.tick_params(labelsize=14)
    cb.ax.yaxis.set_major_locator(ticker.FixedLocator(np.sqrt(np.abs(new_tick_loc)) * np.sign(new_tick_loc)))
    cb.ax.yaxis.set_major_formatter(ticker.FixedFormatter(new_tick_loc))
    # fig.suptitle("Components")
    # plt.savefig(os.path.join(os.getcwd(), save_dir, 'diff_comp.png'), bbox_inches='tight')
    fig.show()

    print([(im**2).sum() for im in diff_comp[0]])
    print([(im**2).sum() for im in diff_comp[1]])

    # Components convolved
    vlim = max([np.abs(im).max() for im in diff_comp[1]])
    fig = plt.figure(figsize=(15, 5))
    grid = ImageGrid(fig, 111, nrows_ncols=(1, 3), axes_pad=0.1,
                     share_all=True, cbar_location="right", cbar_mode="single")
    for ax, im, t in zip(grid, diff_comp[1], ['PolyCLEAN minus source', "WS-CLEAN minus source", "PolyCLEAN minus WS-CLEAN"]):
        arr = im[0, 0]
        # arr = np.sqrt(np.abs(arr)) * np.sign(arr)
        im = ax.imshow(arr, interpolation="none", origin="lower", cmap='seismic', vmax=vlim, vmin=-vlim)
        # ax.set_title(t)
        ax.set_xticks([])
        ax.set_yticks([])
    cb = grid.cbar_axes[0].colorbar(im)
    cb.ax.tick_params(labelsize=14)
    # fig.suptitle("Components convolved")
    # plt.savefig(os.path.join(os.getcwd(), save_dir, 'diff_comp_conv.png'), bbox_inches='tight')
    fig.show()

    plt.figure()
    plt.hist(diff_comp[1][-1].flatten(), bins=100)
    plt.yscale('log')
    plt.show()

    # print median, mean and standard deviation of the absolute value of the last image
    diff_conv = np.abs(diff_comp[1][-1])
    print(f"Median: {np.median(diff_conv):.3e} - Mean: {np.mean(diff_conv):.3e} - Standard deviation: {np.std(diff_conv):.3e}")
    # Same thing considering only the non zeros pixels of diff_conv
    masked_diff = diff_conv[diff_conv>1e-3]
    print(f"Median: {np.median(masked_diff):.3e} - Mean: {np.mean(masked_diff):.3e} - Standard deviation: {np.std(masked_diff):.3e}")

    for im in diff_comp[0]:
        arr = np.abs(im)
        print(
            f"Median: {np.median(arr):.3e} - Mean: {np.mean(arr):.3e} - Standard deviation: {np.std(arr):.3e}")

    for im in diff_comp[1]:
        arr = np.abs(im)
        print(
            f"Median: {np.median(arr):.3e} - Mean: {np.mean(arr):.3e} - Standard deviation: {np.std(arr):.3e}")

    # for d, title in zip(diff_comp, ["Components", "Components convolved", "Restored"]):
    #     sqrt = title == "Components"
    #     vlim = max([np.abs(im).max() for im in d])
    #     if sqrt: vlim = np.sqrt(vlim)
    #     fig = plt.figure(figsize=(15, 5))
    #     grid = ImageGrid(fig, 111,
    #                      nrows_ncols=(1, 3),
    #                      axes_pad=0.1,
    #                      share_all=True,
    #                      cbar_location="right", cbar_mode="single"
    #                      )
    #     for ax, im, t in zip(grid, d, ['PolyCLEAN minus source', "WS-CLEAN minus source", "PolyCLEAN minus WS-CLEAN"]):
    #         arr = im[0, 0]
    #         if sqrt: arr = np.sqrt(np.abs(arr)) * np.sign(arr)
    #         im = ax.imshow(arr, interpolation="none", origin="lower", cmap='seismic', vmax=vlim, vmin=-vlim)
    #         ax.set_title(t)
    #         ax.set_xticks([])
    #         ax.set_yticks([])
    #     cb = grid.cbar_axes[0].colorbar(im)
    #     if sqrt:
    #         new_tick_loc = np.array([-1, -.5, 0, .5, 1])
    #         cb.ax.yaxis.set_major_locator(ticker.FixedLocator(np.sqrt(np.abs(new_tick_loc))*np.sign(new_tick_loc)))
    #         cb.ax.yaxis.set_major_formatter(ticker.FixedFormatter(new_tick_loc))
    #     fig.suptitle(title)
    #     fig.show()

