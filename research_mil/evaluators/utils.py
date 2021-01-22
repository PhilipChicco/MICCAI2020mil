
import torch
import numpy as np
import logging
import datetime
import os

from tqdm import tqdm
from torchvision.utils import make_grid, save_image
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.decomposition import PCA
from matplotlib import offsetbox
import seaborn as sns
#sns.set_palette('muted')
#sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
# visualization
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def compute_features_top(model, batch, dataloader):
    model.eval()
    N = len(dataloader.dataset)

    with torch.no_grad():
        final_itr = tqdm(dataloader, desc='Extracting features ...')
        for i, data in enumerate(final_itr):
            input = data[0]
            input = torch.stack(input, 0).squeeze(1).cuda()
            input = input.to(device)
            label = data[1]

            aux = model(input)[1]
            aux = model.module.pooling.get_global(aux).unsqueeze(0)
            aux = aux.data.cpu().numpy()
            lbl = label.data.cpu().numpy()

            if i == 0:
                features = np.zeros((N, aux.shape[1]), dtype='float32')
                labels   = np.zeros((N,), dtype='int')

            aux = aux.astype('float32')
            if i < len(dataloader) - 1:
                features[i * batch: (i + 1) * batch] = aux
                labels[i * batch: (i + 1) * batch] = lbl

            else:
                # special treatment for final batch
                features[i * batch:] = aux
                labels[i * batch:]   = lbl

    return features, labels


def compute_features(model, batch, dataloader):
    model.eval()
    N = len(dataloader.dataset)
    mean = np.array([0.5, 0.5, 0.5])
    std  = np.array([0.1, 0.1, 0.1])

    with torch.no_grad():
        final_itr = tqdm(dataloader, desc='Extracting features ...')
        for i, (input_tensor, label) in enumerate(final_itr):

            input_var = input_tensor.to(device)

            aux = model.module.features(input_var)
            aux = aux.data.view(input_var.shape[0], -1).cpu().numpy()
            lbl = label.data.cpu().numpy()
            #img = input_var.data.view(input_var.shape[0], -1).cpu().numpy()
            img  = input_var.data.cpu().squeeze(0).numpy()
            img  = std * img.transpose((1, 2, 0)) + mean
            img  = np.clip(img, 0, 1)
            img  = np.expand_dims(img.transpose((2, 1, 0)),0)
            img  = img.reshape((img.shape[0],-1))

            if i == 0:
                features = np.zeros((N, aux.shape[1]), dtype='float32')
                labels   = np.zeros((N,), dtype='int')
                images   = np.zeros((N, img.shape[1]), dtype='float32')

            aux = aux.astype('float32')
            if i < len(dataloader) - 1:
                features[i * batch: (i + 1) * batch] = aux
                labels[i * batch: (i + 1) * batch] = lbl
                images[i * batch: (i + 1) * batch] = img
            else:
                # special treatment for final batch
                features[i * batch:] = aux
                labels[i * batch:]   = lbl
                images[i * batch:]   = img

    return features, labels, images


def compute_tsne(X, y, n_class=2,
                 savepath=None,
                 xlim=(-50,50), ylim=(-50,50),
                 cls_lbl=['Benign','Tumor'],
                 title=' ',PCADIM=50):

    tsne = TSNE(n_jobs=4, random_state=1337)
    #X = PCA(n_components=PCADIM).fit_transform(X)
    embs = tsne.fit_transform(X)

    plt.figure(figsize=(10,10))
    for i in range(n_class):
        inds = np.where(y == i)[0]
        plt.scatter(embs[inds, 0], embs[inds, 1], color=colors[i], marker='*', s=30)
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(cls_lbl)
    plt.grid(b=None)
    plt.title(title)
    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
        plt.savefig(savepath.replace('.png','.pdf'), dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.clf()

    # plot distributions
    # savepath = savepath.replace('.png','_dist.png')
    # kwargs = dict(hist_kws={'alpha':.6}, kde_kws={'linewidth':2})
    # plt.figure(figsize=(10, 10))
    # plt.title(title)
    # sns.distplot(embs[:, 0], color=colors[0], label=cls_lbl[0], **kwargs)
    # sns.distplot(embs[:, 1], color=colors[1], label=cls_lbl[1],  **kwargs)
    # plt.gca()
    # plt.legend(cls_lbl)
    # plt.savefig(savepath, dpi=300, bbox_inches='tight')
    # print('Done ...........')

def compute_tsne_aux(X, y, y_images, n_class=2, savepath=None,
                     xlim=(-50,50), ylim=(-50,50),
                     cls_lbl=['Benign','Tumor'], title='Fully Supervised'):

    tsne = TSNE(n_jobs=4, random_state=1337)
    #X = PCA(n_components=100).fit_transform(X)
    embs = tsne.fit_transform(X)

    fig, ax = plt.subplots(figsize=(10, 10))
    for i in range(n_class):
        inds = np.where(y == i)[0]
        plt.scatter(embs[inds, 0], embs[inds, 1], alpha=0.5, color=colors[i], marker='*')
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(cls_lbl)

    if y_images is not None:
        thumb_frac = 0.09
        min_dist_2 = (thumb_frac * max(embs.max(0) - embs.min(0))) ** 2
        shown_images = np.array([2 * embs.max(0)])
        for i in range(y_images.shape[0]):
            dist = np.sum((embs[i] - shown_images) ** 2, 1)
            if np.min(dist) < min_dist_2:
                # don't show points that are too close
                continue
            shown_images = np.vstack([shown_images, embs[i]])
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(to_image(y_images[i]), cmap='jet'),
                embs[i])
            ax.add_artist(imagebox)

    plt.title(title)
    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
        plt.savefig(savepath.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
    else:
        plt.show()

    print('Done ....')


def to_image(tensor,size=32):
    from PIL import Image
    tensor = torch.from_numpy(tensor.reshape(3, 256, 256))
    grid   = make_grid(tensor, nrow=1, normalize=True, scale_each=True)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = (Image.fromarray(ndarr)).resize((size, size), Image.ANTIALIAS)
    return np.array(im)