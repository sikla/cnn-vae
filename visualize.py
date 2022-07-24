from msilib.schema import Class


import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import torch

use_cuda = torch.cuda.is_available()
GPU_indx  = 0
device = torch.device(GPU_indx if use_cuda else "cpu")

class Vizualize():
    def __init__():
        pass

    def show_rand_recon(model, train_loader,  save_as=None):
        iterator = iter(train_loader)
        x, y, type = iterator.next()
        model.eval()
        with torch.no_grad():
            recon, z, mu, log_var, prior_mean = model(x.to(device))
            img = np.moveaxis(np.array(x[0].cpu()), 0, 2)
            recon = np.moveaxis(np.array(recon[0].cpu()), 0, 2)
            fig = plt.figure(figsize=(160, 160))
            ax1 = fig.add_subplot(121)
            ax1.imshow(recon)
            ax2 = fig.add_subplot(122)
            ax2.imshow(img)

    def plot_ls(model, train_loader, test_loader, latent_dim, plot_dim=2, save_as=None):

        z_train, z_test, y_train, y_test, type_test, type_train = [], [], [], [], [], []
        model.eval()
        with torch.no_grad():
            for x, y, type in test_loader:
                x = x.to(device)
                recon, z, mu, log_var, prior_mean = model(x)
                z_test.append(np.asarray(z.cpu()))
                y_test.append(np.asarray(y))
                type_test.append(np.asarray(type))
            for x, y, type in train_loader:
                x = x.to(device)
                recon, z, mu, log_var, prior_mean = model(x)
                z_train.append(np.asarray(z.cpu()))
                y_train.append(np.asarray(y))
                type_train.append(np.asarray(type))

        z_train_ = np.concatenate(z_train).reshape(-1, latent_dim)
        z_test_ = np.concatenate(z_test).reshape(-1, latent_dim)
        type_train_ = np.concatenate(type_train).reshape(-1, 1)
        type_test_ = np.concatenate(type_test).reshape(-1, 1)


        stacked_train = np.array(np.hstack([z_train_, type_train_]), dtype=float)
        stacked_test = np.array(np.hstack([z_test_, type_test_]), dtype=float)


        colors=['red', 'blue']
        c = np.arange(2)

        fig = plt.figure(figsize=(16,5))
        if plot_dim==3:
            ax = fig.add_subplot(121, projection='3d')
            scatter_train = ax.scatter(stacked_train[:,0], stacked_train[:, 1], stacked_train[:, 2],c=stacked_train[:, -1], alpha=.4, s=3**2, cmap=mpl.colors.ListedColormap(colors))
        else:
            ax = fig.add_subplot(121)
            scatter_train = ax.scatter(stacked_train[:, 0], stacked_train[:, 1],c=stacked_train[:, -1], alpha=.4, s=3**2, cmap=mpl.colors.ListedColormap(colors))
        plt.legend(handles=scatter_train.legend_elements()[0], labels=['Normal', 'Anomaly'])
        plt.title('Latent space of training data', fontsize=15)

        if plot_dim==3:
            ax2 = fig.add_subplot(122, projection='3d')
            scatter_test = ax2.scatter(stacked_test[:, 0], stacked_test[:, 1], stacked_test[:, 2], c=stacked_test[:, -1], alpha=.4, s=3**2, cmap=mpl.colors.ListedColormap(colors))
        else:
            ax2 = fig.add_subplot(122)
            scatter_test = ax2.scatter(stacked_test[:,0], stacked_test[:, 1], c=stacked_test[:,-1], alpha=.4, s=3**2, cmap=mpl.colors.ListedColormap(colors))
        plt.legend(handles=scatter_test.legend_elements()[0], labels=['Normal', 'Anomaly'])
        plt.title('Latent space of test data ', fontsize=15)
        plt.show()
        if save_as is not None:
            plt.savefig(save_as + '.png')