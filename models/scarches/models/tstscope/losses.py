from scipy.special import gamma
import torch
import torch.nn as nn
import torch.nn.functional as F
import ot
import numpy as np

# Adapted from
# Title: Information Constraints on Auto-Encoding Variational Bayes
# Authors: Romain Lopez, Jeffrey Regier, Nir Yosef, Michael I. Jordan
# Code: https://github.com/romain-lopez/HCV/blob/master/hcv.py

def kernel_matrix(x: torch.Tensor, sigma):
    x1  = torch.unsqueeze(x, 0)
    x2  = torch.unsqueeze(x, 1)

    return torch.exp( -sigma * torch.sum(torch.pow(x1-x2, 2), axis=2) )

def bandwidth(d):
    """
    in the case of Gaussian random variables and the use of a RBF kernel,
    this can be used to select the bandwidth according to the median heuristic
    """
    gz = 2 * gamma(0.5 * (d+1)) / gamma(0.5 * d)
    return 1. / (2. * gz**2)

def hsic(z, s):
    d_z = z.shape[1]
    d_s = s.shape[1]

    zz = kernel_matrix(z, bandwidth(d_z))
    ss = kernel_matrix(s, bandwidth(d_s))

    h  = (zz * ss).mean() + zz.mean() * ss.mean() - 2 * torch.mean(zz.mean(1) * ss.mean(1))
    return h.sqrt()

def rbf_kernel(X, sigma=1.0):

    X_norm = torch.sum(X ** 2, dim=1, keepdim=True)
    X_t = torch.transpose(X, 0, 1)
    H = X_norm + torch.transpose(X_norm, 0, 1) - 2 * torch.matmul(X, X_t)
    
    return torch.exp(-H / (2 * sigma**2))

def tcrhsic(z):
    
    # input: output of TCR encoders
    # calculate hsic loss across 10 latent variables of TCR embeddings
    z = z.squeeze(0)
    latent_size = z.shape[1]
    batch_size = z.shape[0]
    z_centered = z - z.mean(dim=0, keepdim=True)
    K = [rbf_kernel(z_centered[:, i].view(-1, 1)) for i in range(latent_size)]
    
    hsic_loss = 0
    for i in range(latent_size):
        for j in range(i+1, latent_size):

            hsic = torch.trace(torch.matmul(K[i], K[j])) / (batch_size**2)
            hsic_loss += hsic

    return hsic_loss

def compute_mean_by_label(matrix, labels):
    unique_labels = torch.unique(labels)
    means = []
    
    for label in unique_labels:
        # Get the rows corresponding to the current label
        mask = (labels == label)
        selected_rows = matrix[mask]
        
        # Compute the mean of the selected rows
        mean = selected_rows.mean(dim=0)
        means.append(mean)
    
    return torch.stack(means)

def cm_loss(Z1, Z2, tcrlabel, temperature=0.5):

    # gene, tcr, emb
    Z1_mean = compute_mean_by_label(Z1, tcrlabel)
    Z2_mean = compute_mean_by_label(Z2, tcrlabel)

    # mse and cos loss across z1 and z2
    mse12 = F.mse_loss(Z1_mean,Z2_mean)
    cos_sim12 = 1 - F.cosine_similarity(Z1_mean.T,Z2_mean.T).mean()

    batch_size, latent_dim = Z1_mean.size()
    similarity_matrix = torch.matmul(Z1_mean, Z1_mean.T) / temperature

    # Compute the cross-entropy loss on z1 scale
    labels = torch.arange(batch_size).to(Z1_mean.device)
    closs = F.cross_entropy(similarity_matrix, labels)

    # align loss weight
    t1 = 1
    t2 = 10
    t3 = 1

    return t1*mse12 + t2*cos_sim12 + t3 * closs

def dcor_loss(Z1, Z2, tcrlabel):

    # gene, tcr, emb
    # unique_Z2, inverse_indices = torch.unique(Z2, dim=0, return_inverse=True)
    # grouped_Z1_means = torch.stack([Z1[inverse_indices == i].mean(dim=0) for i in range(unique_Z2.size(0))])
    # Z1_mean = compute_mean_by_label(Z1, tcrlabel)
    # Z2_mean = compute_mean_by_label(Z2, tcrlabel)
    def compute_correlation_matrix(Z):
        Z_centered = Z - Z.mean(dim=0, keepdim=True)  # Center the matrix
        Z_norm = Z_centered / Z_centered.norm(dim=0, keepdim=True)  # Normalize
        correlation_matrix = Z_norm.T @ Z_norm  # Correlation matrix
        return correlation_matrix
    
    Z1_cor_row = compute_correlation_matrix(Z1)
    Z1_cor_col = compute_correlation_matrix(Z1.T)
    Z2_cor_row = compute_correlation_matrix(Z2)
    Z2_cor_col = compute_correlation_matrix(Z2.T)

    mseloss = F.mse_loss(Z1_cor_row,Z2_cor_row) + F.mse_loss(Z1_cor_col,Z2_cor_col) + F.mse_loss(Z1,Z2)

    # cos_sim12 = 1 - F.cosine_similarity(Z1_mean.T,Z2_mean.T).mean()
    # mse13 = F.mse_loss(Z1,Z3)
    # mse23 = F.mse_loss(Z2,Z3)

    # t1 = 1
    # t2 = 10

    return mseloss

def gaussian_kernel(x, y, sigma=1.0):
    x = x.unsqueeze(1)
    y = y.unsqueeze(0)
    diff = x - y
    return torch.exp(-torch.sum(diff ** 2, axis=-1) / (2 * sigma ** 2))

def mmd_loss(Z1,Z2,kernel=gaussian_kernel):

    # gene, tcr, emb
    xx = kernel(Z1, Z1)
    yy = kernel(Z2, Z2)
    xy = kernel(Z1, Z2)
    
    return torch.mean(xx) + torch.mean(yy) - 2 * torch.mean(xy)








