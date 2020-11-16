import warnings
import numpy as np
import scipy as sp
from scipy import stats

import torch
import torch.nn as nn
import torch.nn.functional as F


def create_batches(features, y, batchsize):

    # Create random indices
    n = features.shape[0]
    p = features.shape[1]
    inds = torch.randperm(n)

    # Iterate through and create batches
    i = 0
    batches = []
    while i < n:
        batches.append([features[inds][i : i + batchsize], y[inds][i : i + batchsize]])
        i += batchsize
    return batches


class DeepPinkModel(nn.Module):
    def __init__(self, p, inds, rev_inds, hidden_sizes=[64], y_dist="gaussian"):
        """
        Adapted from https://arxiv.org/pdf/1809.01185.pdf.

        Module has two components:
        1. A sparse linear layer with dimension 2*p to p.
        However, there are only 2*p weights (each feature
        and knockoff points only to their own unique node).
        This is (maybe?) followed by a ReLU activation.
        2. A MLP 

        :param p: The dimensionality of the data
        :param hidden_sizes: A list of hidden sizes
        for the mlp layer(s). Defaults to [64], which 
        means there will be one two hidden layers 
        (one p -> 64, one p -> 128). 
        """

        super().__init__()

        # Initialize weight for first layer
        self.p = p
        self.y_dist = y_dist
        self.Z_weight = nn.Parameter(torch.ones(2 * p))

        # Create MLP layers
        mlp_layers = [nn.Linear(p, hidden_sizes[0])]
        for i in range(len(hidden_sizes) - 1):
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
        # Prepare for either MSE loss or cross entropy loss
        mlp_layers.append(nn.ReLU())
        if y_dist == "gaussian":
            mlp_layers.append(nn.Linear(hidden_sizes[-1], 1))
        else:
            mlp_layers.append(nn.Linear(hidden_sizes[-1], 2))

        # Then create MLP
        self.mlp = nn.Sequential(*mlp_layers)

    def normalize_Z_weight(self):

        # First normalize
        normalizer = torch.abs(self.Z_weight[0 : self.p]) + torch.abs(
            self.Z_weight[self.p :]
        )
        return torch.cat(
            [
                torch.abs(self.Z_weight[0 : self.p]) / normalizer,
                torch.abs(self.Z_weight[self.p :]) / normalizer,
            ],
            dim=0,
        )

    def forward(self, features):
        """
        NOTE: FEATURES CANNOT BE SHUFFLED
        """

        # First layer: pairwise weights (and sum)
        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features).float()
        features = self.normalize_Z_weight().unsqueeze(dim=0) * features
        features = features[:, 0 : self.p] - features[:, self.p :]

        # Apply MLP
        return self.mlp(features)

    def predict(self, features):
        """
        Wraps forward method, for compatibility
        with sklearn classes.
        """
        with torch.no_grad():
            return self.forward(features).numpy()

    def l1norm(self):
        out = 0
        for parameter in self.mlp.parameters():
            out += torch.abs(parameter).sum()
        out += torch.abs(self.Z_weight).sum()  # This is just for stability
        return out

    def l2norm(self):
        out = 0
        for parameter in self.mlp.parameters():
            out += (parameter ** 2).sum()
        out += (self.Z_weight ** 2).sum()
        return out

    def Z_regularizer(self):

        normZ = self.normalize_Z_weight()
        return -0.5 * torch.log(normZ).sum()

    def feature_importances(self, weight_scores=True):

        with torch.no_grad():
            # Calculate weights from MLP
            if weight_scores:
                layers = list(self.mlp.named_children())
                W = layers[0][1].weight.detach().numpy().T
                for layer in layers[1:]:
                    if isinstance(layer[1], nn.ReLU):
                        continue
                    weight = layer[1].weight.detach().numpy().T
                    W = np.dot(W, weight)
                    W = W.squeeze(-1)
            else:
                W = np.ones(self.p)

            # Multiply by Z weights
            feature_imp = self.normalize_Z_weight()[0 : self.p] * W
            knockoff_imp = self.normalize_Z_weight()[self.p :] * W
            return np.concatenate([feature_imp, knockoff_imp])


def train_deeppink(
    model,
    features,
    y,
    batchsize=100,
    num_epochs=50,
    lambda1=None,
    lambda2=None,
    verbose=True,
    **kwargs,
):

    # Infer n, p, set default lambda1, lambda2
    n = features.shape[0]
    p = int(features.shape[1] / 2)
    if lambda1 is None:
        lambda1 = 10 * np.sqrt(np.log(p) / n)
    if lambda2 is None:
        lambda2 = 0

    # Batchsize can't be bigger than n
    batchsize = min(features.shape[0], batchsize)

    # Create criterion
    features, y = map(lambda x: torch.tensor(x).detach().float(), (features, y))
    if model.y_dist == "gaussian":
        criterion = nn.MSELoss(reduction="sum")
    else:
        criterion = nn.CrossEntropyLoss(reduction="sum")
        y = y.long()

    # Create optimizer
    opt = torch.optim.Adam(model.parameters(), **kwargs)

    # Loop through epochs
    for j in range(num_epochs):

        # Create batches, loop through
        batches = create_batches(features, y, batchsize=batchsize)
        predictive_loss = 0
        for Xbatch, ybatch in batches:

            # Forward pass and loss
            output = model(Xbatch)
            loss = criterion(output, ybatch.unsqueeze(-1))
            predictive_loss += loss

            # Add l1 and l2 regularization
            loss += lambda1 * model.l1norm()
            loss += lambda2 * model.l2norm()
            loss += lambda1 * model.Z_regularizer()

            # Step
            opt.zero_grad()
            loss.backward()
            opt.step()

        if verbose and j % 10 == 0:
            print(f"At epoch {j}, mean loss is {predictive_loss / n}")

    return model
