import numpy as np
import torch
import torch.nn as nn

from .. import utilities


def create_batches(features, y, batchsize):
    # Create random indices to reorder datapoints
    n = features.shape[0]
    features.shape[1]
    inds = torch.randperm(n)

    # Iterate through and create batches
    i = 0
    batches = []
    while i < n:
        batches.append([features[inds][i : i + batchsize], y[inds][i : i + batchsize]])
        i += batchsize
    return batches


class DeepPinkModel(nn.Module):
    def __init__(self, p, hidden_sizes=[64], y_dist="gaussian", normalize_Z=True):
        """
        Adapted from https://arxiv.org/pdf/1809.01185.pdf.

        The module has two components:
        1. A sparse linear layer with dimension 2*p to p.
        However, there are only 2*p weights (each feature
        and knockoff points only to their own unique node).
        This is (maybe?) followed by a ReLU activation.
        2. A multilayer perceptron (MLP)

        Parameters
        ----------
        p : int
            The dimensionality of the data
        hidden_sizes: list
            A list of hidden sizes for the mlp layer(s).
            Defaults to [64].
        normalize_Z : bool
            If True, the first sparse linear layer is normalized
            so the weights for each feature/knockoff pair have an
            l1 norm of 1. This can modestly improve power in some
            settings.
        """

        super().__init__()

        # Initialize weight for first layer
        self.p = p
        self.y_dist = y_dist
        self.Z_weight = nn.Parameter(torch.ones(2 * p))
        self.norm_Z_weight = normalize_Z

        # Save indices/reverse indices to prevent violations of FDR control
        self.inds, self.rev_inds = utilities.random_permutation_inds(2 * p)
        self.feature_inds = self.rev_inds[0 : self.p]
        self.ko_inds = self.rev_inds[self.p :]

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

    def _fetch_Z_weight(self):
        # Possibly don't normalize
        if not self.norm_Z_weight:
            return self.Z_weight

        # Else normalize, first construct denominator
        normalizer = torch.abs(self.Z_weight[self.feature_inds]) + torch.abs(
            self.Z_weight[self.ko_inds]
        )
        # Normalize
        Z = torch.abs(self.Z_weight[self.feature_inds]) / normalizer
        Ztilde = torch.abs(self.Z_weight[self.ko_inds]) / normalizer
        # Concatenate and reshuffle
        return torch.cat([Z, Ztilde], dim=0)[self.inds]

    def forward(self, features):
        """
        Note: features are now shuffled
        """

        # First layer: pairwise weights (and sum)
        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features).float()
        features = features[:, self.inds]  # shuffle features to prevent FDR violations
        features = self._fetch_Z_weight().unsqueeze(dim=0) * features
        features = features[:, self.feature_inds] - features[:, self.ko_inds]

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
            out += (parameter**2).sum()
        out += (self.Z_weight**2).sum()
        return out

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
            Z = self._fetch_Z_weight().numpy()
            feature_imp = Z[self.feature_inds] * W
            knockoff_imp = Z[self.ko_inds] * W
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

            # Step
            opt.zero_grad()
            loss.backward()
            opt.step()

        if verbose and j % 10 == 0:
            print(f"At epoch {j}, mean loss is {predictive_loss / n}")

    return model
