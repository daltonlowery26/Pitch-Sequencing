# %% packages
import os
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.manifold import TSNE
os.chdir("C:/Users/dalto/OneDrive/Pictures/Documents/Projects/Coding Projects/Optimal Pitch/data/")

# %% numerical stability utlis arising from handling of covar matrices
class SafeLog(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sigma):
        # standard log matrix
        L, U = torch.linalg.eigh(sigma)
        # clamp eigen to avoid log 0
        L = torch.clamp(L, min=1e-8)
        L_log = torch.log(L)
        ctx.save_for_backward(L, U, L_log)  # check if safe for gradient descent
        return U @ torch.diag_embed(L_log) @ U.transpose(-2, -1)

    @staticmethod
    def backward(ctx, grad_output):
        L, U, L_log = ctx.saved_tensors
        # denominators of matrix (eigenvalues)
        L_i = L.unsqueeze(-1)
        L_j = L.unsqueeze(-2)
        L_diff = L_i - L_j
        # similairty mask, do we need do take the limit for the backward pass
        is_close = torch.abs(L_diff) < 1e-6
        # distinict eigenvalues case
        L_diff_safe = torch.where(is_close, torch.ones_like(L_diff), L_diff)
        log_i = L_log.unsqueeze(-1)
        log_j = L_log.unsqueeze(-2)
        P_distinct = (log_i - log_j) / L_diff_safe
        # repeated eigenvalues case
        inv_L = 1.0 / L
        inv_L_i = inv_L.unsqueeze(-1)
        inv_L_j = inv_L.unsqueeze(-2)
        P_limit = (inv_L_i + inv_L_j) / 2.0
        # use distinict when far limit withn close
        P = torch.where(is_close, P_limit, P_distinct)
        # find gradient, for backprop (Daleckii-Krein formula)
        U_T = U.transpose(-2, -1)
        grad_sym = U_T @ grad_output @ U
        grad_sigma = U @ (P * grad_sym) @ U_T
        # enforce that the matrix is symmetirc, covar matrices must be sdp
        return (grad_sigma + grad_sigma.transpose(-2, -1)) / 2.0


def make_spd(matrix, eps=1e-6):
    # project to ensure matrix is symmetric, postive, definite
    sym = (matrix + matrix.transpose(-2, -1)) / 2
    # add jiggle to axis to enure positivity
    eye = torch.eye(matrix.shape[-1], device=matrix.device, dtype=matrix.dtype)
    return sym + (eye * eps)


# %% efficent distance
def relative_distance(batch_features, normalization_stats=None):
    # take log of matrix, project curved manifold into eucledian space
    def matrix_log(sigma):
        return SafeLog.apply(sigma)

    # extract and reshape
    rmu = batch_features["rmu"]
    rsigma_log = matrix_log(batch_features["rsigma"])
    kmu = batch_features["kmu"]
    ksigma_log = matrix_log(batch_features["ksigma"])
    avg_cmd = batch_features["avg_command"].view(-1, 1)  # add dim to mantain shape
    std_cmd = batch_features["std_command"].view(-1, 1)

    # cannot use full wass. sqrt of matrix are not paralizable
    # release
    r_mean_dist = torch.sum(
        (rmu.unsqueeze(1) - rmu.unsqueeze(0)) ** 2, dim=2
    )  # dist between means
    r_cov_dist = torch.sum(
        (rsigma_log.unsqueeze(1) - rsigma_log.unsqueeze(0)) ** 2, dim=(2, 3)
    )
    r_dists = r_mean_dist + r_cov_dist

    # kinematics
    k_mean_dist = torch.sum((kmu.unsqueeze(1) - kmu.unsqueeze(0)) ** 2, dim=2)
    k_cov_dist = torch.sum(
        (ksigma_log.unsqueeze(1) - ksigma_log.unsqueeze(0)) ** 2, dim=(2, 3)
    )
    k_dists = k_mean_dist + k_cov_dist

    # eucledian dist for command
    cmd_dists = (avg_cmd - avg_cmd.T) ** 2 + (std_cmd - std_cmd.T) ** 2

    # normalization based on scale of values
    if normalization_stats is not None:
        k_mean_val, r_mean_val, cmd_mean_val = normalization_stats
    else:
        k_mean_val = torch.mean(k_dists)
        r_mean_val = torch.mean(r_dists)
        cmd_mean_val = torch.mean(cmd_dists)

    # normalize and add epsilion to prevent div by 0
    k_norm = k_dists / (k_mean_val + 1e-8)
    r_norm = r_dists / (r_mean_val + 1e-8)
    cmd_norm = cmd_dists / (cmd_mean_val + 1e-8)

    # weighted sum of componenet based on features
    total_dist_matrix = (k_norm * 0.4411) + (r_norm * 0.2153) + (cmd_norm * 0.3435)

    return total_dist_matrix


# %% triplet loss, with hard mining
class TripletLoss(nn.Module):  # lambda_scale might need to be tuned
    def __init__(self, lambda_scale=0.25, pos_threshold=0.2, neg_threshold=0.8):
        super(TripletLoss, self).__init__()
        self.lambda_scale = lambda_scale  # gt units to embedding units
        self.pos_p = pos_threshold  # take top x clostest
        self.neg_p = neg_threshold  # take bottom 1-x as negs

    def _pairwise_distance(self, embeddings):
        # distance between all embeddings
        dot_product = torch.matmul(embeddings, embeddings.t())
        square_norm = torch.diag(dot_product)
        distances = (
            square_norm.unsqueeze(1) - 2.0 * dot_product + square_norm.unsqueeze(0)
        )
        # clamp only to postive values
        distances = torch.clamp(distances, min=0.0)
        return torch.sqrt(distances + 1e-16)

    def forward(self, embeddings, gt_distances):
        # find distances between all embeddings
        emb_dists = self._pairwise_distance(embeddings)

        # batch size, find pos and neg
        batch_size = embeddings.size(0)

        # take postive and neg embeddings
        flat_gt = gt_distances.view(-1)
        pos_cutoff = torch.quantile(flat_gt, self.pos_p)
        neg_cutoff = torch.quantile(flat_gt, self.neg_p)

        # postive mask
        mask_pos = (gt_distances < pos_cutoff) & ~torch.eye(
            batch_size, device=embeddings.device
        ).bool()

        # negative mask
        mask_neg = gt_distances > neg_cutoff

        # find hardest postive
        pos_search_matrix = emb_dists.clone()
        pos_search_matrix[~mask_pos] = -1.0
        hardest_pos_dists, hardest_pos_indices = torch.max(pos_search_matrix, dim=1)

        # find hardest negative
        neg_search_matrix = emb_dists.clone()
        neg_search_matrix[~mask_neg] = float("inf")
        hardest_neg_dists, hardest_neg_indices = torch.min(neg_search_matrix, dim=1)

        # distance in ground truth
        gt_pos_dists = torch.gather(
            gt_distances, 1, hardest_pos_indices.unsqueeze(1)
        ).squeeze()
        gt_neg_dists = torch.gather(
            gt_distances, 1, hardest_neg_indices.unsqueeze(1)
        ).squeeze()

        # margin based on dists
        dynamic_margin = self.lambda_scale * (gt_neg_dists - gt_pos_dists)

        # loss function, with dynamic margin. weight heavily if confident in distance, zero out if close
        loss = torch.clamp(
            hardest_pos_dists - hardest_neg_dists + dynamic_margin, min=0.0
        )

        # only keep valid triplets
        valid_triplets = (mask_pos.sum(1) > 0) & (mask_neg.sum(1) > 0)
        return loss[valid_triplets].mean()  # average loss for triplets for every target


# %% data loader & dataset
class ManifoldDataset(Dataset):
    def __init__(self, df):
        # cols to tensor
        def col_to_tensor(col_name):
            return torch.tensor(df[col_name].to_list(), dtype=torch.float32)

        # input features
        kmu = col_to_tensor("kmu")
        rmu = col_to_tensor("rmu")
        avg_cmd = col_to_tensor("avg_command").view(
            -1, 1
        )  # reshape to add dims for concat
        std_cmd = col_to_tensor("std_command").view(-1, 1)
        # raw input vector
        raw_inputs = torch.cat([kmu, rmu, avg_cmd, std_cmd], dim=1)  # 6 + 3 + 2 = 11
        # stanardize all the inputs
        mean = raw_inputs.mean(dim=0)
        std = raw_inputs.std(dim=0)
        self.input_emb = (raw_inputs - mean) / (std + 1e-8)
        # average
        self.avg_cmd_target = avg_cmd.squeeze()
        self.std_cmd_target = std_cmd.squeeze()
        self.kmu_target = kmu
        self.rmu_target = rmu
        # load and reshape matrices
        rsigma_flat = torch.tensor(df["rsigma"].to_list(), dtype=torch.float64)
        ksigma_flat = torch.tensor(df["ksigma"].to_list(), dtype=torch.float64)
        # dim of matrices
        r_dim = int(np.sqrt(rsigma_flat.shape[1]))
        k_dim = int(np.sqrt(ksigma_flat.shape[1]))
        # reshape
        self.rsigma = rsigma_flat.view(-1, r_dim, r_dim)
        self.ksigma = ksigma_flat.view(-1, k_dim, k_dim)
        # sdp saftey, covar matrixs are sdp
        self.rsigma = make_spd(self.rsigma)
        self.ksigma = make_spd(self.ksigma)

    def __len__(self):
        return len(self.input_emb)

    def __getitem__(self, idx):
        # return precomputed data from idx, much quicker
        return {
            "input_emb": self.input_emb[idx],
            "avg_command": self.avg_cmd_target[idx],
            "std_command": self.std_cmd_target[idx],
            "rmu": self.rmu_target[idx],
            "kmu": self.kmu_target[idx],
            "rsigma": self.rsigma[idx],
            "ksigma": self.ksigma[idx],
        }

# %% residual mlp
class SiameseNet(nn.Module):
    def __init__(self, input_dim, embedding_dim=32, hidden_dims=[64, 128]):
        # inherit
        super(SiameseNet, self).__init__()
        # layers and input dim
        layers = []
        curr_dim = input_dim
        # simple fcn
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.2))
            curr_dim = h_dim

        # projection layer
        layers.append(nn.Linear(curr_dim, embedding_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # generate embeddings
        embeddings = self.net(x)
        # normalize so model doesnt just explode embedding magnitutde
        return torch.nn.functional.normalize(embeddings, p=2, dim=1)


# %% train loop
def train_model(model, dataloader, optimizer, criterion, device, epochs, normalization_stats):
    model.to(device)
    model.train()
    # training loop
    for epoch in range(epochs):
        total_loss = 0.0
        best_loss = np.inf
        for batch_idx, batch_features in enumerate(dataloader):
            # batch features
            for k, v in batch_features.items():
                batch_features[k] = v.to(device)
            optimizer.zero_grad()
            # forward pass create
            embeddings = model(batch_features["input_emb"])
            # ground truth distances, based on gt distance
            with torch.no_grad():
                gt_distances = relative_distance(batch_features, normalization_stats)
            # triplet loss
            loss = criterion(embeddings, gt_distances)
            # save best model
            if loss < best_loss:
                best_loss = loss
                torch.save(model.state_dict(), '../models/pitch_embed.pth')
            # backward pass
            if loss.requires_grad:
                loss.backward()
                # gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.6f}")
    return model

# %% embeddings extracting
def get_embeddings(model, dataloader, device):
    model.eval()
    model.to(device)
    all_embeddings = []
    # forward pass
    with torch.no_grad():
        for batch_features in dataloader:
            # input to device
            input_data = batch_features["input_emb"].to(device)
            emb = model(input_data)
            # detatch embeddings, to numpy
            all_embeddings.append(emb.cpu().numpy())

    # concate all embeddings together
    return np.vstack(all_embeddings)

# %% find 5 clostest embeddings
def get_closest_embeddings(query_emb, embedding_database, k=5):
    # correct dim
    if query_emb.ndim == 1:
        query_emb = query_emb[np.newaxis, :]
    # normalize for cos sim
    query_norm = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)
    db_norm = embedding_database / np.linalg.norm(
        embedding_database, axis=1, keepdims=True
    )
    # dot product
    scores = np.dot(query_norm, db_norm.T).squeeze()
    # topk
    top_k_indices = np.argpartition(scores, -k)[-k:]
    # sort the top k
    top_k_indices = top_k_indices[np.argsort(scores[top_k_indices])[::-1]]
    top_k_scores = scores[top_k_indices]
    return top_k_indices, top_k_scores

# %% normalize based on gloabl statitics, not just batch subset
def get_normalization_stats(dataset, device="cuda"):
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)
    # dists for global stats =
    k_dists_list, r_dists_list, cmd_dists_list = [], [], []

    for batch_features in loader:
        for k, v in batch_features.items():
            batch_features[k] = v.to(device)

        # matrix log
        def matrix_log(sigma):
            return SafeLog.apply(sigma)
        
        # same distance calculation as above
        rmu = batch_features["rmu"]
        rsigma_log = matrix_log(batch_features["rsigma"])
        kmu = batch_features["kmu"]
        ksigma_log = matrix_log(batch_features["ksigma"])
        avg_cmd = batch_features["avg_command"].view(-1, 1)
        std_cmd = batch_features["std_command"].view(-1, 1)

        r_mean_dist = torch.sum((rmu.unsqueeze(1) - rmu.unsqueeze(0)) ** 2, dim=2)
        r_cov_dist = torch.sum((rsigma_log.unsqueeze(1) - rsigma_log.unsqueeze(0)) ** 2, dim=(2, 3))
        r_dists = r_mean_dist + r_cov_dist

        k_mean_dist = torch.sum((kmu.unsqueeze(1) - kmu.unsqueeze(0)) ** 2, dim=2)
        k_cov_dist = torch.sum((ksigma_log.unsqueeze(1) - ksigma_log.unsqueeze(0)) ** 2, dim=(2, 3))
        k_dists = k_mean_dist + k_cov_dist

        cmd_dists = (avg_cmd - avg_cmd.T) ** 2 + (std_cmd - std_cmd.T) ** 2
        k_dists_list.append(k_dists.detach().view(-1))
        r_dists_list.append(r_dists.detach().view(-1))
        cmd_dists_list.append(cmd_dists.detach().view(-1))

    k_mean = torch.cat(k_dists_list).mean()
    r_mean = torch.cat(r_dists_list).mean()
    cmd_mean = torch.cat(cmd_dists_list).mean()
    return k_mean, r_mean, cmd_mean

# %% model prep
# data loaders
input = pl.read_parquet("cleaned_data/metrics/pitch_mu_cov.parquet")
dataset = ManifoldDataset(df=input)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True)
normalization_stats = get_normalization_stats(dataset, device="cuda") # norm. stats

# model params
embedding_dim = 64
model = SiameseNet(input_dim=11, embedding_dim=embedding_dim)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
criterion = TripletLoss()  # default params in method sig

# %% train, returns last iter model
trained_model = train_model(model, dataloader, optimizer, criterion, device="cuda", epochs=100, normalization_stats=normalization_stats)

# %% nearest embeddings 
best_model = SiameseNet(input_dim=11, embedding_dim=embedding_dim)
dict = torch.load('../models/pitch_embed.pth')
best_model.load_state_dict(dict)

# infrence and closeness
inference_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
embeddings = get_embeddings(model=trained_model, dataloader=inference_loader, device="cuda")
indices = (
    input.with_row_index(name="orig_index")
    .filter(
        (pl.col("pitcher_name") == "Cabrera, Edward")
        & (pl.col("pitch_name") == "Changeup")
        & (pl.col("game_year") == 2025)
    )
    .get_column("orig_index"))
indices = indices.item() # detach get just int index

# get 5 closest embeddings by distance
index, _ = get_closest_embeddings(embeddings[indices], embeddings)
players = (input.with_row_index(name="orig_index")
    .filter(pl.col("orig_index").is_in(index))
    .select(["pitcher_name", "pitch_name", "game_year"]))
print(players)
# %% tsne embeddings visuals
color_col = pl.col('pitch_name')
reducer = TSNE(n_components=2, metric='cosine', init='pca', learning_rate='auto', random_state=42)
tsne_coords = reducer.fit_transform(embeddings)
# raw data with info
plot_data = input.select(color_col).to_pandas()
plot_data['UMAP_1'] = tsne_coords[:, 0]
plot_data['UMAP_2'] = tsne_coords[:, 1]

# plot
plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=plot_data, x='UMAP_1',y='UMAP_2',hue=color_col,
    alpha=0.6,s=15,palette='viridis')
plt.title(f"UMAP Projection of Embeddings (Colored by {color_col})")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title=color_col)
plt.tight_layout()
plt.show()
