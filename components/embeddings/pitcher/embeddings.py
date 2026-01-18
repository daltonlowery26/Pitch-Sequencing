# embeddings to represent how a pitcher is precived, a prior context embedding
# %% packages
import os
import gc
import numpy as np
import polars as pl
import polars.selectors as cs
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
import ot
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
def relative_distance(batch_features, normalization_stats):
    weights_dict = {
        'vaa_diff': 0.07719401628733857,
        'haa_diff': 0.088933057497284,
        'effective_speed': 0.08040670227854997,
        'ax': 0.07495841538536129,
        'ay': 0.1593238757793289,
        'az': 0.12804355247650187,
        'arm_angle': 0.07956186876642489,
        'release_height': 0.08793159987116557,
        'release_x': 0.07079566880738072
    }
    cmd_weight = 0.15285124285066423
    
    # kmu 
    kmu_order = ['vaa_diff', 'haa_diff', 'effective_speed', 'ax', 'ay', 'az', 
                 'arm_angle', 'release_height', 'release_x']
    
    # weight tensor
    feature_weights = torch.tensor([weights_dict[k] for k in kmu_order], device='cuda', dtype=torch.double)
    k_weight_sum = feature_weights.sum().item()
    
    # covar weights
    cov_scale_diag = torch.sqrt(feature_weights)
    cov_scale_matrix = torch.diag_embed(cov_scale_diag)
    
    # matrix log
    def matrix_log(sigma):
        try:
            return SafeLog.apply(sigma)
        except NameError:
            return torch.matrix_power(sigma, 1) 
    
    # extract and reshape
    kmu = batch_features["kmu"]
    ksigma = batch_features["ksigma"]
    avg_cmd = batch_features["cmd_value"].view(-1, 1)
    
    # weighted covars based on feature importance
    ksigma_weighted = cov_scale_matrix @ ksigma @ cov_scale_matrix
    ksigma_log = matrix_log(ksigma_weighted)
    
    # weighted mean distances
    w_reshaped = feature_weights.view(1, 1, -1)
    diff_sq = (kmu.unsqueeze(1) - kmu.unsqueeze(0)) ** 2
    k_mean_dist = torch.sum(w_reshaped * diff_sq, dim=2)
    
    # covarience distance
    k_cov_dist = torch.sum(
        (ksigma_log.unsqueeze(1) - ksigma_log.unsqueeze(0)) ** 2, dim=(2, 3)
    )
    
    # combined distance
    k_dists = k_mean_dist + k_cov_dist
    
    # simple eucledian
    cmd_dists = (avg_cmd - avg_cmd.T) ** 2
    
    # normalization
    if normalization_stats is not None:
        k_mean_val, cmd_mean_val = normalization_stats
    else:
        k_mean_val = torch.mean(k_dists)
        cmd_mean_val = torch.mean(cmd_dists)
    
    # normalize on global values
    k_norm = k_dists / (k_mean_val + 1e-8)
    cmd_norm = cmd_dists / (cmd_mean_val + 1e-8)
    
    # weighted values
    total_dist_matrix = k_norm * k_weight_sum + cmd_norm * cmd_weight
    
    return total_dist_matrix

# %% triplet loss, with hard mining
class TripletLoss(nn.Module): 
    def __init__(self, lambda_scale=0.5, pos_threshold=0.2, neg_threshold=0.8):
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
        dynamic_margin = self.lambda_scale * torch.tanh(gt_neg_dists - gt_pos_dists)

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
        avg_cmd = col_to_tensor("cmd_value").view(-1, 1)  # reshape to add dims for concat
        # raw input vector
        raw_inputs = torch.cat([kmu, avg_cmd], dim=1)  # 6 + 3 + 2 = 10
        # stanardize all the inputs
        mean = raw_inputs.mean(dim=0)
        std = raw_inputs.std(dim=0)
        self.input_emb = (raw_inputs - mean) / (std + 1e-8)
        # average
        self.avg_cmd_target = avg_cmd.squeeze()
        self.kmu_target = kmu
        # load and reshape matrices
        ksigma_flat = torch.tensor(df["ksigma"].to_list(), dtype=torch.float64)
        # dim of matrices
        k_dim = int(np.sqrt(ksigma_flat.shape[1]))
        # reshape
        self.ksigma = ksigma_flat.view(-1, k_dim, k_dim)
        # sdp saftey, covar matrixs are sdp
        self.ksigma = make_spd(self.ksigma)

    def __len__(self):
        return len(self.input_emb)

    def __getitem__(self, idx):
        # return precomputed data from idx, much quicker
        return {
            "input_emb": self.input_emb[idx],
            "cmd_value": self.avg_cmd_target[idx],
            "kmu": self.kmu_target[idx],
            "ksigma": self.ksigma[idx],
        }

# %% mlp
class SiameseNet(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dims=[64, 128]):
        # inherit
        super(SiameseNet, self).__init__()
        # layers and input dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            nn.Linear(256, embedding_dim)
        )
    def forward(self, x):
        # generate embeddings
        embeddings = self.net(x)
        # normalize so model doesnt just explode embedding magnitutde
        return torch.nn.functional.normalize(embeddings, p=2, dim=1)

# %% train loop
def train_model(model, dataloader, val_loader, optimizer, criterion, device, epochs, normalization_stats):
    model.to(device)
    # training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        best_loss = np.inf
        
        # loop through data loader
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
            
            # backward pass
            if loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
        
        train_loss = total_loss / len(dataloader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_features in val_loader:
                for k, v in batch_features.items():
                    batch_features[k] = v.to(device)

                # forward pass
                embeddings = model(batch_features["input_emb"])
                
                # gt distances
                gt_distances = relative_distance(batch_features, normalization_stats)
                
                # val loss
                loss = criterion(embeddings, gt_distances)
                val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            # save best model
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save(model.state_dict(), '../models/pitch_embed.pth')
        
        print(f"Epoch [{epoch + 1}/{epochs}], ValLoss: {avg_val_loss:.6f}, TrainLoss: {train_loss:.6f}")
    
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
def get_closest_embeddings(query_emb, embedding_database, k):
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
    k_dists_list, cmd_dists_list = [], []

    for batch_features in loader:
        for k, v in batch_features.items():
            batch_features[k] = v.to(device)

        # matrix log
        def matrix_log(sigma):
            return SafeLog.apply(sigma)
        
        # same distance calculation as above
        kmu = batch_features["kmu"]
        ksigma_log = matrix_log(batch_features["ksigma"])
        avg_cmd = batch_features["cmd_value"].view(-1, 1)

        k_mean_dist = torch.sum((kmu.unsqueeze(1) - kmu.unsqueeze(0)) ** 2, dim=2)
        k_cov_dist = torch.sum((ksigma_log.unsqueeze(1) - ksigma_log.unsqueeze(0)) ** 2, dim=(2, 3))
        k_dists = k_mean_dist + k_cov_dist

        cmd_dists = (avg_cmd - avg_cmd.T) ** 2
        k_dists_list.append(k_dists.detach().view(-1))
        cmd_dists_list.append(cmd_dists.detach().view(-1))

    k_mean = torch.cat(k_dists_list).mean()
    cmd_mean = torch.cat(cmd_dists_list).mean()
    return k_mean, cmd_mean

# %% data loaders
input = pl.read_parquet("cleaned_data/embed/pitch_mu_cov.parquet")
input = input.filter(pl.col('count') > 0)
dataset = ManifoldDataset(df=input)
# train and val
train_size = int(0.80 * len(dataset))
val_size = len(dataset) - train_size
# split into train and val
train_dataset, val_dataset = random_split(
    dataset, 
    [train_size, val_size], 
    generator=torch.Generator().manual_seed(26)
)
# norm on only train data
normalization_stats = get_normalization_stats(train_dataset, device="cuda")
# dataloaders
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

# %% model params
embedding_dim = 64
model = SiameseNet(input_dim=10, embedding_dim=embedding_dim)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
criterion = TripletLoss()  # default params in method sig

# %% train, returns last iter model
trained_model = train_model(model, train_loader, val_loader, optimizer, 
                criterion, device="cuda", epochs=50, normalization_stats=normalization_stats)

# %% clean env
torch.cuda.empty_cache()
gc.collect()

# %% load saved best model
best_model = SiameseNet(input_dim=10, embedding_dim=embedding_dim)
dict = torch.load('../models/pitch_embed.pth', weights_only=True)
best_model.load_state_dict(dict)

# %% infrence and closeness
inference_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
embeddings = get_embeddings(model=best_model, dataloader=inference_loader, device="cuda")

# add preformance data
ars = pl.read_csv('cleaned_data/metrics/arsenal.csv')
input = input.join(ars, on=['pitcher_id', 'pitch_name', 'game_year'], how='left')
input = pl.concat(
    [input, pl.from_numpy(embeddings, schema=["embeds"])], 
    how="horizontal"
)

# %% group diff from means and varience
def weighted_comps(df, embeddings, metric_cols, weight_col, k):
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric='cosine', n_jobs=-1)
    nbrs.fit(embeddings)
    dists, indices = nbrs.kneighbors(embeddings)
    # exclude self
    dists = dists[:, 1:]
    indices = indices[:, 1:]
    # flatten
    n_rows = len(df)
    target_indices = np.repeat(np.arange(n_rows), k)
    neighbor_indices = indices.flatten()
    neighbor_dists = dists.flatten()
    # weighted based on cos distance
    neighbor_sims = 1 - neighbor_dists
    neighbor_sims = np.clip(neighbor_sims, a_min=0.0001, a_max=1.0)
    
    neighbors_df = pl.DataFrame({
        "orig_index": target_indices,
        "neighbor_index": neighbor_indices,
        "similarity": neighbor_sims
    })

    df_indexed = df.with_row_index("row_id")
    target_stats = df_indexed.select(
        [pl.col("row_id").alias("orig_index")] + 
        [pl.col(m).alias(f"{m}_target") for m in metric_cols]
    )
    
    neighbor_stats = df_indexed.select(
        [pl.col("row_id").alias("neighbor_index"), pl.col(weight_col).alias("w_count")] + 
        [pl.col(m).alias(f"{m}_neighbor") for m in metric_cols]
    )

    joined = (
        neighbors_df
        .join(target_stats, on="orig_index")
        .join(neighbor_stats, on="neighbor_index")
    )
    agg_exprs = []
    
    
    for m in metric_cols:
        x_i = pl.col(f"{m}_neighbor")
        x_t = pl.col(f"{m}_target")
        w_total = pl.col("similarity") # * pl.col("w_count") 
        w_mean_expr = (x_i * w_total).sum() / w_total.sum()
        w_diff = (w_mean_expr - x_t.first()).abs() 
        w_var = ((w_total * (x_i - w_mean_expr).pow(2)).sum() / w_total.sum()).sqrt()
        agg_exprs.append(w_diff.alias(f"{m}_weighted_diff"))
        agg_exprs.append(w_var.alias(f"{m}_weighted_std"))

    result = joined.group_by("orig_index").agg(agg_exprs).sort("orig_index")

    # stats
    return result

# %% embedding result preformance
embed_pref = weighted_comps(input, embeddings, metric_cols=['rv_100', 'xwoba', 'whiff_percent'], weight_col='pitches', k=25)
for columns in embed_pref.columns:
    print(f'{columns} mean: {embed_pref[columns].mean()}')

# %% refrence
for columns in input.select(cs.numeric()).columns:
    print(f'{columns} std: {input[columns].std()}')

# %% tsne visual tool
def tsne_viz(color_col):
    color_col = color_col
    # tsne dim reduction
    reducer = TSNE(n_components=2, metric='cosine', init='pca', learning_rate='auto', random_state=42)
    tsne_coords = reducer.fit_transform(embeddings)
    # raw data with info
    plot_data = input.select(pl.col(color_col)).to_pandas()
    plot_data['UMAP_1'] = tsne_coords[:, 0]
    plot_data['UMAP_2'] = tsne_coords[:, 1]
    # plot
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
    data=plot_data, x='UMAP_1',y='UMAP_2',hue=color_col,
    alpha=0.6,s=15)
    plt.title(f"UMAP Projection of Embeddings (Colored by {color_col})")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title=color_col)
    plt.tight_layout()
    plt.show()

# %% throws
tsne_viz('p_throws')

# %% tsne by pitch type
types = {
    'fb': ['4-Seam Fastball'],
    'sink':['Sinker'],
    'cut': ['Cutter'],
    'curve': ['Curveball', 'Knuckle Curve'],
    'sweeper':['Sweeper'],
    'slide': ['Slider', 'Slurve'],
    'off': ['Split-Finger', 'Changeup', 'Forkball'],
    'knuckle': ['Knuckleball']
}
mapping = {v: k for k, values in types.items() for v in values}
input = input.with_columns(
    pitch_group = pl.col("pitch_name").replace(mapping)
)
tsne_viz('pitch_group')

# %% run value
tsne_viz('rv_100')

# %% use wassermans metric and optimal transport to calculate areseal similairty
def arsenal_distance(emb_A, usage_A, emb_B, usage_B):
    # cost matrix
    M = cdist(emb_A, emb_B, metric='cosine')
    # solve optimal transport problem
    return ot.emd2(usage_A, usage_B, M)

def similar_pitchers(df_pitchers, k=20):
    # polars to search schema 
    ids = df_pitchers["p_season_id"].to_list()
    # float 32
    embeds_list = [np.vstack(e).astype(np.float32) for e in df_pitchers["embeds"]]
    usage_list = [np.array(u, dtype=np.float32) for u in df_pitchers["usage"]]

    # weighted avg embeddng for each pitcher
    centroids = np.array([
        np.average(e, axis=0, weights=u) 
        for e, u in zip(embeds_list, usage_list)
    ])

    # cosine distance between all to narrow search
    centroid_dists = cdist(centroids, centroids, metric='cosine')
    
    results = {}
    for i in range(len(ids)):
        # take top k canaidate
        candidate_indices = np.argsort(centroid_dists[i])[1 : k + 1]
        
        best_match_id = None
        min_ot_dist = float('inf')

        # run only on canidates
        for j in candidate_indices:
            dist = arsenal_distance(
                embeds_list[i], usage_list[i],
                embeds_list[j], usage_list[j]
            )
            # return clostest
            if dist < min_ot_dist:
                min_ot_dist = dist
                best_match_id = ids[j]
        # add best match
        results[ids[i]] = (best_match_id, min_ot_dist)
    return results
    
# %% closest ares.
input = input.with_columns(
    p_season_id = pl.concat_str(
            [pl.col("pitcher_id"), pl.col("game_year")], 
            separator="-"
        )
)
ars = (input.lazy()
        .filter(pl.col("percent").is_not_null())
        .group_by(["p_season_id"])
        .agg([
            pl.col("pitcher_name"),
            pl.col("embeds"), 
            (pl.col("percent") / pl.col("percent").sum()).alias("usage") # normalize usage
        ])
        .collect(engine="streaming"))
sim = similar_pitchers(ars, k=5000)

# %% match to input df
ids = list(sim.keys())
matched_ids = [val[0] for val in sim.values()]
scores = [val[1] for val in sim.values()]

# lookup df
lookup = pl.DataFrame({
    "current_id": ids,
    "match_id": matched_ids,
    "match_score": scores
})
# names
lookup = lookup.join(input.select(['pitcher_name', 'p_season_id']), left_on=['match_id'], right_on=['p_season_id'], suffix='_match')
lookup = lookup.join(input.select(['pitcher_name', 'p_season_id']), left_on=['current_id'], right_on=['p_season_id'], suffix='_current')
lookup = lookup.unique()
lookup.write_csv('pitcher_pairs.csv')
