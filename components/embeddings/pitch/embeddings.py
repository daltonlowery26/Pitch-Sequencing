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
os.chdir("C:/Users/dalto/OneDrive/Pictures/Documents/Projects/Coding Projects/Optimal Pitch/data/")

# TODO: Probablitistic distances based on the quanity of the class, double size of embedding for var and mu of point

# %% efficent distance
def relative_distance(batch_features, normalization_stats):
    # feature weights
    weights_dict = {}
    
    # extract and reshape
    kmu = batch_features["kmu"]
    kmu_order = ['hra', 'vra', 'effective_speed', 'arm_angle', 'release_height',
                'release_x', 'deltax', 'deltaz', 'ay']
    
    # weight tensor
    feature_weights = torch.tensor([weights_dict[k] for k in kmu_order], device='cuda', dtype=torch.double)
    
    # simple eucledian distance
    w_reshaped = feature_weights.view(1, 1, -1)
    diff_sq = (kmu.unsqueeze(1) - kmu.unsqueeze(0)) ** 2
    k_mean_dist = torch.sum(w_reshaped * diff_sq, dim=2)
    
    # normalize and add epsilion to prevent div by 0
    k_norm = k_mean_dist / (normalization_stats + 1e-8)
    
    return k_norm

# %% triplet loss, with hard mining
class TripletLoss(nn.Module): 
    def __init__(self, lambda_scale=1, pos_threshold=0.2, neg_threshold=0.8):
        super(TripletLoss, self).__init__()
        self.lambda_scale = lambda_scale  # gt units to embedding units
        self.pos_p = pos_threshold  # take top x clostest
        self.neg_p = neg_threshold  # take bottom 1-x as negs
    # deperciated
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
        
    def _probalistic_distance(self, embeddings):
            # get mus and sigma from embeddings
            dim = embeddings.size(1) // 2
            mu = embeddings[:, :dim]
            
            # ensure std and numerical 
            raw_sigma = embeddings[:, dim:]
            sigma = nn.functional.softplus(raw_sigma) + 1e-6 
    
            # eucledian dist between means
            mu_dot = torch.matmul(mu, mu.t())
            mu_sq_norm = torch.diag(mu_dot)
            dist_sq_mu = (
                mu_sq_norm.unsqueeze(1) - 2.0 * mu_dot + mu_sq_norm.unsqueeze(0)
            )
            dist_sq_mu = torch.clamp(dist_sq_mu, min=0.0)
    
            # squared eucledian dist for sigmas
            sigma_dot = torch.matmul(sigma, sigma.t())
            sigma_sq_norm = torch.diag(sigma_dot)
            dist_sq_sigma = (sigma_sq_norm.unsqueeze(1) - 2.0 * sigma_dot + sigma_sq_norm.unsqueeze(0))
            dist_sq_sigma = torch.clamp(dist_sq_sigma, min=0.0)
    
            # wasserstein distance 
            wasserstein_dist = torch.sqrt(dist_sq_mu + dist_sq_sigma + 1e-16)
            
            return wasserstein_dist

    def forward(self, embeddings, gt_distances):
        # find distances between all embeddings
        emb_dists = self._probalistic_distance(embeddings)

        # batch size, find pos and neg
        batch_size = embeddings.size(0)

        # take postive and neg embeddings
        flat_gt = gt_distances.view(-1)
        pos_cutoff = torch.quantile(flat_gt, self.pos_p)
        neg_cutoff = torch.quantile(flat_gt, self.neg_p)

        # postive mask
        mask_pos = (gt_distances < pos_cutoff) & ~torch.eye(
            batch_size, device=embeddings.device).bool()

        # negative mask
        mask_neg = gt_distances > neg_cutoff

        # find hardest postive
        pos_search_matrix = emb_dists.clone()
        pos_search_matrix[~mask_pos] = -float("inf")
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
        avg_cmd = col_to_tensor("cmd_value").view(
            -1, 1
        )  # reshape to add dims for concat
        # raw input vector
        raw_inputs = torch.cat([kmu, avg_cmd], dim=1)  # 6 + 3 + 2 = 10
        # stanardize all the inputs
        mean = raw_inputs.mean(dim=0)
        std = raw_inputs.std(dim=0)
        self.input_emb = (raw_inputs - mean) / (std + 1e-8)
        # average
        self.avg_cmd_target = avg_cmd.squeeze()
        self.kmu_target = kmu

    def __len__(self):
        return len(self.input_emb)

    def __getitem__(self, idx):
        # return precomputed data from idx, much quicker
        return {
            "input_emb": self.input_emb[idx],
            "cmd_value": self.avg_cmd_target[idx],
            "kmu": self.kmu_target[idx]
        }

# %% mlp
class SiameseNet(nn.Module):
    def __init__(self, input_dim, embedding_dim=32, hidden_dims=[64, 128]):
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
    # dists for global stats 
    k_dists_list = []
    for batch_features in loader:
        for k, v in batch_features.items():
            batch_features[k] = v.to(device)
        # same distance calculation as above
        kmu = batch_features["kmu"]
        k_mean_dist = torch.sum((kmu.unsqueeze(1) - kmu.unsqueeze(0)) ** 2, dim=2)
        k_dists = k_mean_dist
        k_dists_list.append(k_dists.detach().view(-1))
    k_mean = torch.cat(k_dists_list).mean()
    return k_mean

# %% loading input
input = pl.read_parquet('cleaned_data/embed/pitch.parquet')
input = input.with_columns(
    kmu = pl.concat_arr(pl.col(['hra', 'vra', 'effective_speed', 'arm_angle', 
                            'release_height', 'release_x', 'deltax', 'deltaz', 'ay']))
)

# %% data loaders
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
dict = torch.load('../models/pitcher_embed.pth', weights_only=True)
best_model.load_state_dict(dict)

# %% infrence and closeness
inference_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
embeddings = get_embeddings(model=best_model, dataloader=inference_loader, device="cuda")

# embeddings
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
embed_pref = weighted_comps(input, embeddings, metric_cols=['pitch_value','delta_run_exp'], weight_col='pitches', k=10)
for columns in embed_pref.columns:
    print(f'{columns} mean: {embed_pref[columns].mean()}')
# refrence
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