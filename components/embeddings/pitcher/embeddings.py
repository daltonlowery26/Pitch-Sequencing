# embeddings to represent how a pitcher is precived, a prior context embedding
# %% packages
import os
import umap
import gc
import numpy as np
import polars as pl
import polars.selectors as cs
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from scipy.spatial.distance import cdist
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
    weights_dict = {'vaa_diff': np.float32(0.07331265), 'haa_diff': np.float32(0.07780298), 'release_speed': np.float32(0.08941311),
        'release_extension': np.float32(0.090783864), 'ax': np.float32(0.09481193), 'ay': np.float32(0.09528503), 'az': np.float32(0.104321584), 
        'arm_angle': np.float32(0.088479534), 'release_height': np.float32(0.08340817), 'release_x': np.float32(0.088434786)}
    cmd_weight = 0.11394637
    
    # kmu 
    kmu_order = ['vaa_diff', 'haa_diff', 'release_speed', 'release_extension', 'ax', 'ay', 'az', 
                 'arm_angle', 'release_height', 'release_x']
    
    # weight tensor, reorder to same as cols
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
    def __init__(self, lambda_scale=1, pos_threshold=0.2, neg_threshold=0.8):
        super(TripletLoss, self).__init__()
        self.lambda_scale = lambda_scale
        self.pos_p = pos_threshold
        self.neg_p = neg_threshold

    def forward(self, embeddings, gt_distances):
        # mean and var embeddings
        emb_dim = embeddings.shape[1] // 2
        mu = embeddings[:, :emb_dim]
        sigma = embeddings[:, emb_dim:]
        
        # sigma to var for MLS distance
        var = sigma.pow(2) + 1e-6

        # formula: (mu_i - mu_j)^2 / (var_i + var_j) + log(var_i + var_j)
        
        # varience sum 
        var_sum = var.unsqueeze(1) + var.unsqueeze(0)
        
        # diff inb means
        mu_diff_sq = (mu.unsqueeze(1) - mu.unsqueeze(0)).pow(2)
        
        # weighted dist based on var
        term1 = mu_diff_sq / var_sum
        
        # uncertainty pen
        term2 = torch.log(var_sum)
        
        # scalar distance per pair
        mls_distances = torch.sum(term1 + term2, dim=2)

        # triplet selection, neg and pos percentiles and masks
        batch_size = embeddings.size(0)
        flat_gt = gt_distances.view(-1)
        pos_cutoff = torch.quantile(flat_gt, self.pos_p)
        neg_cutoff = torch.quantile(flat_gt, self.neg_p)

        mask_pos = (gt_distances < pos_cutoff) & ~torch.eye(batch_size, device=embeddings.device).bool()
        mask_neg = gt_distances > neg_cutoff

        # positve triplet
        pos_search = mls_distances.clone()
        pos_search[~mask_pos] = -float("inf") 
        hardest_pos_dists, hardest_pos_indices = torch.max(pos_search, dim=1)
        
        # negative triplet
        neg_search = mls_distances.clone()
        neg_search[~mask_neg] = float("inf") 
        hardest_neg_dists, hardest_neg_indices = torch.min(neg_search, dim=1)

        # dynamic margin, based on dist and loss function
        gt_pos_dists = torch.gather(gt_distances, 1, hardest_pos_indices.unsqueeze(1)).squeeze()
        gt_neg_dists = torch.gather(gt_distances, 1, hardest_neg_indices.unsqueeze(1)).squeeze()
        dynamic_margin = self.lambda_scale * torch.tanh(gt_neg_dists - gt_pos_dists)
        loss = torch.relu(hardest_pos_dists - hardest_neg_dists + dynamic_margin)

        # ensure valid pairs exist
        valid_triplets = (mask_pos.sum(1) > 0) & (mask_neg.sum(1) > 0)
        
        # if loss is 0
        if valid_triplets.sum() == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        return loss[valid_triplets].mean()

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
    def __init__(self, input_dim, embedding_dim):
        super(SiameseNet, self).__init__()
        self.embedding_dim = embedding_dim
        
        # feature extractor
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        # mu head
        self.mu_head = nn.Sequential(
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )

        # sigma head
        self.sigma_head = nn.Sequential(
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # need to intialize weights for sigma 
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
    
        # override last layer so init 
        final_sigma_layer = self.sigma_head[-1]
        if isinstance(final_sigma_layer, nn.Linear):
            # small weights so bias dominates
            nn.init.normal_(final_sigma_layer.weight, mean=0, std=0.001)
            nn.init.constant_(final_sigma_layer.bias, -3.0)

    def forward(self, x):
        # extract shared features
        shared_features = self.backbone(x)
        # mean
        mu = self.mu_head(shared_features)
        mu = torch.nn.functional.normalize(mu, p=2, dim=1)
        
        # sigma, softplus in loss function handles
        raw_sigma = self.sigma_head(shared_features)
        sigma = nn.functional.softplus(raw_sigma) + 1e-6 
        
        # return full embedding
        return torch.cat([mu, sigma], dim=1)

# %% train loop
def train_model(model, dataloader, val_loader, optimizer, criterion, device, epochs, normalization_stats):
    model.to(device)
    # training loop
    best_loss = np.inf
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_sigma_mag = 0.0
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
            # find sigma val so model doesnt just explode sigma
            dim = embeddings.shape[1] // 2
            sigma_val = embeddings[:, dim:]
            sigma_loss = (sigma_val.mean().item()) * 0.1
            
            # triplet loss
            loss = criterion(embeddings, gt_distances)
            loss = loss + sigma_loss
            # backward pass
            if loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
                total_sigma_mag += sigma_val.mean().item()
        
        # avg train and sigma, to ensure sigma isnt exploded
        train_loss = total_loss / len(dataloader)
        sigma = total_sigma_mag / len(dataloader)

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
                torch.save(model.state_dict(), '../models/pitcher_embed.pth')
        
        print(f"Epoch [{epoch + 1}/{epochs}], ValLoss: {avg_val_loss:.6f}, TrainLoss: {train_loss:.6f}, sigma: {sigma:.6f}")
    
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
        
    #  split my and sigma
    dim = query_emb.shape[1] // 2
    q_mu = query_emb[:, :dim]
    q_sigma = query_emb[:, dim:]
    db_mu = embedding_database[:, :dim]
    db_sigma = embedding_database[:, dim:]
    # sigma to var, epsilon for stability
    q_var = np.square(q_sigma) + 1e-9
    db_var = np.square(db_sigma) + 1e-9
    # sum of variences
    var_sum = q_var + db_var
    # sqaured mean var
    diff_sq = np.square(q_mu - db_mu)
    # mahaoblins distance
    term1 = diff_sq / var_sum
    # penalize high uncertanityt
    term2 = np.log(var_sum)
    # total distance
    mls_distances = np.sum(term1 + term2, axis=1)
    # retrive k smallest distances 
    k = min(k, len(embedding_database))
    idx_partitioned = np.argpartition(mls_distances, k)[:k]
    # sort by distances
    top_k_indices = idx_partitioned[np.argsort(mls_distances[idx_partitioned])]
    top_k_scores = mls_distances[top_k_indices]
    
    return top_k_indices, top_k_scores

# %% normalize based on gloabl statitics, not just batch subset
def get_normalization_stats(dataset, device="cuda"):
    loader = torch.utils.data.DataLoader(dataset, batch_size=2048, shuffle=True)
    
    # dists for global stats 
    k_dists_list, cmd_dists_list = [], []

    for batch_features in loader:
        for k, v in batch_features.items():
            batch_features[k] = v.to(device)
        
        # feature weights
        weights_dict = {'vaa_diff': np.float32(0.07331265), 'haa_diff': np.float32(0.07780298), 'release_speed': np.float32(0.08941311),
            'release_extension': np.float32(0.090783864), 'ax': np.float32(0.09481193), 'ay': np.float32(0.09528503), 'az': np.float32(0.104321584), 
            'arm_angle': np.float32(0.088479534), 'release_height': np.float32(0.08340817), 'release_x': np.float32(0.088434786)}
        cmd_weight = 0.11394637
        
        # kmu 
        kmu_order = ['vaa_diff', 'haa_diff', 'release_speed', 'release_extension', 'ax', 'ay', 'az', 
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
        k_dists = (k_mean_dist + k_cov_dist) * k_weight_sum
        cmd_dists = (avg_cmd - avg_cmd.T) ** 2
        cmd_dists = cmd_dists * cmd_weight
        
        k_dists_list.append(k_dists.detach().view(-1))
        cmd_dists_list.append(cmd_dists.detach().view(-1))

    k_mean = torch.cat(k_dists_list).mean()
    cmd_mean = torch.cat(cmd_dists_list).mean()
    return k_mean, cmd_mean

# %% data loaders
input = pl.read_parquet("cleaned_data/embed/input/pitch_mu_cov.parquet")
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
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

# %% model params
embedding_dim = 32 # effectively 32
model = SiameseNet(input_dim=11, embedding_dim=embedding_dim)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
criterion = TripletLoss()  # default params in method sig
sum(p.numel() for p in model.parameters())

# %% train, returns last iter model
trained_model = train_model(model, train_loader, val_loader, optimizer, 
                criterion, device="cuda", epochs=20, normalization_stats=normalization_stats)

# %% clean env
torch.cuda.empty_cache()
gc.collect()

# %% load saved best model
best_model = SiameseNet(input_dim=11, embedding_dim=embedding_dim)
dict = torch.load('../models/pitcher_embed.pth', weights_only=True)
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
def weighted_comps(df, embeddings, metric_cols, weight_col, k, batch_size=256):
    dim = embeddings.shape[1] // 2
    mu = embeddings[:, :dim]
    sigma = embeddings[:, dim:]
    # var
    var = np.square(sigma) + 1e-9
    # simple 
    n_samples = len(embeddings)
    k_adj = min(k + 1, n_samples)
    
    # output array
    indices = np.empty((n_samples, k_adj), dtype=np.int32)
    dists = np.empty((n_samples, k_adj), dtype=embeddings.dtype)

    # batches
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        current_batch_len = end_idx - start_idx
        
        # query batch
        q_mu = mu[start_idx:end_idx]
        q_var = var[start_idx:end_idx]
        
        # broadcast to fund var
        var_sum = q_var[:, np.newaxis, :] + var[np.newaxis, :, :]
        diff_sq = np.square(q_mu[:, np.newaxis, :] - mu[np.newaxis, :, :])
        
        # MLS dist
        term1 = diff_sq / var_sum
        term2 = np.log(var_sum)
        batch_dists = np.sum(term1 + term2, axis=2)
        
        # k+1 smallest
        unsorted_indices = np.argpartition(batch_dists, k_adj - 1, axis=1)[:, :k_adj]
        
        # gather dists
        batch_row_indices = np.arange(current_batch_len)[:, None]
        unsorted_dists = batch_dists[batch_row_indices, unsorted_indices]
        
        # local sort
        sort_order = np.argsort(unsorted_dists, axis=1)
        
        # store in main index
        indices[start_idx:end_idx] = unsorted_indices[batch_row_indices, sort_order]
        dists[start_idx:end_idx] = unsorted_dists[batch_row_indices, sort_order]
    
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
        w_total = pl.col("w_count") 
        w_mean_expr = (x_i * w_total).sum() / w_total.sum()
        w_diff = (w_mean_expr - x_t.first()).abs() 
        w_var = ((w_total * (x_i - w_mean_expr).pow(2)).sum() / w_total.sum()).sqrt()
        agg_exprs.append(w_diff.alias(f"{m}_weighted_diff"))
        agg_exprs.append(w_var.alias(f"{m}_weighted_std"))

    result = joined.group_by("orig_index").agg(agg_exprs).sort("orig_index")

    # stats
    return result

# %% embedding result preformance
embed_pref = weighted_comps(input, embeddings, metric_cols=['rv_100', 'xwoba', 'whiff_percent'], weight_col='pitches', k=10)
for columns in embed_pref.columns:
    print(f'{columns} mean: {embed_pref[columns].mean()}')

# %% refrence
for columns in input.select(cs.numeric()).columns:
    print(f'{columns} std: {input[columns].std()}')

# %% tsne visual tool
def umap_viz(color_col):
    color_col = color_col
    # tsne dim reduction
    reducer = umap.UMAP(n_components=2, metric ='cosine')
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
    plt.savefig('pPitchThrows.png')
    plt.show()

# %% throws
umap_viz('p_throws')

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
umap_viz('pitch_group')

# %% run value
umap_viz('rv_100')

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
