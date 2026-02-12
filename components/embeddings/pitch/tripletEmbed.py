# %% package
import gc
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import os
os.chdir("C:/Users/dalto/OneDrive/Pictures/Documents/Projects/Coding Projects/Optimal Pitch/data/")

# %% efficent distance
def relative_distance(batch_features, normalization_stats):
    # feature weights
    weights_dict = {'hra': np.float32(0.105568744), 'vra': np.float32(0.093146056), 'release_height': np.float32(0.09126035), 'release_x': np.float32(0.23855388),
    'deltax': np.float32(0.21256568), 'deltaz': np.float32(0.15866818), 'midx': np.float32(0.041773327), 'midz': np.float32(0.058463812)}

    # extract and reshape
    kmu = batch_features["kmu"]
    kmu_order = ['hra', 'vra', 'release_height', 'release_x', 'deltax', 'deltaz', 'midx', 'midz', 'ay']

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
    def __init__(self, lambda_scale=1.75, pos_threshold=0.2, neg_threshold=0.6):
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
        loss = torch.relu(hardest_pos_dists - hardest_neg_dists + dynamic_margin)

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
        # raw input vector
        raw_inputs = torch.cat([kmu], dim=1)  #9
        # stanardize all the inputs
        mean = raw_inputs.mean(dim=0)
        std = raw_inputs.std(dim=0)
        self.input_emb = (raw_inputs - mean) / (std + 1e-8)
        # average
        self.kmu_target = kmu

    def __len__(self):
        return len(self.input_emb)

    def __getitem__(self, idx):
        # return precomputed data from idx, much quicker
        return {
            "input_emb": self.input_emb[idx],
            "kmu": self.kmu_target[idx]
        }

# %% residual
class resBlock(nn.Module):
    def __init__(self, dim, dropout):
      super().__init__()

      # layer
      self.block = nn.Sequential(
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
      )
    def forward(self, x):
      return x + self.block(x)

class SiameseNet(nn.Module):
    def __init__(self, input_dim, dim, embedding_dim):
        # inherit
        super(SiameseNet, self).__init__()
        # layers and input dim
        self.input_proj = nn.Linear(input_dim, dim)
        self.layers = nn.ModuleList([
            resBlock(dim, 0.1) for _ in range(24)
        ])
        
        self.o_activ = nn.ReLU()
        self.o_norm = nn.BatchNorm1d(dim)
        self.output = nn.Linear(dim, embedding_dim)

    def forward(self, x):
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x)
        # normalize for final layer
        x = self.o_norm(x)
        x = self.o_activ(x)
        embeddings = self.output(x)
        return torch.nn.functional.normalize(embeddings, p=2, dim=1)

# %% train loop
def train_model(model, dataloader, val_loader, optimizer, criterion, device, epochs, normalization_stats):
    model.to(device)
    best_loss = np.inf
    # training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

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
                torch.save(model.state_dict(), '/content/drive/MyDrive/indiv_pitch_embeds.pth')

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
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)
    # weights dict
    weights_dict = {'hra': np.float32(0.105568744), 'vra': np.float32(0.093146056), 'release_height': np.float32(0.09126035), 'release_x': np.float32(0.23855388),
    'deltax': np.float32(0.21256568), 'deltaz': np.float32(0.15866818), 'midx': np.float32(0.041773327), 'midz': np.float32(0.058463812)}

    kmu_order = ['hra', 'vra', 'release_height', 'release_x', 'deltax', 'deltaz', 'midx', 'midz' ,'ay']
    # weight tensor
    feature_weights = torch.tensor([weights_dict[k] for k in kmu_order], device=device, dtype=torch.float32)
    w_reshaped = feature_weights.view(1, 1, -1)

    k_dists_list = []

    for batch_features in loader:
        # move to device
        kmu = batch_features["kmu"].to(device)

        # calculated weighted diff
        diff_sq = (kmu.unsqueeze(1) - kmu.unsqueeze(0)) ** 2
        k_mean_dist = torch.sum(w_reshaped * diff_sq, dim=2)
        k_dists_list.append(k_mean_dist.detach().view(-1))

    # mean of weighted distances
    k_mean = torch.cat(k_dists_list).mean()
    return k_mean

# %% input
input = pl.read_parquet('cleaned_data/embed/input/pitch.parquet')
features = ['hra', 'vra', 'release_height', 'release_x', 'deltax', 'deltaz', 'midx', 'midz', 'ay']
input = input.drop_nulls(subset=features)

# normalize
input = input.with_columns(
    kmu = pl.concat_list([(pl.col(f) - pl.col(f).mean()) / pl.col(f).std() for f in features])
)

# %% data loaders
dataset = ManifoldDataset(df=input)
# train and val
train_size = int(0.9 * len(dataset))
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
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

# %% model params
embedding_dim = 64
model = SiameseNet(input_dim=12, dim = 256, embedding_dim=embedding_dim)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.001)
criterion = TripletLoss() 

# %% parameter count
sum(p.numel() for p in model.parameters())

# %% train, returns last iter model
trained_model = train_model(model, train_loader, val_loader, optimizer,
                criterion, device="cuda", epochs=50, normalization_stats=normalization_stats)

# %% clean env
torch.cuda.empty_cache()
gc.collect()

# %% load saved best model
model = SiameseNet(input_dim=12, dim = 256, embedding_dim=embedding_dim)
dict = torch.load('../models/pitch.pth', weights_only=True)
model.load_state_dict(dict)

# %% infrence and closeness
inference_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
embeddings = get_embeddings(model=model, dataloader=inference_loader, device="cuda")

# embeddings
input = pl.concat(
    [input, pl.from_numpy(embeddings, schema=["embeds"])], 
    how="horizontal"
)
# %% save pitch embeddings
#input.write_parquet('cleaned_data/embed/output/pitch_embeded.parquet')
input = pl.read_parquet('cleaned_data/embed/output/pitch_embeded.parquet')
print(input.columns)
# %% umap visual tool
def umap_viz(color_col):
    import umap
    color_col = color_col
    # tsne dim reduction
    reducer = umap.UMAP(n_components=2, metric='cosine')
    tsne_coords = reducer.fit_transform(input['embeds'])
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
    plt.savefig(f'{color_col}.png')
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
    'knuckle': ['Knuckleball']}
mapping = {v: k for k, values in types.items() for v in values}
input = input.with_columns(
    pitch_group = pl.col("pitch_name").replace(mapping))

umap_viz('pitch_group')

# %%
umap_viz('abs_strike')
