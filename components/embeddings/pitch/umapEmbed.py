# %% package
import gc
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from scipy.sparse import coo_matrix
from scipy.optimize import curve_fit
from sklearn.model_selection import train_test_split
import os

def relative_distance(batch_features, n_neighbors):
    kmu = torch.tensor(batch_features["kmu"], device="cuda")
    # weights
    weights_dict = {
        'hra': 0.105568744, 'vra': 0.093146056,
        'release_height': 0.09126035, 'release_x': 0.23855388,
        'deltax': 0.21256568, 'deltaz': 0.15866818,
        'midx': 0.041773327, 'midz': 0.058463812
    }
    kmu_order = ['hra', 'vra', 'release_height', 'release_x', 'deltax', 'deltaz', 'midx', 'midz']

    # weights and take square root
    weights = torch.tensor([weights_dict[k] for k in kmu_order], device=kmu.device, dtype=torch.float32)
    sqrt_weights = torch.sqrt(weights)

    # preapply data
    kmu_scaled = kmu * sqrt_weights

    n_samples = kmu.shape[0]
    chunk_size = 1600
    knn_dists_list = []
    knn_indices_list = []

    for i in range(0, n_samples, chunk_size):
        end_i = min(i + chunk_size, n_samples)

        # slice scaled data
        batch_data = kmu_scaled[i:end_i]

        # compute eucd dist
        dists_chunk = torch.cdist(batch_data, kmu_scaled, p=2)

        # find neighbors
        k_dists, k_indices = torch.topk(dists_chunk, k=n_neighbors + 1, dim=1, largest=False)

        # square dist
        k_dists = k_dists ** 2

        knn_dists_list.append(k_dists.cpu())
        knn_indices_list.append(k_indices.cpu())

    dists = torch.cat(knn_dists_list)
    indices = torch.cat(knn_indices_list)

    return dists, indices

def sigmaAndRho(distances, k, n_iter=64, bandwith=1.0):
    # target entropy for the distribution based on k neighbors
    target = np.log2(k) * bandwith

    # true nearest neighbor for every single point
    rhos = distances[:, 1]

    # binary search
    lo = np.zeros(distances.shape[0])
    hi = np.full(distances.shape[0], np.inf)
    mid = np.ones(distances.shape[0])

    for _ in range(n_iter):
        # (distance between i/j minus nn ) / varience of i
        val = np.maximum(distances[:, 1:] - rhos[:, None], 0.0) / mid[:, None]
        # sum of weights
        p_sum = np.sum(np.exp(-val), axis=1)
        # binary search
        diff = p_sum - target
        mask_low = diff < 0
        mask_high = diff > 0
        # increase sigma
        lo[mask_low] = mid[mask_low]
        inf_mask = np.isinf(hi)
        mid[mask_low & inf_mask] *= 2
        mid[mask_low & ~inf_mask] = (lo[mask_low & ~inf_mask] + hi[mask_low & ~inf_mask]) / 2
        #decrease sigma
        hi[mask_high] = mid[mask_high]
        mid[mask_high] = (lo[mask_high] + hi[mask_high]) / 2
    return rhos, mid

def adjMatrix(indices, distances, sigmas, rhos):
    n_samples = indices.shape[0]
    n_neighbors = indices.shape[1] # k nearest neighbors

    # coo matrix, stores row indices, column indices, and values, rest is zeros\
    rows = np.repeat(np.arange(n_samples), n_neighbors-1)
    cols = indices[:, 1:].flatten()
    dists = distances[:, 1:].flatten()

    # align paramters
    rhos_flat = np.repeat(rhos, n_neighbors - 1)
    sigmas_flat = np.repeat(sigmas, n_neighbors - 1)

    # weight of edge, how close a connection is
    val = np.maximum(dists - rhos_flat, 0.0) / sigmas_flat
    weights = np.exp(-val)

    # the probablity j is similar from i, how close i to j from i
    p_cond = coo_matrix((weights, (rows, cols)), shape=(n_samples, n_samples))

    # tranpose, how close i is to j from j perspective
    tP_cond = p_cond.transpose()

    # element wise mult
    prod = p_cond.multiply(tP_cond)

    # sym
    p_sym = p_cond + tP_cond - prod # if two points think they are close they are, if only one does the stronger weight

    return p_sym

def ab(spread, min_dist):
    # curve
    def curve(x, a, b):
        return 1.0 / (1.0 + a * x ** (2 * b))
    # xvalue and yvalue
    xv = np.linspace(0, spread *3, 300)
    yv = np.zeros(xv.shape)
    # if xv and yv are less then min distance they are connected with certian
    yv[xv < min_dist] = 1.0
    yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread) # if not expoential decay
    # fit a curve to the piecewise function
    params, covar = curve_fit(curve, xv, yv)
    return params[0], params[1]


class UMAPLoss(nn.Module):
    def __init__(self, minDist=0.4, spread=1.0, negSample = 5):
        super(UMAPLoss, self).__init__()

        self.neg_sample_rate = negSample
        self.device = "cuda"
        # ab hyperparmeter based on spread
        self.a, self.b = ab(spread, minDist)
        # tensors for foward pass
        self.a = torch.tensor(self.a, device="cuda")
        self.b = torch.tensor(self.b, device="cuda")
    def forward(self, embedding_to, embedding_from):
        batch_size = embedding_to.shape[0]

        # postive edge loss, we want to minimize the distance between the two
        dist_squared_pos = torch.sum((embedding_to - embedding_from) ** 2, dim =1)
        term_pos = 1.0 + self.a * (dist_squared_pos + 1e-6) ** self.b
        loss_pos = torch.sum(torch.log(term_pos))

        # maximize distance between random points
        loss_neg = 0.0
        for _ in range(self.neg_sample_rate):
            perm = torch.randperm(batch_size).to("cuda")
            embedding_neg = embedding_from[perm] # pair with random embedding in set
            dist_squared_neg = torch.sum((embedding_to - embedding_neg) ** 2, dim=1)
            # term to minimize
            inv_term_neg = 1.0 / (self.a * (dist_squared_neg + 1e-6) ** self.b + 1e-6)
            loss_neg += torch.sum(torch.log(1.0 + inv_term_neg))
        # postive and negitive loss
        total_loss = (loss_pos + loss_neg) / batch_size
        return total_loss


class umapDataset(Dataset):
    def __init__(self, df, n_neighbors, n_epochs):
        # neighbors
        self.df = df

        # nearest points
        dists, indices = relative_distance(df, n_neighbors)

        # to cpu
        dists = dists.cpu().numpy()
        indices = indices.cpu().numpy()

        # sigmas and rhos, sparse graph
        rhos, sigmas = sigmaAndRho(dists, n_neighbors)
        graph = adjMatrix(indices, dists, sigmas, rhos) # how likely a point is to be connected

        # from sparse matrix
        graph = graph.tocoo()
        graph.eliminate_zeros()
        # save coo
        self.graph_rows = graph.row
        self.graph_cols = graph.col
        self.weights = graph.data

        # first batch of indices
        self.heads = None
        self.tails = None
        self.resample()

    def resample(self):
        # probablity of including
        counts = np.floor(self.weights).astype(np.int32)
        fractional = self.weights - counts
        counts += np.random.binomial(1, fractional)

        # indicies for epoch based on prob
        self.heads = np.repeat(self.graph_rows, counts)
        self.tails = np.repeat(self.graph_cols, counts)

        # shuffle
        perm = np.random.permutation(len(self.heads))
        self.heads = self.heads[perm]
        self.tails = self.tails[perm]

    def __len__(self):
        return len(self.heads)

    def __getitem__(self, idx):
        return self.heads[idx], self.tails[idx]


# res block residual
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
            nn.Linear(dim, dim))
    def forward(self, x):
        return x + self.block(x)

class autoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout, layers, embedding):
       super().__init__()
       # encoder
       self.input = nn.Linear(input_dim, hidden_dim)
       self.encoder = nn.ModuleList([
           resBlock(hidden_dim, dropout) for _ in range(layers)
       ])

       self.eActiv = nn.ReLU()
       self.eNorm = nn.BatchNorm1d(hidden_dim)
       self.eOutput = nn.Linear(hidden_dim, embedding)

       # decoder
       self.dInput = nn.Linear(embedding, hidden_dim)
       self.decoder = nn.ModuleList([
           resBlock(hidden_dim, dropout) for _ in range(layers)
       ])
       self.dActiv = nn.ReLU()
       self.dNorm = nn.BatchNorm1d(hidden_dim)
       self.dOutput = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        x = self.input(x)
        for layer in self.encoder:
            x = layer(x)
        x = self.eNorm(x)
        x = self.eActiv(x)
        embed = self.eOutput(x)

        recon = self.dInput(embed)
        for layer in self.decoder:
            recon = layer(recon)
        recon = self.dNorm(recon)
        recon = self.dActiv(recon)
        recon = self.dOutput(recon)

        return embed, recon

def train(model, trainDataset, valDataset, optimizer, umapLoss, reconLoss, reconW, batchsize, epochs, type):
    model.to('cuda')
    best_loss = np.inf
    # load df into memory
    kmu_train = torch.tensor(trainDataset.df["kmu"].to_list(), device="cuda")
    kmu_val = torch.tensor(valDataset.df["kmu"].to_list(), device="cuda")

    for epoch in range(epochs):
        trainDataset.resample() # resample so new edges
        loader = DataLoader(trainDataset, batch_size=batchsize, shuffle=True, num_workers=2, pin_memory=True)
        total_loss = 0.0
        model.train()
        for h_index, t_index in loader:
            h_index, t_index = h_index.cuda(), t_index.cuda()
            # lookup on gpu
            x_h = kmu_train[h_index]
            x_t = kmu_train[t_index]
            # autoencode both
            z_h, x_h_recon = model(x_h)
            z_t, x_t_recon = model(x_t)
            # umap loss
            loss_topology = umapLoss(z_h, z_t)
            # reconstruction loss
            loss_mse = reconLoss(x_h_recon, x_h) + reconLoss(x_t_recon, x_t)
            # combine
            loss = loss_topology + (reconW * loss_mse)
            # opti
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # add loss
            total_loss += loss.item()
        # train loss
        train_loss = total_loss / len(loader)

        # validation loop
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
          valLoader = DataLoader(valDataset, batch_size=batchsize, shuffle=False, num_workers=2, pin_memory=True)
          for vh_index, vt_index in valLoader:
              vh_index, vt_index = vh_index.cuda(), vt_index.cuda()
              # lookup on gpu
              x_h_v = kmu_val[vh_index]
              x_t_v = kmu_val[vt_index]
              # autoencode both
              z_h_v, x_h_recon_v = model(x_h_v)
              z_t_v, x_t_recon_v = model(x_t_v)
              # umap loss
              vloss_topology = umapLoss(z_h_v, z_t_v)
              # reconstruction loss
              vloss_mse = reconLoss(x_h_recon_v, x_h_v) + reconLoss(x_t_recon_v, x_t_v)
              # combine
              vloss = vloss_topology + (reconW * vloss_mse)
              val_loss += vloss.item()

        avg_val_loss = val_loss / len(valLoader)
        # save best model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), f'/content/drive/MyDrive/{type}UmapEmbed.pth')

        print(f'epoch: {epoch}, valLoss: {avg_val_loss}, trainLoss: {train_loss}')
    return model
        
# %%
input = pl.read_parquet('/content/drive/MyDrive/pitch.parquet')
features = ['hra', 'vra', 'release_height', 'release_x', 'deltax', 'deltaz', 'midx', 'midz']
input = input.drop_nulls(subset=features)

# normalize
input = input.with_columns(kmu = pl.concat_list([(pl.col(f) - pl.col(f).mean()) / pl.col(f).std() for f in features]))

# presplit data
train, test = train_test_split(input, test_size=0.2, random_state=26)

# %% model
reconLoss = nn.MSELoss()
umapLoss = UMAPLoss()
model = autoEncoder(input_dim=8, hidden_dim=512, dropout=0.15, layers=24, embedding=64)
sum(p.numel() for p in model.parameters())

# %%
gc.collect()
torch.cuda.empty_cache()

# %% model train, train broad then train narrow
# broad train
trainDataset = umapDataset(train, n_neighbors=100, n_epochs=30)
valDataset = umapDataset(test, n_neighbors=100, n_epochs=30)
opti = torch.optim.AdamW(model.parameters(), lr=0.01)
lastModel = train(model = model, trainDataset=trainDataset, valDataset=valDataset, optimizer=opti,
                  umapLoss=umapLoss, reconLoss = reconLoss, reconW=1.0, batchsize = 32768, epochs = 6, type="b")
# load best broad model
lDict = torch.load('/content/drive/MyDrive/bUmapEmbed.pth')
model = autoEncoder(input_dim=8, hidden_dim=512, dropout=0.15, layers=24, embedding=64)
model.load_state_dict(lDict)

# narrow train
trainDataset = umapDataset(train, n_neighbors=15, n_epochs=30)
valDataset = umapDataset(test, n_neighbors=15, n_epochs=30)
opti = torch.optim.AdamW(model.parameters(), lr=0.001)
lastModel = train(model = model, trainDataset=trainDataset, valDataset=valDataset, optimizer=opti,
                  umapLoss=umapLoss, reconLoss = reconLoss, reconW=1.0, batchsize = 32768, epochs = 15, type="n")


# %% embeddings and test
def get_embeddings(model, data, device):
    model.to('cuda')
    model.eval()
    embeddings_list = []
    chunk_size = 65536

    with torch.no_grad():
      for i in range(0, len(data), chunk_size):
          batch = data[i : i + chunk_size]
          emb, _ = model(batch)
          embeddings_list.append(emb)

    final_embeddings = torch.cat(embeddings_list)
    return final_embeddings.cpu().numpy()

# load model get embedding
mDict = torch.load('/content/drive/MyDrive/umapEmbed.pth')
model = autoEncoder(input_dim=8, hidden_dim=512, dropout=0.15, layers=24, embedding=64)
model.load_state_dict(mDict)
fullData = torch.tensor(input["kmu"].to_list(), device="cuda")
embed = get_embeddings(model, fullData, "cuda")

def umap_viz(color_col, graph):
    import umap
    color_col = color_col
    # tsne dim reduction
    reducer = umap.UMAP(n_components=2, metric='cosine')
    tsne_coords = reducer.fit_transform(graph["embed"])
    # raw data with info
    plot_data = graph.select(pl.col(color_col)).to_pandas()
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
    
embedPL = pl.DataFrame({'embed': list(embed)})
input = pl.concat([input, embedPL], how='horizontal')
input.write_parquet('/content/drive/MyDrive/pitchUEmbed.parquet')
graph = input.sample(n=10000)

# %% umap viz by pitch type
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
graph = graph.with_columns(
    pitch_group = pl.col("pitch_name").replace(mapping))

umap_viz('pitch_group', graph)