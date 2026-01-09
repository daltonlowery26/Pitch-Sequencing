import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np

# %% efficent loading
def relative_distance(batch_features):
    # extract and reshape
    rmu = batch_features['rmu'] 
    rsigma = batch_features['rsigma'] 
    kmu = batch_features['kmu']      
    ksigma = batch_features['ksigma']
    avg_cmd = batch_features['avg_command'].view(-1, 1) # add dim to mantain shape
    std_cmd = batch_features['std_command'].view(-1, 1)

    # squared frobenius norm proxy, does relative ranking of dist between matrices doesnt provide exact value
    # release
    r_mean_dist = torch.sum((rmu.unsqueeze(1) - rmu.unsqueeze(0))**2, dim=2) # dist between means
    r_cov_dist = torch.sum((rsigma.unsqueeze(1) - rsigma.unsqueeze(0))**2, dim=(2, 3))
    r_dists = r_mean_dist + r_cov_dist
    # kinematics 
    k_mean_dist = torch.sum((kmu.unsqueeze(1) - kmu.unsqueeze(0))**2, dim=2)
    k_cov_dist = torch.sum((ksigma.unsqueeze(1) - ksigma.unsqueeze(0))**2, dim=(2, 3))
    k_dists = k_mean_dist + k_cov_dist

    # eucledian dist for command
    cmd_dists = torch.sqrt((avg_cmd - avg_cmd.T)**2 + (std_cmd - std_cmd.T)**2)

    # normalization based on scale of values 
    k_mean_val = torch.mean(k_dists)
    r_mean_val = torch.mean(r_dists)
    cmd_mean_val = torch.mean(cmd_dists)
    
    # normalize and add epsilion to prevent div by 0
    k_norm = k_dists / (k_mean_val + 1e-8)
    r_norm = r_dists / (r_mean_val + 1e-8)
    cmd_norm = cmd_dists / (cmd_mean_val + 1e-8)

    # weighted sum of componenet based on features
    total_loss = (k_norm * 0.4411) + (r_norm * 0.2153) + (cmd_norm * 0.3435)

    return total_loss

# %% triplet loss, with hard mining
class BatchHardTripletLoss(nn.Module):
    def __init__(self, lambda_scale=.25, pos_threshold=0.2, neg_threshold=0.8):
        super(BatchHardTripletLoss, self).__init__()
        self.lambda_scale = lambda_scale # gt units to embedding units
        self.pos_p = pos_threshold # take top x clostest
        self.neg_p = neg_threshold # take bottom 1-x as negs

    def _pairwise_distance(self, embeddings):
        # distance between all embeddings
        dot_product = torch.matmul(embeddings, embeddings.t())
        square_norm = torch.diag(dot_product)
        distances = square_norm.unsqueeze(1) - 2.0 * dot_product + square_norm.unsqueeze(0)
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
        mask_pos = (gt_distances < pos_cutoff) & ~torch.eye(batch_size, device=embeddings.device).bool()
        # negative mask
        mask_neg = (gt_distances > neg_cutoff)

        # find hardest postive
        pos_search_matrix = emb_dists.clone()
        pos_search_matrix[~mask_pos] = -1.0
        hardest_pos_dists, hardest_pos_indices = torch.max(pos_search_matrix, dim=1)
        
        # find hardest negative
        neg_search_matrix = emb_dists.clone()
        neg_search_matrix[~mask_neg] = float('inf')
        hardest_neg_dists, hardest_neg_indices = torch.min(neg_search_matrix, dim=1)
        
        # distance in ground truth
        gt_pos_dists = torch.gather(gt_distances, 1, hardest_pos_indices.unsqueeze(1)).squeeze()
        gt_neg_dists = torch.gather(gt_distances, 1, hardest_neg_indices.unsqueeze(1)).squeeze()
        # margin based on dists
        dynamic_margin = self.lambda_scale * (gt_neg_dists - gt_pos_dists)  
        # loss function, with dynamic margin. weight heavily if confident in distance, zero out if close
        loss = torch.clamp(hardest_pos_dists - hardest_neg_dists + dynamic_margin, min=0.0)
        # only keep valid triplets
        valid_triplets = (mask_pos.sum(1) > 0) & (mask_neg.sum(1) > 0)
        return loss[valid_triplets].mean() # average loss for triplets for every target

# %% data loader, need to have rsigma, rmu ...  calculations in

# %% mlp, multi layer perception

# %% train loop     
