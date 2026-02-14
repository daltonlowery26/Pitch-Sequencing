# %%
import os
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.utils import class_weight
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, random_split

os.chdir("/Users/daltonlowery/Desktop/projects/Optimal Pitch/data")
device = "mps"  # set training device


# %% model and train blocks
class resBlock(nn.Module):
    def __init__(self, dim, dropout):
        super(resBlock, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return x + self.block(x)


class bipModel(nn.Module):
    def __init__(self, hidden, layer1, layer2, layer3, outputs):
        super(bipModel, self).__init__()
        # embeddings
        self.eInput = nn.Linear(64, hidden)
        self.eBlock = nn.ModuleList([resBlock(hidden, 0.2) for _ in range(layer1)])
        # swingTraits
        self.sInput = nn.Linear(7, hidden)
        self.sBlock = nn.ModuleList([resBlock(hidden, 0.2) for _ in range(layer2)])

        # output
        self.combine = nn.Linear((hidden * 2), hidden)
        self.backBone = nn.ModuleList([resBlock(hidden, 0.2) for _ in range(layer3)])
        self.oNorm = nn.BatchNorm1d(hidden)
        self.oActiv = nn.ReLU()

        # classification
        self.fLinear = nn.Linear(hidden, outputs)

    def forward(self, embeds, swing):
        # embed model
        embeds = self.eInput(embeds)
        for block in self.eBlock:
            embeds = block(embeds)

        # swing traits
        swing = self.sInput(swing)
        for block in self.sBlock:
            swing = block(swing)

        # combine
        combined = torch.cat([swing, embeds], dim=1)
        combined = self.combine(combined)
        for block in self.backBone:
            combined = block(combined)

        combined = self.oNorm(combined)
        combined = self.oActiv(combined)

        # output
        return self.fLinear(combined)


class bipDataset(Dataset):
    def __init__(self, df):
        super(bipDataset, self).__init__()

        # helper function
        embeddings = torch.tensor(df["embed"].to_list(), dtype=torch.float32).to(device)
        swingTraits = torch.tensor(df["traits"].to_list(), dtype=torch.float32)

        mean = swingTraits.mean(dim=0)
        std = swingTraits.std(dim=0)
        traits = (swingTraits - mean) / std
        traits = traits.to(device)
        
        try:
            label = torch.tensor(df["outcome"].to_list(), dtype=torch.long).to(device)
        except Exception:
            print('No Labels!')
            label = None
        self.e = embeddings
        self.t = traits
        self.l = label

    def __len__(self):
        return len(self.e)

    # return indices and hold dataset on gpu
    def __getitem__(self, idx):
        return {"embeds": self.e[idx], "traits": self.t[idx], "labels": self.l[idx]}


def train(model, dataLoader, valLoader, optimizer, lossFunc, epochs):
    model.to(device)
    best_loss = np.inf
    for epoch in range(epochs):
        total_loss = 0.0
        model.train()
        for batchIdx, batchFeat in enumerate(dataLoader):
            # all feature in the batch to device
            for key, value in batchFeat.items():
                batchFeat[key] = value.to(device)
            # zero the gradient accumilation
            optimizer.zero_grad()
            # model pass
            predicted = model(batchFeat["embeds"], batchFeat["traits"])
            labels = batchFeat["labels"].view(-1)
            # loss function
            loss = lossFunc(predicted, labels)
            # opti step
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            # step the optimizer and lr scheduler
            optimizer.step()
            scheduler.step()
            # add loss value
            total_loss += loss.item()
        # train loss
        train_loss = total_loss / len(dataLoader)

        # validation loop
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for valIdx, valFeat in enumerate(valLoader):
                # all feature in the batch to device
                for key, value in valFeat.items():
                    valFeat[key] = value.to(device)
                # model pass
                predicted = model(valFeat["embeds"], valFeat["traits"])
                labels = valFeat["labels"].view(-1)
                # loss function
                loss = lossFunc(predicted, labels)
                # add loss value
                val_loss += loss.item()

        avg_val_loss = val_loss / len(valLoader)

        # save best model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), "../models/sdModels/bipModel.pth")
        print(f"epoch: {epoch}, valLoss: {avg_val_loss}, trainLoss: {train_loss}")
    return model


# %% data prep
df = pl.read_parquet("cleaned_data/metrics/xswing/xtraitContact.parquet")
df = df.with_columns(
    traits=pl.concat_arr(["bat_speed","swing_length",
            "swing_path_tilt","attack_angle",
            "attack_direction","intercept_x","intercept_y"])
)
des = ["hit_into_play"]
df = df.filter(pl.col("description").is_in(des))

# add events, catagories
swing = (pl.scan_csv("cleaned_data/pitch_2015_2026.csv")
    .select(["game_pk","batter_id","pitcher_id",
            "at_bat_number","pitch_number","count","events"]).collect(engine="streaming"))
df = df.join(swing, 
    on=["game_pk", "batter_id", "pitcher_id", "at_bat_number", "pitch_number", "count"], validate="1:1", how="left")
df = df.filter(pl.col("events") != "catcher_interf")

# map events to catagories
eventsMap = {"single": 0, "double": 1, "triple": 2, "home_run": 3}
df = df.with_columns(
    outcome=(pl.col("events").replace_strict(eventsMap, default=4)).cast(pl.Int8)
)

# final check (should only drop one row)
df = df.drop_nulls(
    subset=["bat_speed","swing_length","swing_path_tilt",
        "attack_angle","attack_direction","intercept_x",
        "intercept_y","embed","events"]
)
# %% class weights
weights = class_weight.compute_class_weight(
    "balanced", classes=np.unique(df["outcome"].to_numpy()), y=df["outcome"].to_numpy()
)
weightsTensor = torch.tensor(weights, dtype=torch.float, device=device)

# %% data loading and splitting
dataset = bipDataset(df)
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(
    dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(26),
)

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

# loss, model, opti
epochs = 20
loss = nn.CrossEntropyLoss(weight=weightsTensor)
model = bipModel(hidden=128, layer1=8, layer2=3, layer3=8, outputs=5)
opti = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = CosineAnnealingLR(opti, T_max=epochs, eta_min=1e-6)
sum(p.numel() for p in model.parameters())

# %% train model
lastModel = train(model,dataLoader=train_loader, valLoader=val_loader,optimizer=opti, lossFunc=loss,epochs=epochs,)

# %% testing
def test(model, testLoader):
    model.eval()
    gLabels = []
    gPreds = []

    for batchIdx, batchFeat in enumerate(testLoader):
        # to cuda
        for k, v in batchFeat.items():
            batchFeat[k] = v.to(device)

        predictions = model(batchFeat["embeds"], batchFeat["traits"])
        labels = batchFeat["labels"].view(-1)
        predictions = torch.softmax(predictions, dim=1)

        labels = labels.cpu().detach().numpy()
        predictions = predictions.cpu().detach().numpy()
        gLabels.append(labels)
        gPreds.append(predictions)

    all_predictions = np.concatenate(gPreds)
    all_labels = np.concatenate(gLabels)
    return all_predictions, all_labels

# %% predeciton function
def bipValue(df):
    # load model
    model = bipModel(hidden=128, layer1=8, layer2=3, layer3=8, outputs=5)
    stateDict = torch.load('../models/sdModels/bipvModel.pth')
    model.load_state_dict(stateDict)
    model.to('mps')
    
    # dataloader
    data = bipDataset(df)
    dataLoad = DataLoader(data, batch_size=512, shuffle=False)
    
    # preds
    preds = []
    for batchIdx, batchFeat in enumerate(dataLoad):
        # to mps
        for k, v in batchFeat.items():
            batchFeat[k] = v.to('mps')

        # predections
        predictions = model(batchFeat["embeds"])
        predictions = torch.sigmoid(predictions)
        predictions = predictions.cpu().detach().numpy()
        preds.append(predictions)

    all_predictions = np.concatenate(preds)
    return all_predictions

# %% load and test
stateDict = torch.load("../models/sdModels/bipModel.pth")
model = bipModel(hidden=128, layer1=8, layer2=3, layer3=8, outputs=5)
model.load_state_dict(stateDict)
model.to(device)
predictions, labels = test(model, test_loader)

print(roc_auc_score(labels, predictions, multi_class="ovr"))
print(average_precision_score(labels, predictions))
