import math
import timeit

import os
import os.path as osp
from pathlib import Path
import numpy as np

import torch

from torch_geometric.loader import TemporalDataLoader

# Internal imports from your TGN setup
from tgb.utils.utils import get_args, set_random_seed, save_results
from tgb.linkproppred.evaluate import Evaluator
from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset

from modules.decoder import LinkPredictor
from modules.early_stopping import EarlyStopMonitor
from modules.neighbor_loader import LastNeighborLoader
from modules.tgat_emb import TGATEncoder


# === TRAIN ===
def train():
    model['gnn'].train()
    model['link_pred'].train()
    neighbor_loader.reset_state()  # Start with an empty graph.

    total_loss = 0

    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        src, pos_dst, t = batch.src, batch.dst, batch.t
        neg_dst = torch.randint(min_dst_idx, max_dst_idx + 1, (src.size(0),), device=device)

        n_id = torch.cat([src, pos_dst, neg_dst]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        z = model['gnn'](model['x'], edge_index, data.t[e_id].to(device))

        pos_out = model['link_pred'](z[assoc[src]], z[assoc[pos_dst]])
        neg_out = model['link_pred'](z[assoc[src]], z[assoc[neg_dst]])

        loss = criterion(pos_out, torch.ones_like(pos_out))
        loss += criterion(neg_out, torch.zeros_like(neg_out))

        loss.backward()
        optimizer.step()
        neighbor_loader.insert(src, pos_dst)

        total_loss += float(loss) * batch.num_events

    return total_loss / train_data.num_events

@torch.no_grad()
def test(loader, neg_sampler, split_mode):
    model['gnn'].eval()
    model['link_pred'].eval()
    perf_list = []

    for pos_batch in loader:
        pos_src, pos_dst, pos_t = pos_batch.src, pos_batch.dst, pos_batch.t
        neg_batch_list = neg_sampler.query_batch(pos_src, pos_dst, pos_t, split_mode=split_mode)

        for idx, neg_batch in enumerate(neg_batch_list):
            src = torch.full((1 + len(neg_batch),), pos_src[idx], device=device)
            dst = torch.tensor(
                np.concatenate(([np.array([pos_dst.cpu().numpy()[idx]]), np.array(neg_batch)]), axis=0),
                device=device,
            )

            n_id = torch.cat([src, dst]).unique()
            n_id, edge_index, e_id = neighbor_loader(n_id)
            assoc[n_id] = torch.arange(n_id.size(0), device=device)

            z = model['gnn'](model['x'], edge_index, data.t[e_id].to(device))
            y_pred = model['link_pred'](z[assoc[src]], z[assoc[dst]])

            input_dict = {
                "y_pred_pos": np.array([y_pred[0, :].squeeze(dim=-1).cpu()]),
                "y_pred_neg": np.array(y_pred[1:, :].squeeze(dim=-1).cpu()),
                "eval_metric": [metric],
            }
            perf_list.append(evaluator.eval(input_dict)[metric])
        neighbor_loader.insert(pos_src, pos_dst)

    return float(torch.tensor(perf_list).mean())

# === MAIN ===
start_overall = timeit.default_timer()
# DATA = "tgbl-wiki"
args, _ = get_args()
print("INFO: Arguments:", args)

DATA = args.data

LR = args.lr
BATCH_SIZE = args.bs
K_VALUE = args.k_value
NUM_EPOCH = args.num_epoch
SEED = args.seed
MEM_DIM = args.mem_dim
TIME_DIM = args.time_dim
EMB_DIM = args.emb_dim
TOLERANCE = args.tolerance
PATIENCE = args.patience
NUM_RUNS = args.num_run
NUM_NEIGHBORS = 10
MODEL_NAME = 'TGAT'

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load data
dataset = PyGLinkPropPredDataset(name=DATA, root="datasets")
train_mask = dataset.train_mask
val_mask = dataset.val_mask
test_mask = dataset.test_mask
data = dataset.get_TemporalData().to(device)
metric = dataset.eval_metric

train_data = data[train_mask]
val_data = data[val_mask]
test_data = data[test_mask]

train_loader = TemporalDataLoader(train_data, batch_size=BATCH_SIZE)
val_loader = TemporalDataLoader(val_data, batch_size=BATCH_SIZE)
test_loader = TemporalDataLoader(test_data, batch_size=BATCH_SIZE)

min_dst_idx, max_dst_idx = int(data.dst.min()), int(data.dst.max())

print("==========================================================")
print(f"=================*** {MODEL_NAME}: LinkPropPred: {DATA} ***=============")
print("==========================================================")

results_path = f'{osp.dirname(osp.abspath(__file__))}/saved_results'
os.makedirs(results_path, exist_ok=True)
results_filename = f'{results_path}/{MODEL_NAME}_{DATA}_results.json'
evaluator = Evaluator(name=DATA)
neg_sampler = dataset.negative_sampler

for run_idx in range(NUM_RUNS):
    print('-------------------------------------------------------------------------------')
    print(f"INFO: >>>>> Run: {run_idx} <<<<<")
    start_run = timeit.default_timer()

    torch.manual_seed(run_idx + SEED)
    set_random_seed(run_idx + SEED)

    
    neighbor_loader = LastNeighborLoader(data.num_nodes, size=NUM_NEIGHBORS, device=device)

    #dummy node feature as the default tgbl-wiki data does not have node features
    x = torch.zeros((data.num_nodes, EMB_DIM), device=device)


    gnn = TGATEncoder(EMB_DIM, EMB_DIM, TIME_DIM).to(device)
    link_pred = LinkPredictor(in_channels=EMB_DIM).to(device)
    model = {'gnn': gnn, 'link_pred': link_pred, 'x': x}

    # optimizer = torch.optim.Adam(gnn.parameters() | link_pred.parameters(), lr=LR)
    # criterion = torch.nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        set(model['gnn'].parameters()) | set(model['link_pred'].parameters()),
        lr=LR,
    )
    criterion = torch.nn.BCEWithLogitsLoss()

    assoc = torch.empty(data.num_nodes, dtype=torch.long, device=device)

    save_model_dir = f'{osp.dirname(osp.abspath(__file__))}/saved_models/'
    save_model_id = f'{MODEL_NAME}_{DATA}_{SEED}_{run_idx}'
    early_stopper = EarlyStopMonitor(save_model_dir, save_model_id, TOLERANCE, PATIENCE)

    dataset.load_val_ns()
    val_perf_list = []
    for epoch in range(1, NUM_EPOCH + 1):
        loss = train()
        print(f"Epoch: {epoch:02d}, Loss: {loss:.4f}")
        perf_val = test(val_loader, neg_sampler, split_mode="val")
        print(f"\tValidation {metric}: {perf_val: .4f}")
        val_perf_list.append(perf_val)

        if early_stopper.step_check(perf_val, {'gnn': gnn, 'link_pred': link_pred}):
            break

    early_stopper.load_checkpoint({'gnn': gnn, 'link_pred': link_pred})
    dataset.load_test_ns()
    perf_test = test(test_loader, neg_sampler, split_mode="test")
    print(f"\tTest {metric}: {perf_test: .4f}")

    save_results({'model': MODEL_NAME,
                  'data': DATA,
                  'run': run_idx,
                  'seed': SEED,
                  f'val {metric}': val_perf_list,
                  f'test {metric}': perf_test,
                  }, results_filename)
    print(f"INFO: >>>>> Run: {run_idx}, elapsed time: {timeit.default_timer() - start_run: .4f} <<<<<")
    print('-------------------------------------------------------------------------------')

print(f"Overall Elapsed Time (s): {timeit.default_timer() - start_overall: .4f}")
