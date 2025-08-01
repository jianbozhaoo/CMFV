import random, os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
import texar.torch as tx
from pathlib import Path
from shutil import copyfile
import json
import torch.nn as nn

from utils import init_logger, load_pair_data, cal_nwgm_center_attention
from data_load_utils import build_dataset
from models import CMFV
from torch.cuda.amp import autocast, GradScaler

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=1234, help='Random seed.')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size.')
parser.add_argument('--lr', type=float, default=2e-6, help='Initial learning rate.')
parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')


parser.add_argument("--evi_num", type=int, default=20, help='Evidence num.')
parser.add_argument("--max_seq_length", default=128, type=int,)

args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log_sigma = nn.Parameter(torch.zeros(3, device=device), requires_grad=True)
# device = "cpu"

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)

dir_path = Path('outputs/politihop_3class/')

train_data_path = "data/politihop/original/politihop_train_adv.json"
dev_file_paths = [
    "data/politihop/original/politihop_valid_all.json",
    "data/politihop/original/politihop_test_all.json",
    "data/politihop/original/politihop_valid_adv.json",
    "data/politihop/original/politihop_test_adv.json",
    "data/politihop/original/politihop_valid_even.json",
    "data/politihop/original/politihop_test_even.json",
    "data/politihop/original/Hard_PolitiHop_by_gpt-3.5.json",
    "data/politihop/original/Hard_PolitiHop_by_gpt-4.0.json",
    ]
graph_rep_model_path = "pretrained_models/graph_rep_model_politihop.pt"
lambda_val = 0.41
nclass = 3
max_length = 5
beam_size = 3
causal_method = "cf"
n_cluster = 5
opt_steps = 1

tx.utils.maybe_create_dir(dir_path)
if not os.path.exists(dir_path):
    os.mkdir(dir_path)
if os.path.exists(dir_path / 'results.json'):
    print(dir_path / 'results already exists!')
    exit(0)
else:
    print(dir_path)

logger = init_logger(dir_path / "log.log")
tx.utils.maybe_create_dir(dir_path / 'models')

def correct_prediction(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct

def eval_model(model, log_sigma, dev_data_list, best_accuracy, accuracy, step, best_step):
    model.eval()
    for path in dev_data_list:
        dev_data = dev_data_list[path]
        dev_dataloader = DataLoader(dev_data, batch_size=args.batch_size, shuffle=False)
        dev_tqdm_iterator = dev_dataloader
        correct_pred = 0.0
        running_loss = 0.0
        with torch.no_grad():
            for index, data in enumerate(dev_tqdm_iterator):
                data = [i.to(device) for i in data]
                input_ids, input_mask, segment_ids, labels, sent_labels, evi_labels = data
                
                outputs, outputs1, outputs2, outputs3, outputs4 = model(data)
                # loss1 = F.cross_entropy(outputs, labels)
                loss1 = F.cross_entropy(outputs1, labels)

                outputs_list = [outputs2, outputs3, outputs4]
                ce_losses = [F.cross_entropy(o, labels) for o in outputs_list]
                weighted_loss = 0.0
                for i, Li in enumerate(ce_losses):
                    weighted_loss += 0.5 * torch.exp(-2.0 * log_sigma[i]) * Li + log_sigma[i]
                    
                loss2 = weighted_loss
                loss = loss1 + loss2
                  

                correct = correct_prediction(outputs, labels)
                correct_pred += correct
                running_loss += loss.item()
        
        dev_loss = running_loss / len(dev_dataloader)
        dev_accuracy = correct_pred / len(dev_data)
        logger.info('%s Dev total acc: %lf, total loss: %lf\r\n' % (path, dev_accuracy, dev_loss))

        accuracy[path].append(dev_accuracy.item())

        if dev_accuracy > best_accuracy[path]:
            best_accuracy[path] = dev_accuracy.item()
            best_step[path] = step
            torch.save({'model': model.state_dict(),
                        'log_sigma':   log_sigma.data.cpu(),
                        'best_accuracy': best_accuracy[path],
                        'dev_losses': dev_loss},
                        '%s/models/%s.pt' % (dir_path,path.split("/")[-1]))
    model.train()
    return best_accuracy, accuracy, best_step

logger.info("load data...")
train_data = build_dataset(train_data_path, args.evi_num, args.max_seq_length)
train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
total_steps = len(train_dataloader) * args.epochs / opt_steps


feature_num = 768
dev_data_list = {}
best_accuracy = {}
best_step = {}
accuracy = {}
dev_centers_list = {}
for path in dev_file_paths:
    dev_data = build_dataset(path, args.evi_num, args.max_seq_length)
    dev_data_list[path] = dev_data
    best_accuracy[path] = 0.0
    accuracy[path] = []
    best_step[path] = 0
    dev_dataloader = DataLoader(dev_data, batch_size=args.batch_size, shuffle=False)
    dev_dataloader = tqdm(dev_dataloader)
    dev_centers_list[path] = cal_nwgm_center_attention(graph_rep_model_path, dev_dataloader, nclass, device, feature_num, args.evi_num, n_cluster)

feature_num = 768
model = CMFV(feature_num, nclass, max_length, beam_size, args.evi_num, lambda_val, causal_method)

model = model.to(device)
log_sigma = log_sigma.to(device)
optimizer = optim.Adam(
    [
      {'params': model.parameters()},
      {'params': [log_sigma], 'lr': args.lr}
    ],
    lr=args.lr,
    weight_decay=args.weight_decay
)
logger.info("start train...")
train_tqdm_iterator = tqdm(train_dataloader)
train_centers = cal_nwgm_center_attention(graph_rep_model_path, train_tqdm_iterator, nclass, device, feature_num, args.evi_num, n_cluster)
step = 0
for epoch in range(args.epochs):
    model.train()
    optimizer.zero_grad()
    running_loss = 0.0
    correct_pred = 0.0
    train_tqdm_iterator = tqdm(train_dataloader)
    for index, data in enumerate(train_tqdm_iterator):
        data = [i.to(device) for i in data]
        input_ids, input_mask, segment_ids, labels, sent_labels, evi_labels = data  # sent_labels [batch,6] evi_labels [batch,6]

        outputs, outputs1, outputs2, outputs3, outputs4 = model(data)
        # loss1 = F.cross_entropy(outputs, labels)
        loss1 = F.cross_entropy(outputs1, labels)

        outputs_list = [outputs2, outputs3, outputs4]
        losses = [F.cross_entropy(o, labels) for o in outputs_list]
        weighted_loss = 0.0
        for i, Li in enumerate(losses):
            weighted_loss += 0.5 * torch.exp(-2.0 * log_sigma[i]) * Li + log_sigma[i]
                    
        loss2 = weighted_loss
        loss = loss1 + loss2

        correct = correct_prediction(outputs, labels)
        correct_pred += correct
        running_loss += loss.item()

        description = 'epoch %d Acc: %lf, Loss: %lf' % (epoch, correct_pred / (index + 1) / args.batch_size, running_loss / (index + 1))
        train_tqdm_iterator.set_description(description)

        loss = loss / opt_steps
        loss.backward()
        if index % opt_steps == 0 or index+1 == len(train_tqdm_iterator):
            optimizer.step()
            optimizer.zero_grad()
            # best_accuracy, accuracy, best_step = eval_model(model, dev_data_list, best_accuracy, accuracy, step, best_step)
            step += 1

    train_loss = running_loss / len(train_dataloader)
    train_accuracy = correct_pred / len(train_data)
    logger.info('epoch: %d, Train acc: %lf, total loss: %lf\r\n' % (epoch, train_accuracy, train_loss))
    best_accuracy, accuracy, best_step = eval_model(model, log_sigma, dev_data_list, best_accuracy, accuracy, step, best_step)
 
logger.info(json.dumps(best_accuracy,indent=True))
logger.info(json.dumps(best_step,indent=True))
logger.info("total_steps: %d"%total_steps)

res = {
    "total_steps":total_steps,
    "best_accuracy":best_accuracy,
    "best_step":best_step,
    "accuracy":accuracy,
}
fout = open(dir_path / 'results.json', 'w')
fout.write(json.dumps(res,indent=True))
fout.close()