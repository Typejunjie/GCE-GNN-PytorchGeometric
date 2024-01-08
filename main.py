import argparse
import torch
import time
from model import GCE_GNN
from torch_geometric.loader import DataLoader
from autotraining import autotraining
from dataset import session_graph
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--n_neighbor',type=int , default=5, help='The number of neighbors for global item')
parser.add_argument('--dropout_global',type=float , default=0.5, help='')
parser.add_argument('--dataset',type=str , default='Tmall', help='')
opt = parser.parse_args()

start_time = time.time()
varsion = '1.1-GCE_GNN'
dataset = opt.dataset

# Number of Session Graphs
# Tmall 351268

if dataset == 'diginetica':
    n_node = 43097
elif dataset == 'yoochoose1_64' or dataset == 'yoochoose1_4':
    n_node = 37483
elif dataset == 'RetailRocket':
    n_node = 36968
elif dataset == 'Tmall' :
    n_node = 40727
elif dataset == 'Nowplaying':
    n_node = 60416
else:
    n_node = 310

# Define hyper-parameters
batch_size = 100
learning_rate = 0.001
step_size = 3
device = torch.device('cpu')
gamma = 0.1
l2 = 1e-5
epochs = 10
topk = 20
print(f"Using {device} device")

print('Starting to get dataset')
train = session_graph(f'datasets/{dataset}', 'train.dataset', f'./datasets/{dataset}/raw/train.txt')
test = session_graph(f'datasets/{dataset}', 'test.dataset', f'./datasets/{dataset}/raw/test.txt')

train_dataloader= DataLoader(train, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test, batch_size=batch_size)

# logdir = f'./log/{dataset}'
logdir = f'./log/{dataset}'
# writer = SummaryWriter(logdir + '-beta ' + str(beta) + '--' + varsion)
writer = SummaryWriter(logdir + '-' + varsion)

# Init all parameters of model
# model = TEST_GNN(n_node, beta=beta).to(device)
# model = TEST_GNN(n_node, opt.beta, opt.k)
model = GCE_GNN(n_node, dataset=dataset, n_neighbor=opt.n_neighbor, dropout_global=opt.dropout_global)
loss_fun = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
train_set = autotraining(
    model,
    loss_fun,
    optimizer,
    writer=writer,
    scheduler=scheduler,
    topk=topk,
    device=device,
    )

print('Start training')
train_set.fit(train_dataloader, test_dataloader, epochs, log_parameter=False, eval=True)

# Save model results
print('Saving results')
writer.close()
## torch.save(model, f'./model_repository/{dataset}.model')
end_time = time.time()
total_seconds = end_time - start_time
minutes, seconds = divmod(total_seconds, 60)
print(f"Total duration {int(minutes)}min {int(seconds)}s")
print('Done')

