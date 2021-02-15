import argparse
from os import path
import numpy as np
from dgl import batch
from dgl.data.ppi import LegacyPPIDataset
from dgl.nn.pytorch import GraphConv
from sklearn.metrics import f1_score
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader


MODEL_STATE_FILE = path.join(path.dirname(path.abspath(__file__)), "model_state.pth")


class BasicGraphModel(nn.Module):
    def __init__(self, g, n_layers, input_size, hidden_size, output_size, nonlinearity):
        super().__init__()

        self.g = g
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(input_size, hidden_size, activation=nonlinearity))
        for _ in range(n_layers - 1):
            self.layers.append(GraphConv(hidden_size, hidden_size, activation=nonlinearity))
        self.layers.append(GraphConv(hidden_size, output_size))

    def forward(self, inputs):
        outputs = inputs
        for _, layer in enumerate(self.layers):
            outputs = layer(self.g, outputs)
        return outputs

    def set_graph(self, g):
        self.g = g
        for layer in self.layers:
            layer.g = g

class AttentionHead(nn.Module):
    def __init__(self, in_dim, out_dim, device, input_slope=0.2):
        super().__init__()
        self.g = None
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device
        self.input_slope = input_slope

        self.linear = nn.Linear(self.in_dim, self.out_dim)
        self.attention = nn.Linear(2*self.out_dim, 1)

        self.init_xavier_uniform()

    def init_xavier_uniform(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.linear.weight, gain=gain)
        nn.init.xavier_uniform_(self.attention.weight, gain=gain)

    def forward(self, inputs):
        Wh = self.linear(inputs)
        
        n_nodes = self.g.number_of_nodes()
        U, V = self.g.all_edges()

        edges_repr = torch.cat((Wh[U], Wh[V]), dim=1).to(self.device)

        alpha = torch.zeros((n_nodes, n_nodes)).to(self.device)
        alpha[U, V] = self.attention(edges_repr).squeeze(-1)
        alpha = F.softmax(F.leaky_relu(alpha, negative_slope=self.input_slope), dim=1)
        
        return torch.relu(alpha.mm(Wh))

class MultiHead(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, device, average=False):
        super(MultiHead, self).__init__()
        self.average = average
        self.heads = nn.ModuleList([AttentionHead(in_dim, out_dim, device) for _ in range(num_heads)])

    def forward(self, h):
        head_outs = [attn_head(h) for attn_head in self.heads]
        if self.average:
            return torch.stack(head_outs, dim=2).mean(dim=2)
        return torch.cat(head_outs, dim=1)

class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads=4, device='cpu'):
        super(GAT, self).__init__()
        self.gat = nn.Sequential(
            MultiHead(in_dim, hidden_dim, num_heads, device),
            nn.ELU(inplace=True),
            MultiHead(hidden_dim * num_heads, out_dim, num_heads, device, average=True)
        )

    def forward(self, h):
        return self.gat(h)

    def set_graph(self, g):
        for module in self.gat:
            if isinstance(module, MultiHead):
                for head in module.heads:
                    head.g = g

def main(args, f1_scores=[]):
    # create the dataset
    train_dataset, test_dataset = LegacyPPIDataset(mode="train"), LegacyPPIDataset(mode="test")
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    n_features, n_classes = train_dataset.features.shape[1], train_dataset.labels.shape[1]

    # create the model, loss function and optimizer
    device = torch.device("cpu" if args.gpu < 0 else "cuda:" + str(args.gpu))

    if args.model == "BGM":
        model = BasicGraphModel(g=train_dataset.graph, n_layers=2, input_size=n_features,
                            hidden_size=args.hidden_dim, output_size=n_classes, nonlinearity=F.elu).to(device)
    elif args.model == "GAT":
        model = GAT(n_features, args.hidden_dim, n_classes, num_heads=args.n_heads, device=device).to(device)
    else:
        raise AttributeError('invalid model name')

    loss_fcn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # train and test
    if args.mode == "train":
        train(model, loss_fcn, device, optimizer, train_dataloader, test_dataset, f1_scores=f1_scores)
        torch.save(model.state_dict(), MODEL_STATE_FILE)
    model.load_state_dict(torch.load(MODEL_STATE_FILE, map_location=device))
    return test(model, loss_fcn, device, test_dataloader)


def train(model, loss_fcn, device, optimizer, train_dataloader, test_dataset, f1_scores=[]):
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for _, data in enumerate(train_dataloader):
            subgraph, features, labels = data
            features = features.to(device)
            labels = labels.to(device)
            model.set_graph(subgraph.to(device))
            logits = model(features.float())
            loss = loss_fcn(logits, labels.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        loss_data = np.array(losses).mean()
        print("Epoch {:05d} | Loss: {:.4f}".format(epoch + 1, loss_data))

        if epoch % 1 == 0:
            scores = []
            for _, test_data in enumerate(test_dataset):
                subgraph, features, labels = test_data
                features = features.clone().detach().to(device)
                labels = labels.clone().detach().to(device)
                score, _ = evaluate(features.float(), model, subgraph, labels.float(), loss_fcn, device)
                scores.append(score)
            print("F1-Score: {:.4f} ".format(np.array(scores).mean()))
            f1_scores.append(np.array(scores).mean())
    return f1_scores

def test(model, loss_fcn, device, test_dataloader):
    print("\n-- Testing")
    test_scores = []
    for _, test_data in enumerate(test_dataloader):
        subgraph, features, labels = test_data
        features = features.to(device)
        labels = labels.to(device)
        test_scores.append(evaluate(features, model, subgraph, labels.float(), loss_fcn, device)[0])
    mean_scores = np.array(test_scores).mean()
    print("F1-Score: {:.4f}".format(np.array(test_scores).mean()))
    return mean_scores


def evaluate(features, model, subgraph, labels, loss_fcn, device):
    with torch.no_grad():
        model.eval()
        model.set_graph(subgraph.to(device))
        output = model(features.float())
        loss_data = loss_fcn(output, labels.float())
        predict = np.where(output.data.cpu().numpy() >= 0.5, 1, 0)
        score = f1_score(labels.data.cpu().numpy(), predict, average="micro")
        return score, loss_data.item()


def collate_fn(sample):
    graphs, features, labels = map(list, zip(*sample))
    graph = batch(graphs)
    features = torch.from_numpy(np.concatenate(features))
    labels = torch.from_numpy(np.concatenate(labels))
    return graph, features, labels



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",  choices=["train", "test"], default="train")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU to use. Set -1 to use CPU.")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--model", type=str, default="GAT")
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--hidden_dim", type=int, default=512)
    args = parser.parse_args()
    main(args)
