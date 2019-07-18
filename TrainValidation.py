from mylayer import MyLayer
import numpy as np
import torch
import torch.nn.functional as F
import copy


# weight will be saved to & loaded from 'best_acc.pt'

def train_val(n_epoch, lr_input, dataset, training_idx, val_idx, load):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MyLayer().to(device)
    if load:
        model.load_state_dict(torch.load('best_acc.pt'))
        best_model_wts = copy.deepcopy(model.state_dict())

    optimizer = torch.optim.Adam(model.parameters(), lr=lr_input,
                                 betas=(0.9, 0.999), eps=1e-08,
                                 weight_decay=0, amsgrad=False)
    train_loss = []
    train_accs = []
    val_accs = []
    best_val_acc = 0.0

    for epoch in range(n_epoch):
        model.train()
        train_batch_accs = []
        train_batch_loss = []
        # accs is a list containing all the accuracies during each back propagation
        # epoch_loss is a list including all the losses
        for train_i in training_idx:
            data = dataset[train_i]
            data = data.to(device)
            optimizer.zero_grad()
            train_out = model(data.x, data.edge_index, data.edge_attr)
            # data.y is the groundturth and needs to be transferred to the correct shape
            loss = F.nll_loss(train_out, data.y.view(-1).type(torch.long))
            loss.backward()
            optimizer.step()
            train_batch_loss.append(loss.data.cpu().numpy())
            # out_put.max(1)[1] provides the category with the largest score
            # namely the predicted category
            acc = train_out.max(1)[1].eq(data.y.view(-1).type(torch.long))
            train_batch_accs.append(acc.cpu().numpy())
        train_accs.append(np.array(train_batch_accs).mean())
        train_loss.append(np.array(train_batch_loss).mean())

        model.eval()
        val_batch_acc = []
        with torch.no_grad():
            for val_i in val_idx:
                data = dataset[val_i]
                data = data.to(device)
                val_out = model(data.x, data.edge_index, data.edge_attr)
                acc = val_out.max(1)[1].eq(data.y.view(-1).type(torch.long))
                val_batch_acc.append(acc.cpu().numpy())
        val_batch_mean = np.array(val_batch_acc).mean()
        val_accs.append(val_batch_mean)
        print("epoch: ", epoch, "  val-batch-accuracy: ", val_batch_mean)
        if val_batch_mean > best_val_acc and load:
            best_model_wts = copy.deepcopy(model.state_dict())

    if load:
        torch.save(best_model_wts, 'best_acc.pt')

    return train_accs, val_accs, train_loss