from hamiltonian import Hamiltonian
import numpy as np
import matplotlib.pyplot as plt
from TrainValidation import train_val
import time


dataset = Hamiltonian(root='data/hamiltonian', k_n=1, y_cut=0.2)
# N is number of samples
N = len(dataset)
# idx is the index of the dataset that is going to be splited
idx = np.arange(N)
# random shuffle the idx dataset
np.random.shuffle(idx)

crossva = [None]*5
crossva[0], crossva[1], crossva[2], crossva[3], crossva[4] = np.array_split(idx, 5)

n_epoch = 3000
lr = 0.0001

train_accs = []
val_accs = []
train_loss = []
drop_p = 0.2

since = time.time()
for i in range(5):
    val_idx = crossva[i]
    train_idx = np.setdiff1d(idx, val_idx)
    train_list, val_list, loss_list = train_val(n_epoch=n_epoch, lr_input=lr,
                                                dataset=dataset, training_idx=train_idx,
                                                val_idx=val_idx, rd=drop_p)
    train_accs.append(train_list)
    val_accs.append(val_list)
    train_loss.append(loss_list)

time_elapsed = time.time() - since

# transfer list of list into 2D numpy array
train_array = np.array(train_accs)
val_array = np.array(val_accs)
train_mean = train_array.mean(0)
val_mean = val_array.mean(0)
train_std = train_array.std(0)
val_std = val_array.std(0)

# plot the average accuracy of the 5-fold for
# both train and validation
plt.figure()
plt.plot(np.arange(n_epoch), train_mean, color='green', label='train')
plt.plot(np.arange(n_epoch), val_mean, color='red', label='val')
# plt.errorbar(np.arange(n_epoch), train_mean, yerr=train_std, fmt='o',
#              color='green', alpha=0.1, elinewidth=1)
# plt.errorbar(np.arange(n_epoch), val_mean, yerr=val_std, fmt='o',
#              color='red', alpha=0.1, elinewidth=1)
plt.title('lr='+str(lr)+' drop='+str(drop_p))
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend()
# plt.savefig(fname='Accruacy'+str(lr).replace('.', '')+str(reg_L1).replace('.', ''))
plt.savefig(fname='Accruacy_1')
plt.close()

# plot the average loss across 5-fold during training
plt.figure()
loss_mean = np.array(train_loss).mean(0)
loss_std = np.array(train_loss).std(0)
plt.plot(np.arange(n_epoch), loss_mean, color='blue')
# plt.errorbar(np.arange(n_epoch), loss_mean, yerr=loss_std, fmt='o',
#              color='blue', alpha=0.1, elinewidth=1)
plt.ylabel("loss")
plt.title('lr='+str(lr)+' drop='+str(drop_p))
plt.xlabel("epoch")
# plt.savefig(fname='Loss'+str(lr).replace('.', '')+str(reg_L1).replace('.', ''))
plt.savefig(fname='Loss_1')
plt.close()
print("finish successfully")
print('Training complete in {:.0f}h'.format(time_elapsed // 3600))


# training_idx, val_idx = idx[:300], idx[300:]
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # move the model to the device
# model = MyLayer().to(device)
#
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001,
#                              betas=(0.9, 0.999), eps=1e-08,
#                              weight_decay=0, amsgrad=False)
# train_loss, train_accs, val_accs include the average loss, accruacy of each epoch
# train_loss = []
# train_accs = []
# val_accs =[]
# best_val_acc = 0.0
# best_model_wts = copy.deepcopy(model.state_dict())
# for epoch in range(n_epoch):
#     model.train()
#     train_batch_accs = []
#     train_batch_loss = []
#     # accs is a list containing all the accuracies during each back propagation
#     # epoch_loss is a list including all the losses
#     for train_i in training_idx:
#         data = dataset[train_i]
#         data = data.to(device)
#         optimizer.zero_grad()
#         train_out = model(data.x, data.edge_index, data.edge_attr)
#         # data.y is the groundturth and needs to be transferred to the correct shape
#         loss = F.nll_loss(train_out, data.y.view(-1).type(torch.long))
#         loss.backward()
#         optimizer.step()
#         train_batch_loss.append(loss.data.cpu().numpy())
#         # out_put.max(1)[1] provides the category with the largest score
#         # namely the predicted category
#         acc = train_out.max(1)[1].eq(data.y.view(-1).type(torch.long))
#         train_batch_accs.append(acc.cpu().numpy())
#     train_accs.append(np.array(train_batch_accs).mean())
#     train_loss.append(np.array(train_batch_loss).mean())
#     model.eval()
#     val_batch_acc= []
#     with torch.no_grad():
#         for val_i in val_idx:
#             data = dataset[val_i]
#             data = data.to(device)
#             val_out = model(data.x, data.edge_index, data.edge_attr)
#             acc = val_out.max(1)[1].eq(data.y.view(-1).type(torch.long))
#             val_batch_acc.append(acc.cpu().numpy())
#     val_batch_mean = np.array(val_batch_acc).mean()
#     val_accs.append(val_batch_mean)
#     print("epoch: ", epoch, "  val-batch-accuracy: ", val_batch_mean)
#     if val_batch_mean > best_val_acc:
#         best_model_wts = copy.deepcopy(model.state_dict())
# torch.save(best_model_wts, 'best_acc.pt')
# save the plot of the gradient of correct score w.r.t. node feature matrix
# should be incorporated later in the fintuning
# model.eval()
# for i in range(len(dataset)):
#     data_plot = dataset[i]
#     data_plot = data_plot.to(device)
#     data_plot.x.requires_grad_()
#     plot_out = model(data_plot.x, data_plot.edge_index, data_plot.edge_attr)
#     plot_true = plot_out[:, data_plot.y.view(-1).type(torch.long)]
#     plot_true.backward(retain_graph=True)
#     y_axis = np.arange(data_plot.x.shape[0])
#     x_axis = np.arange(data_plot.x.shape[1])
#     x_axis, y_axis = np.meshgrid(x_axis, y_axis)
#     plt.figure()
#     plt.contourf(x_axis, y_axis, data_plot.x.grad.cpu().numpy(), cmap='RdGy')
#     plt.colorbar()
#     plt.savefig(fname=join('gradient_plot', str(i)))
#     plt.close()
# the following plots the loss and accuracy curve of train and validation
# plt.figure()
# plt.subplot(2, 1, 1)
# plt.plot(np.arange(n_epoch), train_accs, color='blue', label='train')
# plt.plot(np.arange(n_epoch), val_accs, color='red', label='val')
# plt.ylabel("accuracy")
# plt.legend()
# plt.subplot(2, 1, 2)
# plt.plot(np.arange(n_epoch), train_loss)
# plt.xlabel("epoch")
# plt.ylabel("training loss")
