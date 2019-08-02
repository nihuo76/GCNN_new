from hamiltonian import Hamiltonian
import numpy as np
import matplotlib.pyplot as plt
from TrainValidation import train_val
import time

y_threshold = 0

dataset = Hamiltonian(root='data/hamiltonMER', k_n=1, y_cut=y_threshold)
# N is number of samples
N = len(dataset)
# idx is the index of the dataset that is going to be splited
idx = np.arange(N)
# random shuffle the idx dataset
np.random.shuffle(idx)

# get the random guess probability
counter = 0
for j in range(N):
    data = dataset[j]
    if data.y > y_threshold:
        counter += 1

random_guess = max(counter/N, 1-counter/N)

crossva = [None]*5
crossva[0], crossva[1], crossva[2], crossva[3], crossva[4] = np.array_split(idx, 5)

n_epoch = 5
lr = 0.001

train_accs = []
val_accs = []
train_loss = []
drop_p = 0.4

since = time.time()
for i in range(5):
    val_idx = crossva[i]
    train_idx = np.setdiff1d(idx, val_idx)
    train_list, val_list, loss_list = train_val(n_epoch=n_epoch, lr_input=lr,
                                                dataset=dataset, training_idx=train_idx,
                                                val_idx=val_idx, rd=drop_p,
                                                method="Peng")
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
random_guess = np.tile(random_guess, n_epoch)


# plot the average accuracy of the 5-fold for
# both train and validation
plt.figure()
plt.plot(np.arange(n_epoch), train_mean, color='green', label='train')
plt.plot(np.arange(n_epoch), val_mean, color='red', label='val')
plt.plot(np.arange(n_epoch), random_guess, color='blue', linestyle=':', label='guess')
# plt.errorbar(np.arange(n_epoch), train_mean, yerr=train_std, fmt='o',
#              color='green', alpha=0.1, elinewidth=1)
# plt.errorbar(np.arange(n_epoch), val_mean, yerr=val_std, fmt='o',
#              color='red', alpha=0.1, elinewidth=1)
plt.title('lr='+str(lr))
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend()
# plt.savefig(fname='Accruacy'+str(lr).replace('.', '')+str(reg_L1).replace('.', ''))
plt.savefig(fname='Accruacy_Peng')
plt.close()

# plot the average loss across 5-fold during training
plt.figure()
loss_mean = np.array(train_loss).mean(0)
loss_std = np.array(train_loss).std(0)
plt.plot(np.arange(n_epoch), loss_mean, color='blue')
# plt.errorbar(np.arange(n_epoch), loss_mean, yerr=loss_std, fmt='o',
#              color='blue', alpha=0.1, elinewidth=1)
plt.ylabel("loss")
plt.title('lr='+str(lr))
plt.xlabel("epoch")
# plt.savefig(fname='Loss'+str(lr).replace('.', '')+str(reg_L1).replace('.', ''))
plt.savefig(fname='Loss_Peng')
plt.close()
print("finish successfully")
print('Training complete in {:.0f}h'.format(time_elapsed // 3600))
