import matplotlib.pyplot as plt
import numpy as np
#

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

plt.rcParams['axes.facecolor']='white'
plt.rcParams['figure.facecolor']='white'
plt.rcParams['figure.edgecolor']='white'
plt.rcParams['savefig.facecolor']='white'
plt.rcParams['savefig.edgecolor']='white'
plt.rcParams['axes.labelsize']=14
plt.rcParams['xtick.labelsize']=14
plt.rcParams['ytick.labelsize']=14
plt.rcParams['axes.titlesize']=14
plt.rcParams['legend.framealpha']=1
plt.rcParams['legend.fontsize']=13
plt.rcParams['axes.xmargin']=0.2
plt.rcParams['axes.ymargin']=0.2
plt.rcParams['axes.autolimit_mode']='round_numbers'

def read_file(path):
    f = open(path, "r")
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []

    for line in f:
        a = line.split(" ")
        if a[0] == "train":
            train_loss.append(float(a[2]))
            train_acc.append(float(a[4]))
        elif a[0] == "val":
            val_loss.append(float(a[2]))
            val_acc.append(float(a[4]))

    return train_loss, train_acc, val_loss, val_acc

_, _, val_loss0, val_acc0 = read_file("google/SFL_google.txt")
_, _, val_loss3, val_acc3 = read_file("google/HFL_google_g2_decay.txt")
_, _, val_loss7, val_acc7 = read_file("google/HFL_google_g5_decay.txt")

val_loss0 = val_loss0[:200]
val_acc0 = val_acc0[:200]

N_EPOCHS = 200
N_EPOCHS_FL = 200
N_clients = 10
N_TX_FL = N_clients*2 
clients_PER_GROUP = 10
N_GROUPS = int(N_clients/clients_PER_GROUP)
N_TX_EAGLE2 = (2+1)*5
N_TX_EAGLE5 = (5+1)*2


plt.figure()
plt.plot(val_loss0, label='FL - 10 clients', zorder=5)
plt.plot(val_loss3, "tab:orange",label='EAGLE - 10 clients, groups of 2')
plt.plot(val_loss7, "tab:red",label='EAGLE - 10 clients, groups of 5')

plt.legend()
plt.grid()
plt.xlabel("Number of rounds")
plt.ylabel("Loss function value")
plt.xlim(0,200)
# plt.xlim(-2,200)
plt.savefig("figures/google_iid_loss.png")
plt.show()

#############################################


plt.figure()
plt.plot(val_acc0, label='FL - 10 clients', zorder=5)
plt.plot(val_acc3, "tab:orange",label='EAGLE - 10 clients, groups of 2')
plt.plot(val_acc7, "tab:red",label='EAGLE - 10 clients, groups of 5')

plt.legend()
plt.grid()
plt.yticks(np.arange(0, 0.9, 0.1))
plt.xlabel("Number of rounds")
plt.ylabel("Accuracy level")
plt.xlim(0,200)
plt.savefig("figures/google_iid_round.png")
plt.show()

#############################################

plt.figure()
plt.plot(val_acc0, label='FL - 10 clients', zorder=5)
plt.plot(np.arange(0,N_EPOCHS*2,2), val_acc3, "tab:orange",label='EAGLE - 10 clients, groups of 2')
plt.plot(np.arange(0,N_EPOCHS*5,5), val_acc7, "tab:red",label='EAGLE - 10 clients, groups of 5')


plt.legend()
plt.grid()
#plt.xticks(range(0, 101, 5))
plt.yticks(np.arange(0, 0.9, 0.1))
plt.xlabel("Number of epochs")
plt.ylabel("Accuracy level")
plt.xlim(0,1000)
plt.savefig("figures/google_iid_epochs.png")

plt.show()

#####################


plt.figure()
plt.plot(np.arange(N_TX_FL,N_TX_FL*(N_EPOCHS_FL+1),N_TX_FL),val_acc0, label='FL - 10 clients', zorder=5)
plt.plot(np.arange(N_TX_EAGLE2,N_TX_EAGLE2*(N_EPOCHS+1),N_TX_EAGLE2), val_acc3, "tab:orange",label='EAGLE - 10 clients, groups of 2')
plt.plot(np.arange(N_TX_EAGLE5,N_TX_EAGLE5*(N_EPOCHS+1),N_TX_EAGLE5), val_acc7, "tab:red", label='EAGLE - 10 clients, groups of 5')

plt.legend()
plt.grid()

plt.yticks(np.arange(0, 0.9, 0.1))
plt.xlabel("Number of transmissions")
plt.ylabel("Accuracy level")
plt.xlim(0,4000)
plt.savefig("figures/google_iid_tx.png")
plt.show()

