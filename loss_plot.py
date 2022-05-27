import matplotlib.pyplot as plt
import pickle5 as pickle
"""
train_loss = [0.10108,0.03840,0.03816,0.03806,0.03797]
with open("./first_try_data/train_loss"+str(0)+".pkl", "wb") as f:
    pickle.dump(train_loss,f,protocol=pickle.HIGHEST_PROTOCOL)
"""

def lossPlot(loss,dir,i):
    fig = plt.figure()
    plt.plot(loss,label = "loss for training set")
    plt.title("submodel "+str(i))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(dir+"loss_curves_"+str(i)+".png")

with open("./first_try_data/train_loss"+str(0)+".pkl","rb") as f:
    train_loss = pickle.load(f)
lossPlot(train_loss, "./first_try_data/",0)
for i in range(1,10):
    with open("./first_try_data/train_loss"+str(i)+".pkl","rb") as f:
        train_loss = pickle.load(f)
        
    train_loss = [loss.item() for loss in train_loss]
    lossPlot(train_loss, "./first_try_data/", i)
    

