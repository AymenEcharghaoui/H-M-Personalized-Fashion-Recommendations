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
    
def lossPlot2(train_loss, test_loss, dir, i):
    fig = plt.figure()
    plt.plot(train_loss,label = "loss for training set")
    plt.plot(test_loss,label = "loss for validation set")
    plt.title("submodel "+str(i))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(dir+"loss_curves_4alexnet_"+str(i)+".png")


# with open("./first_try_data/train_loss"+str(0)+".pkl","rb") as f:
#     train_loss = pickle.load(f)
# lossPlot(train_loss, "./first_try_data/",0)
# for i in range(1,10):
#     with open("./first_try_data/train_loss"+str(i)+".pkl","rb") as f:
#         train_loss = pickle.load(f)
        
#     train_loss = [loss.item() for loss in train_loss]
#     lossPlot(train_loss, "./first_try_data/", i)

test_loss = [[0.04611,0.04621,0.04565,0.04510,0.04527],
             [0.02769,0.02689,0.02579,0.02738,0.02742],
             [0.03507,0.03374,0.03451,0.03509,0.03390],
             [0.03531,0.03652,0.03472,0.03435,0.03422],
             [0.03423,0.03400,0.03337,0.03380,0.03316],
             [0.02548,0.02438,0.02437,0.02290,0.02318],
             [0.02035,0.02046,0.02071,0.02063,0.01906],
             [0.01868,0.01759,0.01653,0.01744,0.01628],
             [0.02199,0.02140,0.02166,0.02160,0.02060],
             [0.03197,0.03128,0.03103,0.03132,0.03390]
             ]

for i in range(10):
    with open("./second_try_data/second_try_train_loss"+str(i)+".pkl","rb") as f:
        train_loss = pickle.load(f)
        
    train_loss = [loss.item() for loss in train_loss]
    lossPlot2(train_loss, test_loss[i], "./second_try_data/", i)
