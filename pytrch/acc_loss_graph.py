import matplotlib.pyplot as plt
from matplotlib import style

style.use("ggplot")

# Model name from CSV file
model_name = "model-1593333575"


def create_acc_loss_graph(model_name):
    # Open CSV file with logs (one line per log statement)
    contents = open("model.log", "r").read().split("\n")

    times = []
    accuracies = []
    losses = []

    val_accs = []
    val_losses = []

    # Iterate through each line in file
    for c in contents:
        if model_name in c:
            # Append data to array(s)
            name, timestamp, acc, loss, val_acc, val_loss, epoch = c.split(",")

            times.append(float(timestamp))
            accuracies.append(float(acc))
            losses.append(float(loss))

            val_accs.append(float(val_acc))
            val_losses.append(float(val_loss))

    plt.figure()
    # Chart 1
    ax1 = plt.subplot2grid((2, 1), (0, 0))
    # Chart 2, share axis with chart 1
    ax2 = plt.subplot2grid((2, 1), (1, 0), sharex=ax1)

    # Chart 1: x = timestamp, y = accuracies/val_accs
    ax1.plot(times, accuracies, label="acc")
    ax1.plot(times, val_accs, label="val_acc")
    ax1.legend(loc=2)

    # Chart 1: x = timestamp, y = losses/val_loss
    ax2.plot(times, losses, label="loss")
    ax2.plot(times, val_losses, label="val_loss")
    ax2.legend(loc=2)

    plt.show()


create_acc_loss_graph(model_name)
