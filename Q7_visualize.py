import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Because I only had to train 4x, I graphed the accuracies and losses manually. 
plt.figure()

# Plot the Accuracies as a function of training instances 
plt.plot([3188,6375,12750,25500,51000],[0.9322459222,0.95890196078,0.97050980392,0.9794117647,0.98425490196])
plt.plot([3188,6375,12750,25500,51000],[0.9276,0.9525,0.9683,0.9779,0.9822])
plt.title("Training and Test Accuracies vs. Number of Training Instances")
plt.xlabel("Number of Training Instances")
plt.ylabel("Accuracy")
plt.legend(["Training Accuracy","Test Accuracy"])

plt.figure()
plt.plot(np.log([3188,6375,12750,25500,51000]),np.log([0.2229,0.1427,0.1030,0.0657,0.0515]))
plt.plot(np.log([3188,6375,12750,25500,51000]),np.log([0.2514,0.1570,0.1054,0.0691,0.0547]))
plt.title("Training and Test Losses vs. Number of Training Instances")
plt.xlabel("Log Number of Training Instances")
plt.ylabel("Log Loss")
plt.legend(["Log Training Loss","Log Test Loss"])
plt.show()