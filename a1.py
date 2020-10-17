import numpy as np
import matplotlib.pyplot as plt


i1 = 'C:/COMP472_A1/Assig1-Dataset/Assig1-Dataset/info_1.csv'
tn1 = 'C:/COMP472_A1/Assig1-Dataset/Assig1-Dataset/test_no_label_1.csv'
tw1 = 'C:/COMP472_A1/Assig1-Dataset/Assig1-Dataset/test_with_label_1.csv'
t1 = 'C:/COMP472_A1/Assig1-Dataset/Assig1-Dataset/train_1.csv'
v1 = 'C:/COMP472_A1/Assig1-Dataset/Assig1-Dataset/val_1.csv'




data = np.loadtxt(tn1, delimiter=",", skiprows=0)


data = data[2:3, :]
data = np.array(data)
data = data.reshape((32, 32))
print(data.shape)
plt.imshow(data)
plt.show()
