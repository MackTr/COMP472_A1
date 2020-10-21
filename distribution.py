import csv
import matplotlib.pyplot as plt
import collections

i1 = 'Assig1-Dataset/Assig1-Dataset/info_1.csv'
tn1 = 'Assig1-Dataset/Assig1-Dataset/test_no_label_1.csv'
tw1 = 'Assig1-Dataset/Assig1-Dataset/test_with_label_1.csv'
t1 = 'Assig1-Dataset/Assig1-Dataset/train_1.csv'
v1 = 'Assig1-Dataset/Assig1-Dataset/val_1.csv'

i2 = 'Assig1-Dataset/Assig1-Dataset/info_2.csv'
tn2 = 'Assig1-Dataset/Assig1-Dataset/test_no_label_2.csv'
tw2 = 'Assig1-Dataset/Assig1-Dataset/test_with_label_2.csv'
t2 = 'Assig1-Dataset/Assig1-Dataset/train_2.csv'
v2 = 'Assig1-Dataset/Assig1-Dataset/val_2.csv'


def getDistribution(filePath):
    distribution_list = []     #create list
    with open(filePath) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        row_count = 0
        for row in csv_reader:
            distribution_list.append(row[1024])
            row_count+=1
            distribution_list = list(map(int, distribution_list))
        return distribution_list  #return list with last parameter of files with labels. The list will then be of all the labels in order

def indexToLetter(distribution_list, indexFile):
    with open(indexFile) as csv_file:
        index = csv.reader(csv_file, delimiter=',') #we then read the index file to transform the label from its code to letter ex.: 0 -> A
        rows = list(index)
        distribution_list.sort()   #sort to make easier to visualize
        for i in range(len(distribution_list)):
            distribution_list[i] = rows[distribution_list[i]+1][1]   #here the item in the list is changed to its letter counterpart
    return distribution_list


data = collections.Counter(indexToLetter(getDistribution(v2), i2))   # The dataset is then created with the list and its occurence.
                                                                     # instead of ex.: A, A, A -> A:3
plt.bar(data.keys(), data.values()) #the data is then plotted here, data.keys() is the letter, and data.values() is the occurence.
plt.xlabel('Letters')
plt.ylabel('Occurence')
plt.title('Letters by Occurence')
plt.show()
