import csv
import matplotlib.pyplot as plt
import collections

i1 = "Assig1-Dataset/Assig1-Dataset/info_1.csv"
i2 = "Assig1-Dataset/Assig1-Dataset/info_2.csv"
tn1 = "Assig1-Dataset/Assig1-Dataset/test_no_label_1.csv"
tw1 = "Assig1-Dataset/Assig1-Dataset/test_with_label_1.csv"
tw2 = "Assig1-Dataset/Assig1-Dataset/test_with_label_2.csv"
t1 = "Assig1-Dataset/Assig1-Dataset/train_1.csv"
v1 = "Assig1-Dataset/Assig1-Dataset/val_1.csv"



def getDistribution(filePath):
    distribution_list = []     #create list
    with open(filePath) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        row_count = 0
        for row in csv_reader:
            distribution_list.append(row[1024])
            row_count+=1
            distribution_list = list(map(int, distribution_list))
        return distribution_list

def indexToLetter(distribution_list, indexFile):
    with open(indexFile) as csv_file:
        index = csv.reader(csv_file, delimiter=',')
        rows = list(index)
        distribution_list.sort()   #sort to make easier to visualize
        for i in range(len(distribution_list)):
            distribution_list[i] = rows[distribution_list[i]+1][1]
    return distribution_list


data = collections.Counter(indexToLetter(getDistribution(t1), i1))

plt.bar(data.keys(), data.values())
plt.xlabel('Letters')
plt.ylabel('Occurence')
plt.title('Letters by Occurence')
plt.show()