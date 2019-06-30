import pandas as pd
import numpy as np

submissions = ["submission.20.csv", "submission.18.csv", "submission.17.csv"]

train_labels = []
for submit in submissions:
    train_labels.append(pd.read_csv(submit).values)

for i in range(len(train_labels[0])):
    no_tumor_count = 0
    for train_label in train_labels:
        if train_label[i][1] == "1 1":
            no_tumor_count+=1
    
    if no_tumor_count/3 > 0.5:
        train_labels[0][i][1] = "1 1"
    else:
        if train_labels[0][i][1] == "1 1":
            train_labels[0][i][1] = train_labels[1][i][1]
        

    

result_csv = np.array(train_labels[0])
print(result_csv.shape)
np.savetxt('ensemble.csv', result_csv, delimiter=",", fmt="%s")