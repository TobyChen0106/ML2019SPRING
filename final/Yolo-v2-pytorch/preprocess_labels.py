import pandas as pd

train_labels = pd.read_csv('data/train_labels.csv').values

new_train_labels = []
num_train_labels = len(train_labels)
i=0
while i < num_train_labels:
    objects = []
    l = int(train_labels[i, 5])
    if l == 1:
        img_name = train_labels[i, 0]
        while i < num_train_labels and train_labels[i, 0] == img_name:

            x1 = int(train_labels[i, 1])
            y1 = int(train_labels[i, 2])
            w1 = int(train_labels[i, 3])
            h1 = int(train_labels[i, 4])
            # l = int(train_labels[i, 5])

            bbox_1 = [x1, y1, x1+w1, y1+h1, l]
            objects.append(bbox_1)
            i+=1
    else:
        i+=1
    new_train_labels.append(objects)
    


for i in range(len(new_train_labels)):
    print(i,' ',new_train_labels[i])


