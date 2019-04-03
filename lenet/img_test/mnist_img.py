import numpy as np
from matplotlib import pyplot as plt
import csv

digit = 10

results = []
with open("mnist_test.csv") as csvfile:
    reader = csv.reader(csvfile) # change contents to floats
    for row in reader: # each row is a list
        results.append(row)
        
b = np.array(results)
a = b[digit:digit+1, 0:785]

img_arr = np.zeros((28,28))

for x in range(0, 28):
    img_arr[x:x+1, :28] = a[:1, (x)*28:(x+1)*28]
    
filename = a[0,0] + '.jpg'
print(a[0,0])
plt.imsave(filename, img_arr)

np.save(a[0,0], img_arr)
