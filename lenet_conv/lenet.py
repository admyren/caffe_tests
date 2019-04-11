import numpy as np
import sys
import os
import caffe

# load trained network
caffe.set_mode_cpu()
net = caffe.Net('lenet_deploy.prototxt',
                'lenet_iter_2000.caffemodel',
                caffe.TEST)


# load numpy array holding digit
Xdata0 = np.load('img_test/9.npy')

# Normalize
maxVal = Xdata0.max()
for i in range(0,28):
    for j in range(0,28):
        Xdata0[i,j] = Xdata0[i,j]/maxVal 
    
print(Xdata0.max())
print(Xdata0.min())

# make sure dimension is correct: 28*28*3
print('data dim:', Xdata0.shape)

temp1 = Xdata0
print(temp1.shape)
temp=temp1.reshape(1,1,28,28)
#print(temp)

# feed data to the network
net.blobs['data'].data[...] = temp;



# perfrom forward operation
out = net.forward() 




print(out['prob']);
#out_rounded = np.floor(out['prob'])
#print(out_rounded)


#l_idx = list(net._layer_names).index('pool2')

#tops = [(net._blob_names[bi], net.blobs[net._blob_names[bi]].data.shape) 
#        for bi in list(net._top_ids(l_idx))]

#for tn in tops:
#    print(tn)
'''
    print(net.blobs[tn].data.shape)
'''
  #print("output name % s has shape % d" %(tn, net.blobs[tn].data.shape))


print("Predicted class is #{}.".format(out['prob'].argmax()))
