import numpy as np
#import matplotlib.pyplot as plt
import sys
import os



# Make sure that caffe is on the python path:
#caffe_root = $CAFFE_ROOT #'/usr/local/caffe/'  # this file is expected to be in {caffe_root}/examples/caffe-test-mnist-jpg/

#sys.path.insert(0, caffe_root + 'python')
import caffe

#plt.rcParams['figure.figsize'] = (10, 10)
#plt.rcParams['image.interpolation'] = 'nearest'
#plt.rcParams['image.cmap'] = 'gray'


#if not os.path.isfile(caffe_root + 'examples/mnist/lenet_iter_10000.caffemodel'):
#    print("Cannot find trained caffemodel...")

# load trained network
caffe.set_mode_cpu()
net = caffe.Net('haddoc2/example/caffe/lenet_deploy.prototxt',
                'haddoc2/example/caffe/lenet_iter_10000.caffemodel',
                caffe.TEST)


# set net to batch size of 50
#net.blobs['data'].reshape(2,1,28,28)


# load image digit 
#Xdata0= caffe.io.load_image('nr.jpg')
#Xdata0= caffe.io.load_image('img_arr.npy')
Xdata0 = np.load('img_arr.npy')

maxVal = Xdata0.max()
for i in range(0,28):
    for j in range(0,28):
        Xdata0[i,j] = Xdata0[i,j]/maxVal 
    
print(Xdata0.max())


#Xdata1= caffe.io.load_image('./images/one.jpg')
#Xdata7= caffe.io.load_image('./images/seven.jpg')
#Xdata6= caffe.io.load_image('./images/six.jpg')
#Xdata3= caffe.io.load_image('./images/three.jpg')
#Xdata4= caffe.io.load_image('./images/four.jpg')

# make sure dimension is correct: 28*28*3
print('data dim:', Xdata0.shape)#,Xdata1.shape)

#
#temp1=Xdata0[:,:,0];
temp1 = Xdata0
print(temp1.shape)
#temp=temp1.reshape(len(temp1),28,28)
temp=temp1.reshape(1,1,28,28)
print(temp)


# feed data to the network
net.blobs['data'].data[...] = temp;
#net.blobs['data'].data[...] = Xdata0;

# perfrom forward operation
out = net.forward() 

#print(out.argmax())
#print(out.argmin())
#print(max(out))
#print(out['pool2'].max())
#print(out['pool2'].min())

#l_idx = list(net._layer_names).index('pool2')

#tops = [(net._blob_names[bi], net.blobs[net._blob_names[bi]].data.shape) 
#        for bi in list(net._top_ids(l_idx))]

#for tn in tops:
#    print(tn)
'''
    print(net.blobs[tn].data.shape)
'''
  #print("output name % s has shape % d" %(tn, net.blobs[tn].data.shape))

#print(out.shape)
#print(out)

print("Predicted class is #{}.".format(out['prob'].argmax()))