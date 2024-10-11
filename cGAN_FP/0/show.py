import numpy as np
import matplotlib.pyplot as plt
 
y_pred=np.load('/media/data/lokesh/rahul/unet/test/test_0001.npy')
y_test=np.load('/media/data/lokesh/rahul/unet/datasets/fpm/test/3.npy')

plt.imshow(np.squeeze(X[1,:,:,:]))
plt.show()
plt.imshow(np.squeeze(X[1,:,:,:]),cmap='gray')
plt.show()
