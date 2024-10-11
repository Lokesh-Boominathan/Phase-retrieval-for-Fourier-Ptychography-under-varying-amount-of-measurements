from scipy.misc import imsave
import numpy as np

for i in range(149):
	amp=np.load("./test/results/test_{:02d}.npy".format(i+1))
	gt=np.load("./test/gt/test_{:02d}.npy".format(i+1))
	for j in range(1):
		imsave("./results/amp/deep_{:02d}_{:02d}.png".format(i+1,j+1),255*np.squeeze(amp[j,:,:,0]))
		imsave("./results/amp/gt_{:02d}_{:02d}.png".format(i+1,j+1),255*np.squeeze(gt[j,:,:,0]))
		imsave("./results/phase/deep_{:02d}_{:02d}.png".format(i+1,j+1),255*np.squeeze(amp[j,:,:,1]))
		imsave("./results/phase/gt_{:02d}_{:02d}.png".format(i+1,j+1),255*np.squeeze(gt[j,:,:,1]))

