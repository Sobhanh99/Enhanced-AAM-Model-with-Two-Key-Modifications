from visualize_pointset import plot_pointset_with_connections, get_coordinates
from shape_utils import compute_preshape_space, compute_optimal_rotation
from texture_utils import *
from combine_variation_modes import combine_shape_with_normalized_texture
from image_warp import apply_shape

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.color import rgb2gray

# To get largest t eigenvalues
def findt(eig_values, percent):

	total = sum(eig_values)
	total = percent * total
	p = 0
	t = 0
	for i in range(len(eig_values)):
		p = p + eig_values[i]
		if p >= total:
			t = i
			break
	return t

def get_rrmse(A,B):
	rrmse = np.sqrt(np.sum((A-B)**2)/np.sum(A**2))
	return rrmse

def fit_shape(test_pointset_data, mean, cov_matrix, eig_values, eig_vecs, connect_from, connect_to, save_dir, number_of_images, number_of_landmarks):

	eig_values = np.real(eig_values)
	eig_vecs = np.real(eig_vecs)

	t = findt(eig_values, 0.9999)
	t = t + 1

	s0 = mean[0]
	s=test_pointset_data
	for i in range(test_pointset_data.shape[0]):
		s[i] = compute_preshape_space(test_pointset_data[i])

	for i in range(number_of_images):   
		R = compute_optimal_rotation(s[i],s0)
		s[i] = np.matmul(R,s[i])

	phi = eig_vecs[:t]
	phi = np.transpose(phi)

	rrmse_values = []
	test_set_params = np.zeros((test_pointset_data.shape[0], t))

	for i in range(number_of_images):   

		y = s[i] - s0
		y = np.array(y)
		y = np.reshape(y,(number_of_landmarks * 2,1))


		b = np.linalg.lstsq(phi,y)

		sfit = s0 + np.reshape(np.matmul(phi,b[0]), (number_of_landmarks,2))
		sorg = s[i]

		rrmse = get_rrmse(sorg, sfit)
		rrmse_values.append(rrmse)
		test_set_params[i] = b[0].squeeze()
		if i>20:
			continue

		plt.figure(facecolor='white')
		plt.axis([-0.5, 0.5, -0.5, 0.5])
		plt.grid(False)
		plt.scatter(sorg[:,0]*3, sorg[:,1]*3,s=1, label='original annotation', color='red')
		plt.scatter(sfit[:,0]*3, sfit[:,1]*3,s=1, label='shape model fit', color='blue')
		plt.title("Comparing fit against annotation")
		plt.legend()
		plt.savefig(os.path.join(save_dir, 'shape_model_fit_{}.png'.format(i)))
		plt.clf()

	np.save("rrmse_shape_rgb",rrmse_values)
	print('Average Reconstruction Error on Test Set: Mean: {} Std-Dev: {}'.format(np.mean(rrmse_values), np.std(rrmse_values)))
	np.save(os.path.join(save_dir, 'shape_model_fit_param.npy'), test_set_params)


def fit_texture(test_texture_data, mean, cov_matrix, eig_values, eig_vecs, connect_from, connect_to, save_dir):

	t = findt(eig_values, 0.98)
	eig_values = np.real(eig_values[0:t])
	eig_vecs = np.real(eig_vecs[:, 0:t])
	print("current_results\\mean texture: ",mean.shape)

	beta = np.mean(test_texture_data, axis=(1,2,3), keepdims=True)
	alpha = np.mean(test_texture_data*mean, axis=(1,2,3), keepdims=True)

	test_texture_data = (test_texture_data-beta)/alpha

	rrmse_values = []
	test_set_params = np.zeros((test_texture_data.shape[0], t))

	for i in range(test_texture_data.shape[0]):

		model_fit_coeff = np.dot(np.linalg.pinv(eig_vecs), test_texture_data[i].reshape(mean.size,1) - mean.reshape(mean.size,1) )
		texture_recon0 = (mean.reshape(mean.size,1) + np.dot(eig_vecs, model_fit_coeff)).reshape(mean.shape).squeeze(0)
		np.save("texture_recon",texture_recon0)
		texture_recon = (texture_recon0 - texture_recon0.min()) / (texture_recon0.max() - texture_recon0.min())
		test_texture_data[i] = (test_texture_data[i] - test_texture_data[i].min()) / (test_texture_data[i].max() - test_texture_data[i].min())

		rrmse = get_rrmse(test_texture_data[i], texture_recon)
		rrmse_values.append(rrmse)
		test_set_params[i] = model_fit_coeff.squeeze()
		if i>9:
			continue

		f, axarr = plt.subplots(1,2)
		print("test_texture_data: ",test_texture_data[i].shape)
		axarr[0].imshow(cv2.resize(test_texture_data[i], (test_texture_data[i].shape[0]*4,test_texture_data[i].shape[1]*4 )))
		axarr[0].title.set_text('Test Image')
		axarr[0].axis('off')
		print("texture_recon: ",texture_recon.shape)
		axarr[1].imshow(cv2.resize(texture_recon, (texture_recon.shape[0]*4,texture_recon.shape[1]*4 )))
		axarr[1].title.set_text('Model Fit')
		axarr[1].axis('off')

		plt.subplots_adjust(wspace=None, hspace=None)
		plt.suptitle('Analysis of Texture Model fitting: Test Image: {}'.format(i))
		plt.savefig(os.path.join(save_dir, 'texture_model_fit_{}.png'.format(i)), bbox_inches='tight')
		plt.clf()

	np.save("rrmse_texture_gray",rrmse_values)
	print('Average Reconstruction Error on Test Set: Mean: {} Std-Dev: {}'.format(np.mean(rrmse_values), np.std(rrmse_values)))
	np.save(os.path.join(save_dir, 'texture_model_fit_param.npy'), test_set_params)

	reconstruct_image(shape_recon, texture_recon, original_pointset, original_image, shape_mean_x_coord, shape_mean_y_coord)
