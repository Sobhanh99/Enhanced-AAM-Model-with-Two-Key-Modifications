#texture utils


def compute_mean_texture(data):
    N = data.shape[0]
    H = data.shape[1]
    W = data.shape[2]

    # Parameters for lighting normalization
    alpha = np.ones((N, 1))  # Scaling
    beta = np.zeros((N, 1))  # Offset

    # initialize one of the images as the mean to begin the optimization process
    mean_texture = np.expand_dims(data[np.random.randint(N)], 0)

    prev_texture = mean_texture

    counter = 0
    while True:
        # Compute the scaling and offset parameters
        beta = np.mean(data, axis=(1, 2, 3), keepdims=True)
        alpha = np.sqrt(np.mean((data - beta) ** 2, axis=(1, 2, 3), keepdims=True))

        # Normalize the data
        normalized_data = (data - beta) / alpha

        # Compute the new mean texture
        mean_texture = np.mean(normalized_data, axis=0, keepdims=True)

        # Standardize the mean texture
        mean_texture -= np.mean(mean_texture, axis=(1, 2, 3), keepdims=True)
        mean_texture /= np.sqrt(np.mean(mean_texture ** 2, axis=(1, 2, 3), keepdims=True))

        if np.linalg.norm(prev_texture - mean_texture) < 0.0000001:
            break

        prev_texture = mean_texture
        counter += 1

    print('Texture mean found in {} iterations'.format(counter))
    normalized_mean = (mean_texture - np.min(mean_texture)) / (np.max(mean_texture) - np.min(mean_texture))

    return normalized_mean, beta, alpha, normalized_data
