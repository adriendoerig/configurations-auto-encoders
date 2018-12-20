# cf https://github.com/despoisj/LatentSpaceVisualization
# and https://hackernoon.com/latent-space-visualization-deep-learning-bits-2-bd09a46920df

import tensorflow as tf
import numpy as np
import cv2, random, string, os, imageio
from lapjv import lapjv
from sklearn import manifold
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from tensorflow.python.keras.preprocessing import image
from create_sprite import images_to_sprite
from scipy.spatial.distance import cdist


# Show every image, good for picking interplation candidates
def visualizeDataset(X):
    for i, image in enumerate(X):
        cv2.imshow(str(i), image)
        cv2.waitKey()
        cv2.destroyAllWindows()


# the following function makes tf.dataset_tests from numpy batches. if latent_input is not None, also provides a latent input to decode from
def input_fn_pred(batch, latent_input=None):
    batch_size = batch.shape[0]
    batch = tf.convert_to_tensor(batch, dtype=tf.float32)
    if latent_input is None:
        dataset_test = tf.data.Dataset.from_tensor_slices(batch)
    else:
        user_latent_input = tf.convert_to_tensor(latent_input, dtype=tf.float32)
        dataset_test = tf.data.Dataset.from_tensor_slices((batch, user_latent_input))

    dataset_test = dataset_test.batch(batch_size, drop_remainder=True)
    # Use pipelining to speed up things (see https://www.youtube.com/watch?v=SxOsJPaxHME)
    dataset_test = dataset_test.prefetch(2)
    # Create an iterator for the dataset_test and the above modifications.
    iterator = dataset_test.make_one_shot_iterator()

    if latent_input is None:
        # Get the next batch of images and labels.
        images = iterator.get_next()
        feed_dict = {'images': images}
    else:
        # Get the next batch of images and labels.
        [images, user_latent_input] = iterator.get_next()
        feed_dict = {'images': images,
                     'user_latent_input': user_latent_input}
    return feed_dict


# Reconstructions for samples in dataset
def getReconstructedImages(X, im_size, model):
    nbSamples = X.shape[0]
    nbSquares = int(np.sqrt(nbSamples))
    nbSquaresHeight = 2 * nbSquares
    nbSquaresWidth = nbSquaresHeight
    resultImage = np.zeros((nbSquaresHeight * im_size[0], int(nbSquaresWidth * im_size[1] * (3 / 2) / 2), X.shape[-1]))

    model_out = list(model.predict(input_fn=lambda: input_fn_pred(X)))
    reconstructedX = np.array([p["reconstructions"] for p in model_out])
    differences = abs(X - reconstructedX)

    for i in range(nbSamples):
        original = X[i]
        reconstruction = reconstructedX[i]
        difference = differences[i]
        rowIndex = i % nbSquaresWidth
        columnIndex = (i - rowIndex) // nbSquaresHeight
        resultImage[rowIndex * im_size[0]:(rowIndex + 1) * im_size[0],
        columnIndex * 3 * im_size[1]:(columnIndex + 1) * 3 * im_size[1], :] = np.hstack(
            [original, reconstruction, difference])

    return resultImage


# Reconstructions for samples in dataset
def visualizeReconstructedImages(X, im_size, model, save=False):

    np.random.shuffle(X)

    print("Generating image reconstructions...")
    reconstruction = getReconstructedImages(X, im_size, model)
    result = np.hstack([reconstruction, np.zeros([reconstruction.shape[0], 5, reconstruction.shape[-1]])])

    if save is not False:
        cv2.imwrite(save + "reconstructions.png", result)
    else:
        plt.figure()
        plt.imshow(np.squeeze(result), cmap="binary_r")
        plt.title('Original, reconstruction, abs(difference)')
        plt.show()
        plt.close()


# Scatter with images instead of points
def imscatter(x, y, ax, imageData, im_size, zoom):
    images = []
    for i in range(len(x)):
        x0, y0 = x[i], y[i]
        # Convert to image
        img = imageData[i] * 255.
        img = img.astype(np.uint8).reshape([im_size[0], im_size[1]])
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        # Note: OpenCV uses BGR and plt uses RGB
        image = OffsetImage(img, zoom=zoom)
        ab = AnnotationBbox(image, (x0, y0), xycoords='data', frameon=False)
        images.append(ax.add_artist(ab))

    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()


# Show dataset images with T-sne projection of latent space encoding
def computeTSNEProjectionOfLatentSpace(X, im_size, model, display=True):

    np.random.shuffle(X)

    # Compute latent space representation
    print("Computing latent space projection...")
    model_out = list(model.predict(input_fn=lambda: input_fn_pred(X)))
    X_encoded = np.array([p["encoded"] for p in model_out])

    # Compute t-SNE embedding of latent space
    print("Computing t-SNE embedding...")
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    X_tsne = tsne.fit_transform(X_encoded)

    # Plot images according to t-sne embedding
    if display:
        print("Plotting t-SNE visualization...")
        fig, ax = plt.subplots()
        imscatter(X_tsne[:, 0], X_tsne[:, 1], imageData=X, im_size=im_size, ax=ax, zoom=0.6)
        plt.show()
    else:
        return X_tsne


# Show dataset images with T-sne projection of pixel space
def computeTSNEProjectionOfPixelSpace(X, im_size, display=True):

    np.random.shuffle(X)

    # Compute t-SNE embedding of pixel space
    print("Computing t-SNE embedding...")
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    X_tsne = tsne.fit_transform(X.reshape([-1, im_size[0] * im_size[1] * 1]))

    # Plot images according to t-sne embedding
    if display:
        print("Plotting t-SNE visualization...")
        fig, ax = plt.subplots()
        imscatter(X_tsne[:, 0], X_tsne[:, 1], imageData=X, im_size=im_size, ax=ax, zoom=0.6)
        plt.show()
    else:
        return X_tsne


def twoDimensionalTsneGrid(X, im_size, model, out_zoom, out_dim, pixel_or_latent='latent', display=True):

    np.random.shuffle(X)

    # Compute 2d t-SNE embedding of pixel or latent space and show then in a 2d grid
    if pixel_or_latent is 'pixel':
        X_2d = computeTSNEProjectionOfLatentSpace(X, im_size, model, display=False)
    elif pixel_or_latent is 'latent':
        X_2d = computeTSNEProjectionOfLatentSpace(X, im_size, model, display=False)

    grid = np.dstack(np.meshgrid(np.linspace(0, 1, out_dim), np.linspace(0, 1, out_dim))).reshape(-1, 2)
    cost_matrix = cdist(grid, X_2d, "sqeuclidean").astype(np.float32)
    cost_matrix = cost_matrix * (100000 / cost_matrix.max())
    row_asses, col_asses, _ = lapjv(cost_matrix)
    grid_jv = grid[col_asses]
    out = np.ones((int(out_dim * out_zoom * im_size[0]), int(out_dim * out_zoom * im_size[1]), 1))

    for pos, img in zip(grid_jv, X):
        h_range = int(np.floor(pos[0] * (out_dim - 1) * out_zoom * im_size[0]))
        w_range = int(np.floor(pos[1] * (out_dim - 1) * out_zoom * im_size[1]))
        out[h_range:h_range + out_zoom * im_size[0], w_range:w_range + out_zoom * im_size[1]] = image.img_to_array(img)

    im = image.array_to_img(out)
    # Plot images according to t-sne embedding
    if display:
        print("Plotting t-SNE 2d grid visualization...")
        plt.figure()
        plt.axis('off')
        plt.title('t-SNE 2d grid')
        plt.imshow(im)
        plt.show()
    else:
        # im.save(out_dir + out_name, quality=100)
        return


# Shows linear inteprolation in image space vs latent space
def visualizeInterpolation(start, end, model, im_size, save=False, nbSteps=5):
    print("Generating interpolations...")

    # Create micro batch
    X = np.array([start, end])

    # Compute latent space projection
    model_out = list(model.predict(input_fn=lambda: input_fn_pred(X)))
    X_encoded = np.array([p["encoded"] for p in model_out])
    latentStart, latentEnd = X_encoded[0], X_encoded[1]

    # Get original image for comparison
    startImage, endImage = X[0], X[1]

    vectors = []
    normalImages = []
    # Linear interpolation
    alphaValues = np.linspace(0, 1, nbSteps)
    for alpha in alphaValues:
        # Latent space interpolation
        vector = latentStart * (1 - alpha) + latentEnd * alpha
        vectors.append(vector)
        # Image space interpolation
        blendImage = cv2.addWeighted(startImage, 1 - alpha, endImage, alpha, 0)
        normalImages.append(blendImage)

    # Decode latent space vectors
    vectors = np.array(vectors)
    sham_inputs = np.zeros(
        shape=[vectors.shape[0], im_size[0], im_size[1], 1])  # the model still expects an input image. it is useless
    model_out = list(model.predict(input_fn=lambda: input_fn_pred(sham_inputs, vectors)))
    reconstructions = np.array([p["reconstructions"] for p in model_out])

    # Put final image together
    resultLatent = None
    resultImage = None

    if save is not False:
        hashName = ''.join(random.choice(string.lowercase) for i in range(3))

    zoom = 4
    for i in range(len(reconstructions)):
        interpolatedImage = normalImages[i]
        interpolatedImage = cv2.resize(interpolatedImage,
                                       (zoom * im_size[1], zoom * im_size[0]))  # cv2 weirdness - axis inverted
        separating_line = np.ones((interpolatedImage.shape[0], 25))
        resultImage = interpolatedImage if resultImage is None else np.hstack(
            [resultImage, separating_line, interpolatedImage, separating_line])

        reconstructedImage = reconstructions[i]
        reconstructedImage = cv2.resize(reconstructedImage, (zoom * im_size[1], zoom * im_size[0]))
        resultLatent = reconstructedImage if resultLatent is None else np.hstack(
            [resultLatent, separating_line, reconstructedImage, separating_line])

        if save is not False:
            cv2.imwrite(save + "{}_{}.png".format(hashName, i), np.hstack([interpolatedImage, reconstructedImage]))

        result = np.vstack([resultImage, resultLatent])

    if save is False:
        plt.figure()
        plt.imshow(np.squeeze(result), cmap='binary_r')
        plt.title('Interpolation in Pixel Space (top) vs Latent Space (bottom)')
        plt.axis('off')
        plt.show()
        plt.close()


# Computes A, B, C, A+B, A+B-C in latent space
def visualizeArithmetics(a, b, c, model, im_size):
    print("Computing arithmetics...")
    # Create micro batch
    X = np.array([a, b, c])

    # Compute latent space projection
    model_out = list(model.predict(input_fn=lambda: input_fn_pred(X)))
    latentA, latentB, latentC = np.array([p["encoded"] for p in model_out])

    add = latentA + latentB
    addSub = latentA + latentB - latentC

    # Create micro batch
    latent_X = np.array([latentA, latentB, latentC, add, addSub])

    # Compute reconstruction
    sham_inputs = np.zeros(
        shape=[latent_X.shape[0], im_size[0], im_size[1], 1])  # the model still expects an input image. it is useless
    model_out = list(model.predict(input_fn=lambda: input_fn_pred(sham_inputs, latent_X)))
    reconstructedA, reconstructedB, reconstructedC, reconstructedAdd, reconstructedAddSub = np.array(
        [p["reconstructions"] for p in model_out])

    zoom = 4
    reconstructedA = cv2.resize(reconstructedA, (zoom * im_size[1], zoom * im_size[0]))
    reconstructedB = cv2.resize(reconstructedB, (zoom * im_size[1], zoom * im_size[0]))
    reconstructedC = cv2.resize(reconstructedC, (zoom * im_size[1], zoom * im_size[0]))
    reconstructedAdd = cv2.resize(reconstructedAdd, (zoom * im_size[1], zoom * im_size[0]))
    reconstructedAddSub = cv2.resize(reconstructedAddSub, (zoom * im_size[1], zoom * im_size[0]))
    separating_line = np.ones((reconstructedA.shape[0], 25))

    plt.figure()
    plt.imshow(np.squeeze(np.hstack(
        [reconstructedA, separating_line, reconstructedB, separating_line, reconstructedC, separating_line,
         reconstructedAdd, separating_line, reconstructedAddSub])), cmap='binary_r')
    plt.title('Arithmetics in latent space: A, B, C, A+B, A+B-C')
    plt.axis('off')
    plt.show()
    plt.close()


def tensorboard_embeddings(X, im_size, latent_dim, model, LOGDIR, do_pixel_embedding=False):

    np.random.shuffle(X)

    print("Making tensorboard embeddings...")
    tf.summary.FileWriter(LOGDIR)

    np.random.shuffle(X)

    model_out = list(model.predict(input_fn=lambda: input_fn_pred(X)))
    X_encoded = np.array([p["encoded"] for p in model_out])

    # create sprites (if they don't exist yet)
    if not os.path.exists(LOGDIR + '/sprites.png'):
        sprites = images_to_sprite(np.squeeze(X))
        plt.imsave(LOGDIR + '/sprites.png', sprites, cmap='gray')
    SPRITES = './sprites.png'  # CAREFUL, path relative to where you launch tensorboard, not rootdir

    X_in = tf.placeholder(tf.float32)
    X_encoded_in = tf.placeholder(tf.float32)

    with tf.device('/cpu:0'):

        # configure embedding visualizer
        config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()

        # embeddings for the latent layer activations
        embedding_input_latent = tf.cast(tf.reshape(X_encoded_in, [-1, latent_dim]), tf.float32)
        embedding_size_latent = latent_dim
        embedding_latent = tf.Variable(tf.zeros([X.shape[0], embedding_size_latent], dtype=tf.float32),
                                       name='latent_embedding')
        assignment_latent = embedding_latent.assign(embedding_input_latent)
        embedding_config_latent = config.embeddings.add()
        embedding_config_latent.tensor_name = embedding_latent.name
        embedding_config_latent.sprite.image_path = SPRITES
        embedding_config_latent.sprite.single_image_dim.extend([max(im_size), max(im_size)])

        # embeddings for the input images
        if do_pixel_embedding:
            embedding_input_images = tf.cast(tf.reshape(X_in, [-1, im_size[0] * im_size[1]]), tf.float32)
            embedding_size_images = im_size[0] * im_size[1]
            embedding_images = tf.Variable(tf.zeros([X.shape[0], embedding_size_images], dtype=tf.float32),
                                           name='input_images_embedding')
            assignment_images = embedding_images.assign(embedding_input_images)
            embedding_config_images = config.embeddings.add()
            embedding_config_images.tensor_name = embedding_images.name
            embedding_config_images.sprite.image_path = SPRITES
            embedding_config_images.sprite.single_image_dim.extend([max(im_size), max(im_size)])

        with tf.Session() as sess:
            saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            init.run()
            writer = tf.summary.FileWriter(LOGDIR, sess.graph)
            tf.contrib.tensorboard.plugins.projector.visualize_embeddings(writer, config)

            if do_pixel_embedding:
                sess.run([assignment_images, assignment_latent], feed_dict={X_in: X, X_encoded_in: X_encoded})
            else:
                sess.run(assignment_latent,
                         feed_dict={X_in: X, X_encoded_in: X_encoded})  # only latent layer embeddings (faster)

            saver.save(sess, LOGDIR + '/checkpoint.ckpt')


def show_n_best_and_worst_configs(X, im_size, n, model, gif_frame=False):

    np.random.shuffle(X)

    print('Preparing best and worst configurations...')
    n = int((np.floor(np.sqrt(n))))**2  # get closest lower perfect square
    sqrt_n = int(np.sqrt(n))

    # get losses
    model_out = list(model.predict(input_fn=lambda: input_fn_pred(X)))
    all_losses = np.array([p["all_losses"] for p in model_out])
    all_losses_order = all_losses.argsort()

    spacing = 10
    best_result_image  = np.ones(((im_size[0] + spacing) * sqrt_n, (im_size[1] + spacing) * sqrt_n))
    worst_result_image = np.ones(((im_size[0] + spacing) * sqrt_n, (im_size[1] + spacing) * sqrt_n))

    for i in range(sqrt_n):
        for j in range(sqrt_n):
            this_filter = i * sqrt_n + j
            this_best_img = np.squeeze(X[all_losses_order[this_filter]])
            this_worst_img = np.squeeze(X[all_losses_order[-(this_filter + 1)]])
            best_result_image [i * (im_size[0] + spacing) + spacing:(i + 1) * (im_size[0] + spacing), j * (im_size[1] + spacing) + spacing:(j + 1) * (im_size[1] + spacing)] = this_best_img
            worst_result_image[i * (im_size[0] + spacing) + spacing:(i + 1) * (im_size[0] + spacing), j * (im_size[1] + spacing) + spacing:(j + 1) * (im_size[1] + spacing)] = this_worst_img

    if gif_frame is not False:
        fig, (ax1, ax2) = plt.subplots(2,1)
        ax1.imshow(best_result_image, cmap='binary_r')
        ax1.set_title('Best ' + str(n) + ' configs (read from left to right)')
        ax2.plot(all_losses[all_losses_order])
        ax2.set_title('Loss curve for all stimuli')
        plt.axis('off')
        plt.title('For latent dimensions = ' + str(gif_frame))
        fig.canvas.draw()  # draw the canvas, cache the renderer
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return image

    else:
        grid = plt.GridSpec(2, 2, wspace=0.4, hspace=0.4)
        plt.subplot(grid[0,0])
        plt.imshow(best_result_image, cmap='binary_r')
        plt.title('Best ' + str(n) + ' configs (read from left to right)')
        plt.axis('off')

        plt.subplot(grid[0,1])
        plt.imshow(worst_result_image, cmap='binary_r')
        plt.title('Worst ' + str(n) + ' configs (read from left to right)')
        plt.axis('off')

        plt.subplot(grid[1,:])
        plt.plot(all_losses[all_losses_order])
        plt.title('Loss curve for all stimuli')
        plt.show()
        plt.close()


def show_n_best_and_worst_configs_control(X, im_size, n):
    # computes losses between dataset and randomly shuffled dataset and plot the same things as show_n_best_and_worst_configs for comparison
    X_random = X.copy()
    np.random.shuffle(X_random)

    print('Preparing best and worst configurations control...')
    n = int((np.floor(np.sqrt(n))))**2  # get closest lower perfect square
    sqrt_n = int(np.sqrt(n))

    # get losses
    all_losses = np.sum((np.reshape(X, [-1, np.prod(im_size)]) - np.reshape(X_random, [-1, np.prod(im_size)]))**2, axis=1)
    all_losses_order = all_losses.argsort()

    spacing = 10
    best_result_image  = np.ones(((im_size[0] + spacing) * sqrt_n, (im_size[1] + spacing) * sqrt_n))
    worst_result_image = np.ones(((im_size[0] + spacing) * sqrt_n, (im_size[1] + spacing) * sqrt_n))

    for i in range(sqrt_n):
        for j in range(sqrt_n):
            this_filter = i * sqrt_n + j
            this_best_img = np.squeeze(X[all_losses_order[this_filter]])
            this_worst_img = np.squeeze(X[all_losses_order[-(this_filter + 1)]])
            best_result_image [i * (im_size[0] + spacing) + spacing:(i + 1) * (im_size[0] + spacing), j * (im_size[1] + spacing) + spacing:(j + 1) * (im_size[1] + spacing)] = this_best_img
            worst_result_image[i * (im_size[0] + spacing) + spacing:(i + 1) * (im_size[0] + spacing), j * (im_size[1] + spacing) + spacing:(j + 1) * (im_size[1] + spacing)] = this_worst_img

    grid = plt.GridSpec(2, 2, wspace=0.4, hspace=0.4)
    plt.subplot(grid[0,0])
    plt.imshow(best_result_image, cmap='binary_r')
    plt.title('Best ' + str(n) + ' configs (read from left to right)')
    plt.axis('off')

    plt.subplot(grid[0,1])
    plt.imshow(worst_result_image, cmap='binary_r')
    plt.title('Worst ' + str(n) + ' configs (read from left to right)')
    plt.axis('off')

    plt.subplot(grid[1,:])
    plt.plot(all_losses[all_losses_order])
    plt.title('Loss curve for all stimuli')
    plt.show()
    plt.close()


def make_losses_and_scores_barplot(X, model, gif_frame=False):
    # show losses and scores for the entire dataset. if gif_frames=True, doesn't plt.show() and instead returns a frame for make_gif_from_frames

    print('Making losses & scores bar plots...')

    n_configs = X.shape[0]

    # get losses
    model_out = list(model.predict(input_fn=lambda: input_fn_pred(X)))
    all_losses = np.array([p["all_losses"] for p in model_out])
    all_losses_order = all_losses.argsort()
    scores = n_configs - all_losses_order  # originally, the best configs have low values. Switch this for better visualisation.

    ind = np.arange(n_configs)
    if gif_frame is not False:
        fig, (ax1, ax2) = plt.subplots(1,2)
        ax1.bar(ind, all_losses, color=(3. / 255, 57. / 255, 108. / 255))
        ax1.set_xlabel('configuration IDs')
        ax1.set_ylabel('Losses')
        ax1.set_ylim(0, n_configs)
        ax2.bar(ind, scores, color=(3. / 255, 57. / 255, 108. / 255))
        ax2.set_xlabel('configuration IDs')
        ax2.set_ylabel('Scores')
        ax2.set_ylim(0, n_configs)
        plt.title('Losses and scores for latent dimensions = ' + str(gif_frame))
        # Used to return the plot as an image array
        fig.canvas.draw()  # draw the canvas, cache the renderer
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return image
    else:
        plt.subplot(1, 2, 1)
        plt.bar(ind, all_losses, color=(3. / 255, 57. / 255, 108. / 255))
        plt.xlabel('configuration IDs')
        plt.ylabel('Losses')
        plt.subplot(1, 2, 2)
        plt.bar(ind, scores, color=(3. / 255, 57. / 255, 108. / 255))
        plt.xlabel('configuration IDs')
        plt.ylabel('Scores')
        plt.ylim(0, n_configs)
        plt.show()
        plt.close()


def make_gif_from_frames(X, im_size, model, model_type, latent_dims, type, save_path='./'):
    # makes a gif of losses_and_scores or of configs_and_loss_curves for models with latent dim in latent_dims
    imgs_for_gif = []
    if type is 'losses_and_scores':
        for i in range(len(latent_dims)):
            print("\r{}/{} ({:.1f}%) ".format(i, len(latent_dims), i * 100 / len(latent_dims)), end="")
            imgs_for_gif.append(make_losses_and_scores_barplot(X, model, gif_frame=latent_dims[i]))
        print('saving to ' + save_path + model_type + '_losses_and_scores_evolving.gif')
        imageio.mimsave(save_path + model_type + '_losses_and_scores_evolving.gif', imgs_for_gif, fps=1)

    if type is 'configs_and_loss_curves':
        for i in range(len(latent_dims)):
            print("\r{}/{} ({:.1f}%) ".format(i, len(latent_dims), i * 100 / len(latent_dims)), end="")
            imgs_for_gif.append(show_n_best_and_worst_configs(X, im_size, 64, model, gif_frame=latent_dims[i]))
        print('saving to ' + save_path + model_type + '_best_worst_configs_evolving.gif')
        imageio.mimsave(save_path + model_type + '_best_worst_configs_evolving.gif', imgs_for_gif, fps=1)
