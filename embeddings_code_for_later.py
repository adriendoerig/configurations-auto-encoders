from parameters import *
from create_sprite import *
model_folder = 'conv'

LOGDIR = './' + model_type + '/' + model_type + '_' + str(n_hidden_units) + '_hidden_units_logdir'

with tf.device('/cpu:0'):

    embedding_writer = tf.summary.FileWriter(LOGDIR)  # to write summaries
    np.random.shuffle(dataset)
    embedding_data = dataset[:5000, :, :, :]  # we will pass the entire dataset to the embeddings

    # create sprites (if they don't exist yet)
    if not os.path.exists(LOGDIR+'/sprites.png'):
        sprites = invert_grayscale(images_to_sprite(np.squeeze(embedding_data)))
        plt.imsave(LOGDIR+'/sprites.png', sprites, cmap='gray')
    SPRITES = LOGDIR+'/sprites.png'

    # embeddings for the input images
    embedding_input_images = tf.reshape(X_flat, [-1, im_size[0]*im_size[1]])
    embedding_size_images = im_size[0]*im_size[1]
    embedding_images = tf.Variable(tf.zeros([embedding_data.shape[0], embedding_size_images]), name='input_images_embedding')
    assignment_images = embedding_images.assign(embedding_input_images)
    # embeddings for the hidden layer activations
    embedding_input_hidden = tf.reshape(hidden, [-1, n_hidden_units])
    embedding_size_hidden = n_hidden_units
    embedding_hidden = tf.Variable(tf.zeros([embedding_data.shape[0], embedding_size_hidden]), name='hidden_layer_embedding')
    assignment_hidden = embedding_hidden.assign(embedding_input_hidden)
    # embeddings for the reconstructed images
    embedding_input_reconstructions = tf.reshape(X_reconstructed, [-1, im_size[0]*im_size[1]])
    embedding_size_reconstructions = im_size[0] * im_size[1]
    embedding_reconstructions = tf.Variable(tf.zeros([embedding_data.shape[0], embedding_size_reconstructions]), name='reconstructed_images_embedding')
    assignment_reconstructions = embedding_reconstructions.assign(embedding_input_reconstructions)

    # configure embedding visualizer
    # input images embedding
    config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
    # embedding_config_images = config.embeddings.add()
    # embedding_config_images.tensor_name = embedding_images.name
    # embedding_config_images.sprite.image_path = SPRITES
    # embedding_config_images.sprite.single_image_dim.extend([max(im_size), max(im_size)])
    # hidden layer embedding
    embedding_config_hidden = config.embeddings.add()
    embedding_config_hidden.tensor_name = embedding_hidden.name
    embedding_config_hidden.sprite.image_path = SPRITES
    embedding_config_hidden.sprite.single_image_dim.extend([max(im_size), max(im_size)])
    reconstructed images embedding
    embedding_config_reconstructions = config.embeddings.add()
    embedding_config_reconstructions.tensor_name = embedding_reconstructions.name
    embedding_config_reconstructions.sprite.image_path = SPRITES
    embedding_config_reconstructions.sprite.single_image_dim.extend([max(im_size), max(im_size)])

