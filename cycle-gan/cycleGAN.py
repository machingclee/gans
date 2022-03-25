
from random import random
from sklearn.utils import resample
from numpy import zeros, ones, asarray
from numpy import load
from numpy.random import randint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, LeakyReLU, Activation, Concatenate
from instancenormalization import InstanceNormalization
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import Model


os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
tf.config.experimental.list_physical_devices()
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


def get_image_paths(path):
    data_list = []
    for filename in os.listdir(path):
        image_path = os.path.join(path, filename)
        data_list.append(image_path)
    return data_list


n_sample = 500


def generate_real_samples(path, n_samples=n_sample, patch_shape=16):
    image_paths = np.array(get_image_paths(path))
    indexes = randint(0, len(image_paths), n_samples)
    X = np.array([load_img_from_path(path) for path in image_paths[indexes]])
    y = ones((n_samples, patch_shape, patch_shape, 1))
    return X, y


def generate_fake_samples(g_model, dataset, patch_shape=16):
    X = g_model.predict(dataset)
    y = zeros((len(X), patch_shape, patch_shape, 1))
    return X, y


def load_img_from_path(image_path):
    pixels = load_img(image_path, target_size=(256, 256))
    pixels = img_to_array(pixels)
    pixels = pixels/127.5 - 1
    return pixels


def gen_dataset(n_sample=500):
    train_A_paths = get_image_paths("monet2photo/trainA")
    train_B_paths = get_image_paths("monet2photo/trainB")

    random_state = np.random.randint(0, 100)
    print("random_state", random_state)
    shuffle_indexes = resample(range(len(train_A_paths)), replace=False, n_samples=500, random_state=random_state)
    shuffled_trainA_img = (load_img_from_path(train_A_paths[index]) for index in shuffle_indexes)
    shuffled_trainB_img = (load_img_from_path(train_B_paths[index]) for index in shuffle_indexes)
    dataset = zip(shuffled_trainA_img, shuffled_trainB_img)
    return dataset


def check_dataset():
    dataset_ = gen_dataset()
    for i in range(3):
        img_A, img_B = next(dataset_)
        img_A = (img_A + 1) * 127.5
        img_B = (img_B + 1) * 127.5
        plt.subplot(2, 3, 1+i)
        plt.axis("off")
        plt.imshow(img_A.astype("uint8"))

        plt.subplot(2, 3, 4+i)
        plt.axis("off")
        plt.imshow(img_B.astype("uint8"))


def resnet_block(n_filters, input_layer):
    init = RandomNormal(stddev=0.02)
    # as strides = 1, the shape is invariant
    g = Conv2D(n_filters, (3, 3), padding="same", kernel_initializer=init)(input_layer)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation("relu")(g)
    g = Conv2D(n_filters, (3, 3), padding="same", kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Concatenate()([g, input_layer])
    return g


def define_generator(image_shape, n_resnet=9):
    init = RandomNormal(stddev=0.02)
    in_image = Input(shape=image_shape)
    g = Conv2D(64, (7, 7), padding="same", kernel_initializer=init)(in_image)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation("relu")(g)

    g = Conv2D(128, (3, 3), strides=(2, 2), padding="same", kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation("relu")(g)

    g = Conv2D(256, (3, 3), strides=(2, 2), padding="same", kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation("relu")(g)

    for _ in range(n_resnet):
        g = resnet_block(256, g)

    g = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding="same", kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation("relu")(g)

    g = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same", kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)

    g = Conv2D(3, (7, 7), padding="same", kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    out_image = Activation("tanh")(g)
    model = Model(in_image, out_image)

    return model


def define_discriminator(image_shape):
    init = RandomNormal(stddev=0.02)
    in_image = Input(shape=image_shape)
    d = Conv2D(64, (4, 4), strides=2, padding="same", kernel_initializer=init)(in_image)
    d = LeakyReLU(alpha=0.2)(d)

    d = Conv2D(128, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)

    d = Conv2D(256, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)

    d = Conv2D(512, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)

    d = Conv2D(512, (4, 4), padding="same", kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)

    patch_out = Conv2D(1, (4, 4), padding="same", kernel_initializer=init)(d)
    model = Model(in_image, patch_out)
    model.compile(loss="mse", optimizer=Adam(learning_rate=0.0002, beta_1=0.5), loss_weights=[0.5])
    return model


def define_composite_model(g_model_AB, d_model_B, g_model_BA, image_shape):
    g_model_AB.trainable = True
    d_model_B.trainable = False
    g_model_BA.trainable = False
    # adversarial loss
    input_gen = Input(shape=image_shape)
    g_AB = g_model_AB(input_gen)
    critic = d_model_B(g_AB)

    # identity loss
    # we just calculate the identity from B to B
    # we will define quite the same model from opposite direction, which will take care of the reverse identity loss
    input_B_id = Input(shape=image_shape)
    output_B_id = g_model_AB(input_B_id)

    # cycle loss - forward
    g_ABA_cycle = g_model_BA(g_AB)

    # cycle loss - backward
    gen_BA_out = g_model_BA(input_B_id)
    g_BAB_cycle = g_model_AB(gen_BA_out)

    # define model graph
    model = Model([input_gen, input_B_id], [critic, output_B_id, g_ABA_cycle, g_BAB_cycle])
    model.compile(loss=["mse", "mae", "mae", "mae"],
                  loss_weights=[1, 5, 10, 10],
                  optimizer=Adam(learning_rate=0.0002, beta_1=0.5)
                  )
    return model


dataset = gen_dataset()
img_A, img_B = next(dataset)
image_shape = img_A.shape
print("image_shape", image_shape)


def update_image_pool(pool, images, max_size=50):
    selected = []

    for image in images:
        if len(pool) < max_size:
            # stock the pool
            pool.append(image)
            selected.append(image)
        elif random() < 0.5:
            # use image, but don't add it to the pool
            selected.append(image)
        else:
            # replace an existing image and use replaced image
            index = randint(0, len(pool))
            selected.append(pool[index])
            pool[index] = image
    return asarray(selected)


def update_fake_image_pool(pool, images, max_size=50):
    selected = []

    for image in images:
        if len(pool) < max_size:
            pool.append(image)
            selected.append(image)
        elif random() < 0.5:
            selected.append(image)
        else:
            # take one from the pool and and update the pool
            # the pool may contain new image that we never
            index = randint(0, len(pool))
            selected.append(pool[index])
            pool[index] = image

    return asarray(selected)


g_model_AtoB = define_generator(image_shape)
g_model_BtoA = define_generator(image_shape)
d_model_A = define_discriminator(image_shape)
d_model_B = define_discriminator(image_shape)
c_model_AtoB = define_composite_model(g_model_AtoB, d_model_B, g_model_BtoA, image_shape)
c_model_BtoA = define_composite_model(g_model_BtoA, d_model_A, g_model_AtoB, image_shape)
n_epochs, n_batch, = 100, 1
n_patch = d_model_A.output_shape[1]

poolA = []
poolB = []
batch_per_epoch = int(n_sample/n_epochs)
n_steps = batch_per_epoch * n_epochs


def show_result(step):
    dataset_ = gen_dataset()
    for i in range(3):
        img_A, _ = next(dataset_)
        fake_B = g_model_AtoB.predict(np.array([img_A]))
        cycle_A = g_model_BtoA.predict(fake_B)
        img_A = (img_A + 1) * 127.5
        fake_B = (fake_B[0] + 1) * 127.5
        cycle_A = (cycle_A[0] + 1) * 127.5

        plt.subplot(3, 3, 1+i)
        plt.axis("off")
        plt.imshow(img_A.astype("uint8"))

        plt.subplot(3, 3, 4+i)
        plt.axis("off")
        plt.imshow(img_B.astype("uint8"))

        plt.subplot(3, 3, 7+i)
        plt.axis("off")
        plt.imshow(cycle_A.astype("uint8"))
        plt.savefig(f'result_{str(step).zfill(3)}.png')


for i in range(n_steps):
    print(f"step {i}")
    show_result(i)

    X_realA, y_realA = generate_real_samples("./monet2photo/trainA", n_batch, n_patch)
    X_realB, y_realB = generate_real_samples("./monet2photo/trainB", n_batch, n_patch)
    X_fakeA, y_fakeA = generate_fake_samples(g_model_BtoA, X_realB, n_patch)
    X_fakeB, y_fakeB = generate_fake_samples(g_model_AtoB, X_realA, n_patch)
    X_fakeA = update_image_pool(poolA, X_fakeA)
    X_fakeB = update_image_pool(poolB, X_fakeB)
    print("training c_model_BtoA")
    g_loss2, _, _, _, _ = c_model_BtoA.train_on_batch([X_realB, X_realA], [y_realA, X_realA, X_realB, X_realA])
    dA_loss1 = d_model_A.train_on_batch(X_realA, y_realA)
    dA_loss2 = d_model_A.train_on_batch(X_fakeA, y_fakeA)
    print("training c_model_AtoB")
    g_loss1, _, _, _, _ = c_model_AtoB.train_on_batch([X_realA, X_realB], [y_realB, X_realB, X_realA, X_realB])
    dB_loss1 = d_model_B.train_on_batch(X_realB, y_realB)
    dB_loss2 = d_model_B.train_on_batch(X_fakeB, y_fakeB)
    print('>%d, dA[%.3f,%.3f] dB[%.3f,%.3f] g[%.3f,%.3f]' % (i+1, dA_loss1, dA_loss2, dB_loss1, dB_loss2, g_loss1, g_loss2), end="\r")
