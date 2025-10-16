import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
import os

# Simple helper for random crop

def random_crop(lr_img, hr_img, crop_size=256, scale=1):
    lr_size = crop_size // scale

    h, w = tf.shape(lr_img)[0], tf.shape(lr_img)[1]

    lr_x = tf.random.uniform((), 0, w - lr_size + 1, dtype=tf.int32)
    lr_y = tf.random.uniform((), 0, h - lr_size + 1, dtype=tf.int32)

    hr_x = lr_x * scale
    hr_y = lr_y * scale

    lr_crop = lr_img[lr_y:lr_y + lr_size, lr_x:lr_x + lr_size, :]
    hr_crop = hr_img[hr_y:hr_y + crop_size, hr_x:hr_x + crop_size, :]

    return lr_crop, hr_crop


# Fixed center crop

def center_crop(lr_img, hr_img, crop_size=256, scale=1):
    lr_size = crop_size // scale

    h, w = tf.shape(lr_img)[0], tf.shape(lr_img)[1]

    lr_x = (w - lr_size) // 2
    lr_y = (h - lr_size) // 2

    hr_x = lr_x * scale
    hr_y = lr_y * scale

    lr_crop = lr_img[lr_y:lr_y + lr_size, lr_x:lr_x + lr_size, :]
    hr_crop = hr_img[hr_y:hr_y + crop_size, hr_x:hr_x + crop_size, :]

    return lr_crop, hr_crop


# -----------------------------
# Random Flip
# -----------------------------
def random_flip(lr_img, hr_img):
    flip = tf.random.uniform(()) < 0.5
    lr_img = tf.cond(flip, lambda: tf.image.flip_left_right(lr_img), lambda: lr_img)
    hr_img = tf.cond(flip, lambda: tf.image.flip_left_right(hr_img), lambda: hr_img)
    return lr_img, hr_img


# -----------------------------
# Random Rotation
# -----------------------------
def random_rotate(lr_img, hr_img):
    k = tf.random.uniform((), 0, 4, dtype=tf.int32)
    lr_img = tf.image.rot90(lr_img, k)
    hr_img = tf.image.rot90(hr_img, k)
    return lr_img, hr_img


# -----------------------------
# Main preprocessing
# -----------------------------
def preprocess_npy(inp, label, path, cfg):
    h = tf.shape(inp)[0]
    inp = tf.reshape(inp, [h, h, 5])
    label = tf.reshape(label, [h, h, 3])

    if cfg.input == 'only_df':
        inp = inp[:, :, :1]
    elif cfg.input == 'df_zstack1':
        inp = inp[:, :, :3]
    else:
        inp = inp[:, :, :5]

    if cfg.inverted_input:
        inp = 1 - inp

    if not cfg.testbool:
        inp, label = random_crop(inp, label, cfg.image_size, 1)
        inp, label = random_flip(inp, label)
        inp, label = random_rotate(inp, label)
    else:
        inp, label = center_crop(inp, label, cfg.image_size, 1)

    return inp, label, path


# -----------------------------
# Map function for loading npy
# -----------------------------
def load_npy(path):
    inp = np.load(path)
    tar = np.load(path.decode('utf-8').replace('input', 'label'))

    avg_max = 0.1760
    inp = inp[:, :, :5] / avg_max

    return inp, tar, path


# -----------------------------
# Create tf.data iterator
# -----------------------------
def get_dataset_iterator(filename, train_cfg, valid_cfg, dtype):
    image_paths = glob.glob(filename)

    if not train_cfg.testbool:
        np.random.shuffle(image_paths)

    ds = tf.data.Dataset.from_tensor_slices(image_paths)

    shuffle_sz = 10000 if dtype == 'train' else 1
    ds = ds.shuffle(shuffle_sz, reshuffle_each_iteration=True)

    ds = ds.map(lambda p: tf.numpy_function(load_npy, [p], [tf.float32, tf.float32, tf.string]),
                num_parallel_calls=train_cfg.n_threads)

    ds = ds.map(lambda x, y, z: preprocess_npy(x, y, z, train_cfg),
                num_parallel_calls=train_cfg.n_threads)

    if not train_cfg.testbool:
        batch_size = train_cfg.batch_size if dtype == 'train' else valid_cfg.batch_size
    else:
        batch_size = 1

    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    iterator = tf.compat.v1.data.make_initializable_iterator(ds)
    return iterator


# -----------------------------
# Save image helpers
# -----------------------------
def save_image(img, title='Image', vmin=0, vmax=1, img_id='img0', save_path='./'):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plt.figure()
    plt.title(title)
    plt.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f'{save_path}/{img_id}.png')
    plt.close()


def save_jpg(img, img_id='img0', save_path='./'):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    img = (img * 255).astype(np.uint8)
    Image.fromarray(img).save(f'{save_path}/{img_id}.jpg')


def save_as_mat(img, img_id='img0', save_path='./'):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    tf.io.write_file(f'{save_path}/{img_id}.mat', tf.io.encode_base64(tf.io.serialize_tensor(img)))


def save_output_label_mat(output, label, img_id='img0', save_path='./'):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    data = {'out': output.numpy(), 'tar': label.numpy()}
    np.savez(f'{save_path}/{img_id}.npz', **data)