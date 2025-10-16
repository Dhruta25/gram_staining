# test_model.py
# simple testing script for bacteria stain model
import os
import tensorflow as tf
import numpy as np
import argparse
import ntpath
from tensorflow.python.client import device_lib
import get_data as loader
import network

def str2bool(v):
    return v.lower() != 'false'

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu_id', type=int, default=0)
    p.add_argument('--input', type=str, default='df_zstack2')
    p.add_argument('--data_name', type=str, default='exp_1')
    p.add_argument('--test_path', type=str, default='./test_data/input/*.npy')
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--image_size', type=int, default=1024)
    p.add_argument('--n_channels', type=int, default=16)
    p.add_argument('--n_levels', type=int, default=5)
    p.add_argument('--testbool', type=str2bool, default=True)
    args = p.parse_args()
    args.record_file = args.test_path
    return args

def get_gpu():
    devices = device_lib.list_local_devices()
    return [d.name for d in devices if d.device_type == 'GPU'][0]

if __name__ == '__main__':
    cfg = get_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)

    save_dir = f'{cfg.data_name}/test_images/'
    os.makedirs(save_dir, exist_ok=True)

    with tf.Graph().as_default():
        valid_it = loader.get_dataset_iterator_bacteria_npy(cfg.record_file, cfg, cfg, 'test')
        x, y, path = valid_it.get_next()

        with tf.device(get_gpu()):
            with tf.compat.v1.variable_scope('Generator'):
                G = network.Generator(x, cfg)

        output = G.output

        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            saver = tf.compat.v1.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint('ckpts/'))

            sess.run(valid_it.initializer)
            count = 0

            while True:
                try:
                    sr, gt, inp, paths = sess.run([output, y, x, path])
                    name = os.path.splitext(ntpath.basename(paths[0].decode('utf-8')))[0]
                    sr, gt, inp = np.squeeze(sr), np.squeeze(gt), np.squeeze(inp)

                    loader.save_pure_image(np.clip(sr, 0, 1), f'{name}_out', save_dir)
                    loader.save_pure_image(np.clip(gt, 0, 1), f'{name}_tar', save_dir)
                    count += 1
                except tf.errors.OutOfRangeError:
                    break