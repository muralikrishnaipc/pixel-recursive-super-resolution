import numpy as np
import tensorflow as tf
import os

from skimage.io import imsave
from skimage.io import imread

from prsr.utils import logits_2_pixel_value
from prsr.net import Net

from prsr.flags import Flags

args = Flags()


class SuperResModel(object):

    def __init__(self, net, low_res_image, high_res_image):
        self.low_res_image = low_res_image
        self.high_res_image = high_res_image
        self.net = net


def initialize_dir_if_not_exists(dir_name):
    """
    Create the directory if it is not exists already.
    :param dir_name:
    :return:
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)


def save_image(image, image_path):

    image = np.uint8(image)
    imsave(image_path, image)


def create_network(low_res_shape, high_res_shape):

    low_res_image = tf.placeholder(dtype=tf.float32, shape=low_res_shape, name='low_res_image')
    high_res_image = tf.placeholder(dtype=tf.float32, shape=high_res_shape, name='high_res_image')
    net = Net(hr_images=high_res_image, lr_images=low_res_image, scope='prsr')

    return SuperResModel(net=net, low_res_image=low_res_image, high_res_image=high_res_image)


def enhance(model, c_logits, p_logits, session, low_res_input, high_res_shape, mu=1.1):

    high_res_out_image = np.zeros(shape=high_res_shape, dtype=np.float32)

    np_c_logits = session.run(
        c_logits,
        feed_dict={model.low_res_image: low_res_input, model.net.train: False}
    )

    batch_size, height, width, channels = high_res_shape

    for i in range(height):
        for j in range(width):
            for c in range(channels):
                np_p_logits = session.run(
                    p_logits,
                    feed_dict={model.high_res_image: high_res_out_image}
                )

                sum_logits = np_c_logits[:, i, j, c * 256:(c + 1) * 256] + \
                             np_p_logits[:, i, j, c * 256:(c + 1) * 256]

                new_pixel = logits_2_pixel_value(sum_logits, mu=mu)
                high_res_out_image[:, i, j, c] = new_pixel

    return high_res_out_image


def get_slice_bounds(shape):
    start = int(shape[1]/2 - 4)
    end = int(shape[1]/2 + 4)
    return start, end


def predict(input_img_path, output_img_path, ckpt, patch_size):
    ip_img_arr = imread(input_img_path)
    sample_input = ip_img_arr[0:patch_size, 0:patch_size, ].astype("float32")
    sample_input = np.expand_dims(sample_input, axis=0)

    batch_size, height, width, channels = sample_input.shape
    sample_output_shape = [batch_size, height * 4, width * 4, channels]
    op_img_arr = np.zeros((64, 64, 3), "float32")

    model = create_network(list(sample_input.shape), sample_output_shape)
    c_logits = model.net.conditioning_logits
    p_logits = model.net.prior_logits

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as session:
        # Restore checkpoint.
        saver = tf.train.Saver(tf.global_variables())
        saver.restore(session, ckpt)

        rows, columns, _ = ip_img_arr.shape

        for x in range(0, 16):
            if x + patch_size > 16:
                continue
            for y in range(0, 16):
                if y + patch_size > 16:
                    continue
                s = ip_img_arr[x:x+patch_size, y:y+patch_size, ]
                low_res_input = np.expand_dims(s, axis=0)

                # out_high_res = enhance(
                #     model,
                #     c_logits,
                #     p_logits,
                #     session,
                #     low_res_input=low_res_input,
                #     high_res_shape=sample_output_shape
                # )
                st, end = get_slice_bounds(sample_output_shape)
                test_out = np.zeros((64, 64, 3), "float32")
                test_out[:, :, ] = 255.0
                op_img_arr[(x*4)+st:(x*4)+end, (y*4)+st:(y*4)+end, ] += test_out[st:end, st:end, ]

    nv = np.uint8(op_img_arr/4)
    import matplotlib.pyplot as plt
    plt.imshow(nv)
    plt.show()
    save_image(op_img_arr, output_img_path)


