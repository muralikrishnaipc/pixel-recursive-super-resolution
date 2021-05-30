import os
from prsr import predict

input_imgs_dir = "../test_images/"
output_imgs_dir = "../output_images/"
ckpt = "../models/model.ckpt-170000"
file = "lrl_img.png"
patch_size = 8

# for file in os.listdir(input_imgs_dir):
input_img_path = os.path.join(input_imgs_dir, file)
output_img_path = os.path.join(output_imgs_dir, "hrl3_img.png")
predict.predict(input_img_path, output_img_path, ckpt, patch_size)









