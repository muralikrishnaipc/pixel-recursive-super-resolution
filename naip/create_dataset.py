import os
import rasterio

base_dir = "D:\\NAIP"
count = 0


def create_rgb_raster(image_path, out_path):
    image_raster = rasterio.open(image_path)
    image_arr = image_raster.read()
    rgb_image_arr = image_arr[0:3, :, :]
    with rasterio.open(out_path,
                       "w",
                       driver="PNG",
                       height=rgb_image_arr.shape[1],
                       width=rgb_image_arr.shape[2],
                       count=3,
                       dtype="uint8",
                       crs=image_raster.crs,
                       transform=image_raster.transform) as dst:
        dst.write(rgb_image_arr)


def create_new_res_raster(input_img_path, output_img_path, res, interpolation="average"):
    os.system("gdal_translate -of PNG -tr %s %s -r %s %s %s" % (res, res, interpolation, input_img_path, output_img_path))


def subset_img(image_path, sub_dir, patch_size):
    global count
    _, columns, rows = rasterio.open(image_path).read().shape
    for x in range(0, rows, patch_size):
        if x + patch_size > columns:
            continue
        for y in range(0, columns, patch_size):
            if y + patch_size > rows:
                continue
            count = count+1
            ss_img = os.path.join(base_dir, sub_dir, str(count)+".png")
            os.system("gdal_translate -srcwin %s %s %s %s %s %s"%(x, y, patch_size,patch_size,image_path,ss_img))
            print(x, y)
    print(count)


for file in os.listdir(os.path.join(base_dir, "ms_0.6m")):
    if file.endswith(".jp2"):
        multi_spectral_img = os.path.join(base_dir, "ms_0.6m", file)
        orig_rgb_path = os.path.join(base_dir, "rgb_0.6m", file)
        create_rgb_raster(multi_spectral_img, orig_rgb_path)

        # creation of high resolution(2.5m) files
        output_img_250cm = os.path.join(base_dir, "rgb_2.5m", file.replace("_060_", "_250_").strip("jp2")+"png")
        create_new_res_raster(orig_rgb_path, output_img_250cm, 2.5, "average")
        subset_img(output_img_250cm, "ss_new_2.5m", 32)

        # creation of low resolution(10m) files
        # output_img_10m = os.path.join(base_dir, "rgb_10m", file.replace("_060_", "_1000_").strip("jp2")+"png")
        # create_new_res_raster(orig_rgb_path, output_img_10m, 10, "average")
        # subset_img(output_img_10m, "ss_10m", 8)
