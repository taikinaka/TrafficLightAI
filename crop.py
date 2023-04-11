import argparse

from jetson_utils import (cudaAllocMapped, cudaConvertColor, cudaCrop,
                          cudaResize, cudaMemcpy, cudaDeviceSynchronize,
                          cudaDrawCircle, cudaDrawLine, cudaDrawRect,
                          loadImage, saveImage)

# parse the command line
parser = argparse.ArgumentParser(description='Perform some example CUDA processing on an image')

parser.add_argument("file_in", type=str, default="images/granny_smith_1.jpg", nargs='?', help="filename of the input image to process")
parser.add_argument("file_out", type=str, default="images/test/cuda-example.jpg", nargs='?', help="filename of the output image to save")

parser.add_argument("x1", type=float, default="0", nargs='?', help="left top cord x")
parser.add_argument("y1", type=float, default="0", nargs='?', help="left top cord y")
parser.add_argument("x2", type=float, default="0", nargs='?', help="right bottom cord x")
parser.add_argument("y2", type=float, default="0", nargs='?', help="right bottom cord y")
opt = parser.parse_args()


# convert colorspace
def convert_color(img, output_format):
	converted_img = cudaAllocMapped(width=img.width, height=img.height, format=output_format)
	cudaConvertColor(img, converted_img)
	return converted_img


# center crop an image
def crop(img, crop_factor):
	crop_border = ((1.0 - crop_factor[0]) * 0.5 * img.width,
				(1.0 - crop_factor[1]) * 0.5 * img.height)

	#crop_roi = (crop_border[0], crop_border[1], img.width - crop_border[0], img.height - crop_border[1])
	crop_roi = (362.304688, 49.493408, 419.140625, 125.024414)
	print(crop_roi)
	crop_img = cudaAllocMapped(width=img.width * crop_factor[0],
							   height=img.height * crop_factor[1],
							   format=img.format)

	cudaCrop(img, crop_img, crop_roi)
	return crop_img

def crop2(img, crop_rectangle_cordinates):

	crop_img = cudaAllocMapped(width=crop_rectangle_cordinates[2]-crop_rectangle_cordinates[0],
							   height=crop_rectangle_cordinates[3]-crop_rectangle_cordinates[1],
							   format=img.format)
	print(crop_rectangle_cordinates)
	cudaCrop(img, crop_img, crop_rectangle_cordinates)
	return crop_img

#detected obj 3  class #10 (traffic light)  confidence=0.677246
#bounding box 3  (362.304688, 49.493408)  (419.140625, 125.024414)  w=56.835938  h=75.531006

# resize an image
def resize(img, resize_factor):
	resized_img = cudaAllocMapped(width=img.width * resize_factor[0],
								  height=img.height * resize_factor[1],
                                  format=img.format)

	cudaResize(img, resized_img)
	return resized_img


# load the image
input_img = loadImage(opt.file_in)


print('input image:')
print(input_img)

# crop the image
#crop_img = crop(input_img, (0.75, 0.75))
crop_img = crop2(input_img, (int(opt.x1), int(opt.y1), int(opt.x2), int(opt.y2)))
print('cropped image:')
print(crop_img)

# save the image
if opt.file_out is not None:
	cudaDeviceSynchronize()
	saveImage(opt.file_out, crop_img)
	print("saved {:d}x{:d} test image to '{:s}'".format(crop_img.width, crop_img.height, opt.file_out))
