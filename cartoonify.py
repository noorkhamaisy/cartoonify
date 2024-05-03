#################################################################
# FILE : cartoonify.py
# WRITER : Noor Khamaisy , noor.khamaisy , 212809925
# EXERCISE : intro2cs2 ex6 2022
# DESCRIPTION:

#################################################################

import math
import sys
from ex6_helper import *

def separate_channels(image):
    """a function that returns an image as a list of channels list-separates the image channels"""
    channels_num = len(image[0][0])
    channels = []
    for k in range(channels_num):
        channels.append([])
    for i in range(len(image)):
        new_lines = []
        for k in range(channels_num):
            new_lines.append([])
        for j in range(len(image[0])):
            for k in range(channels_num):
                new_lines[k].append(image[i][j][k])
        for k in range(channels_num):
            channels[k].append(new_lines[k])
    return channels

def combine_channels(channels):
    """a function that combines channels to a new image"""
    output_image = []
    for i in range(len(channels[0])):
        new_line = []
        for j in range(len(channels[0][0])):
            channels_in_pixel = []
            for k in range(len(channels)):
                channels_in_pixel.append(channels[k][i][j])
            new_line.append(channels_in_pixel)
        output_image.append(new_line)
    return output_image

def RGB2grayscale(colored_image):
    """function that returns a black-white image"""
    current_sum = 0
    output = []
    for i in range(len(colored_image)):
        newlist = []
        for j in range(len(colored_image[i])):
            red = 0.299 * colored_image[i][j][0]
            green = 0.587 * colored_image[i][j][1]
            blue = 0.114 * colored_image[i][j][2]
            current_sum = red + green + blue
            current_sum = round(current_sum)
            newlist.append(current_sum)
        output.append(newlist)
    return output

def blur_kernel(size):
    """function that returns new kernel"""
    innerlist = []
    exlist = []
    for j in range(size):
        innerlist.append(1 / (size ** 2))
    for i in range(size):
        exlist.append(innerlist)
    return exlist

def get_subimage(image, i, j, kernel_size):
    """function that help us to apply kernel to a given image"""
    top_left_i = i - kernel_size // 2
    top_left_j = j - kernel_size // 2
    subimage = []
    for y in range(top_left_i, top_left_i + kernel_size):
        line = []
        for x in range(top_left_j, top_left_j + kernel_size):
            if y < 0 or y >= len(image) or x < 0 or x >= len(image[0]):
                line.append(image[i][j])
            else:
                line.append(image[y][x])
        subimage.append(line)
    return subimage


def calculate_kernel_value(subimage, kernel):
    """a function that calculate the kernel value to a given image"""
    total_sum = 0
    for i in range(len(subimage)):
        for j in range(len(subimage[0])):
            total_sum += subimage[i][j] * kernel[i][j]
    total_sum = round(total_sum)
    if total_sum < 0:
        total_sum = 0
    if total_sum > 255:
        total_sum = 255
    return total_sum

def apply_kernel(image, kernel):
    """a function that applies a given kernel to the hole image"""
    output = []
    for i in range(len(image)):
        line = []
        for j in range(len(image[0])):
            subimage = get_subimage(image, i, j, len(kernel))
            line.append(calculate_kernel_value(subimage, kernel))
        output.append(line)
    return output

def bilinear_interpolation(image, y, x):
    """function that returns the new value of the pixel"""
    a = image[math.floor(y)][math.floor(x)]
    b = image[math.ceil(y)][math.floor(x)]
    c = image[math.floor(y)][math.ceil(x)]
    d = image[math.ceil(y)][math.ceil(x)]
    dx = x - math.floor(x)
    dy = y - math.floor(y)
    A = a * (1 - dx) * (1 - dy)
    B = b * dy * (1 - dx)
    C = c * dx * (1 - dy)
    D = d * dx * dy
    return round(A + B + C + D)

def resize(image, new_height, new_width):
    """a function that returns a new image to the new given width and height"""
    new_image = []
    for i in range(new_height):
        new_line = []
        for j in range(new_width):
            new_line.append(0)
        new_image.append(new_line)

    row_scale = (len(image) - 1) / (new_height - 1)
    col_scale = (len(image[0]) - 1) / (new_width - 1)

    for i in range(len(new_image)):
        for j in range(len(new_image[0])):
            val = bilinear_interpolation(image, i * row_scale, j * col_scale)
            new_image[i][j] = val
    return new_image

def scale_down_colored_image(image, max_size):
    """function that checks if the image matches the given sizeand if it doesn't it returns a new smaller image"""
    if len(image) <= max_size and len((image[0])) <= max_size:
        return None
    if len(image) >= len(image[0]):
        new_height = max_size
        new_width = round((max_size / len(image)) * len(image[0]))
    else:
        new_height = round((max_size / len(image[0])) * len(image))
        new_width = max_size
    channels = separate_channels(image)
    new_channels = []
    for channel in channels:
        new_channels.append(resize(channel, new_height, new_width))
    return combine_channels(new_channels)

def rotate_90(image, direcrion):
    """ function that turns the image right or left"""
    newlist = []
    if direcrion == 'R':
        for j in range(len(image[0])):
            l = []
            for i in range(len(image) - 1, -1, -1):
                l.append(image[i][j])
            newlist.append(l)
    if direcrion == 'L':
        for j in range(len(image[0]) - 1, -1, -1):
            l = []
            for i in range(len(image)):
                l.append(image[i][j])
            newlist.append(l)
    return newlist


def get_edges(image, blur_size, block_size, c):
    """ function that returns a new image with 2 values-black or white"""
    kernel_blur = blur_kernel(blur_size)
    blurred_image = apply_kernel(image, kernel_blur)
    kernel_block = blur_kernel(block_size)
    threshold = apply_kernel(blurred_image, kernel_block)
    for i in range(len(threshold)):
        for j in range(len(threshold[0])):
            threshold[i][j] -= c
    new_image = []
    for i in range(len(blurred_image)):
        line = []
        for j in range(len(blurred_image[0])):
            if threshold[i][j] > blurred_image[i][j]:
                line.append(0)
            else:
                line.append(255)
        new_image.append(line)
    return new_image

def quantize(image, N):
    """function that returns an image it's channels are calculated according to a given formula"""
    qimg = []
    for i in range(len(image)):
        line = []
        for j in range(len(image[0])):
            val = round(math.floor(image[i][j] * (N / 256)) * (255 / (N - 1)))
            line.append(val)
        qimg.append(line)
    return qimg


def quantize_colored_image(image, N):
    """function that returns a colored image after quantizing"""
    channels = separate_channels(image)
    new_channels = []
    for i in range(len(channels)):
        new_channels.append(quantize(channels[i], N))
    return combine_channels(new_channels)

def add_mask(image1, image2, mask):
    """function that combines two images and mask according to a given formula"""
    new_image = []
    if type(image1[0][0]) is int:
        for i in range(len(image1)):
            line = []
            for j in range(len(image1[0])):
                line.append(round(image1[i][j] * mask[i][j] + image2[i][j] * (1 - mask[i][j])))
            new_image.append(line)
    else:
        for i in range(len(image1)):
            line = []
            for j in range(len(image1[0])):
                pixel = []
                for k in range(len(image1[0][0])):
                    pixel.append(round(image1[i][j][k] * mask[i][j] + image2[i][j][k] * (1 - mask[i][j])))
                line.append(pixel)
            new_image.append(line)
    return new_image

def cartoonify(image, blur_size, th_block_size, th_c, quant_num_shades):
    """ function that returns a cartoon version to a given image"""
    edges = get_edges(RGB2grayscale(image), blur_size, th_block_size, th_c)
    colored_quantized = quantize_colored_image(image, quant_num_shades)
    image1 = []
    for i in range(len(colored_quantized[0][0])):
        image1.append(edges)
    image1 = combine_channels(image1)
    mask = []
    for i in range(len(edges)):
        line = []
        for j in range(len(edges[0])):
            if edges[i][j] == 0:
                line.append(0)
            else:
                line.append(1)
        mask.append(line)
    cartoon = add_mask(colored_quantized,image1, mask)
    return cartoon


if __name__ == '__main__':
    if len(sys.argv) != 8:
        print("Invalid number of arguments")
    else:
        image_source = sys.argv[1]
        cartoon_dest = sys.argv[2]
        max_im_size = int(sys.argv[3])
        blur_size = int(sys.argv[4])
        th_block_size = int(sys.argv[5])
        th_c = int(sys.argv[6])
        quant_num_shades = int(sys.argv[7])

        image = load_image(image_source)
        resized = scale_down_colored_image(image, max_im_size)
        if resized is not None:
            image = resized

        cartoon = cartoonify(image, blur_size, th_block_size, th_c, quant_num_shades)
        save_image(cartoon, cartoon_dest)
