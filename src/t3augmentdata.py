import pickle
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import math
from PIL import Image
import os


sentences_to_read = open("sentences.pickle", "rb")
sentences = pickle.load(sentences_to_read)
sentences_to_read.close()
#print(sentences)

images_to_read = open("images.pickle", "rb")
images = pickle.load(images_to_read)
images_to_read.close()
#print(images)

# plt.imshow(images[0])
# plt.show()
# plt.clf()

print(images[0].shape)

# plt.imshow(images[0])
# plt.show()
# plt.clf()
def pad_frame_once(src_: list, pad) -> list:
    output = [[pad, *line, pad] for line in src_]
    return [[pad] * len(output[0]), *output, [pad] * len(output[0])]


def pad_grid(src_, padding_size: int, pad=255):
    reference = src_
    for _ in range(padding_size):
        reference = pad_frame_once(reference, pad)

    return reference


xpar = [[2, 30], [2, 50], [4, 80], [2, 80]]
ypar = [[2, 30], [4, 80], [2, 70]]

imagesruin = images.copy()

# imagesc = []
# sentencesc = []
imageindex = 7000 ### 0 
for loop in range(1): ### 7
    imagesc = []
    sentencesc = []
    for imgr in range(imageindex, imageindex+458):#len(images) ### imageindex+1000
        imagesc.append(images[imgr])
        sentencesc.append(sentences[imgr])
        print(imgr, end = ' ', flush=True)
        imagesruin[imgr] = np.array(pad_grid(imagesruin[imgr], 4))
        for xr in range(len(xpar)):
            for yr in range(len(ypar)):
                img_output = np.zeros(imagesruin[imgr].shape, dtype=imagesruin[imgr].dtype)
                rows, cols = img_output.shape
                for i in range(rows):
                    for j in range(cols):
                        offset_x = int(xpar[xr][0] * math.sin(2 * 3.14 * i / xpar[xr][1]))
                        offset_y = int(ypar[yr][0] * math.cos(2 * 3.14 * j / ypar[yr][1]))
                        if i+offset_y < rows and j+offset_x < cols:
                            img_output[i,j] = imagesruin[imgr][(i+offset_y)%rows,(j+offset_x)%cols]
                        else:
                            img_output[i,j] = 255
                # plt.title(xpar[xr]+ypar[yr])
                # plt.imshow(img_output)
                # plt.show()
                # plt.clf()
                imagesc.append(img_output)
                sentencesc.append(sentences[imgr])
    imageindex += 1000
    loop = 7
    images_to_store = open(str(loop)+"imagesc.pickle", "wb")
    pickle.dump(imagesc, images_to_store)
    images_to_store.close()

    sentences_to_store = open(str(loop)+"sentencesc.pickle", "wb")
    pickle.dump(sentencesc, sentences_to_store)
    sentences_to_store.close()


# plt.imshow(img_output)
# plt.show()
# plt.clf()
# plt.imshow(images[0])
# plt.show()
# plt.clf()

