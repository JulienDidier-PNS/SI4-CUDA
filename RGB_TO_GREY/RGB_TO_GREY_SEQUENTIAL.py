import sys

if len(sys.argv) < 3:
    print(f"usage: {sys.argv[0]} SRC_IMAGE DST_IMAGE")

src_img = sys.argv[1]
dst_img = sys.argv[2]

from PIL import Image
import numpy as np

def to_gray(rgb):
    R, G, B = rgb
    return (0.3 * R) + (0.59 * G) + (0.11 * B)


img = Image.open(src_img)
src = np.array(img)
#on prends les dimensions de l'image
height, width, _ = src.shape
#on créé un tableau de nouveaux pixels
dst = np.zeros((height, width), dtype=np.int8)
#pour chaques pixels de l'image
for i in range(height):
    for j in range(width):
        #on convertit les pixels en niveaux de gris
        dst[i, j] = to_gray(src[i, j])
#on génère la nouvelle image
Image.fromarray(dst, mode="L").save(dst_img)