from numba import cuda
import sys
import math

if len(sys.argv) < 3:
    print(f"usage: {sys.argv[0]} SRC_IMAGE DST_IMAGE")

src_img = sys.argv[1]
dst_img = sys.argv[2]

from PIL import Image
import numpy as np

img = Image.open(src_img)
src = np.array(img)

#On range les données en mémoire de manière contigues (sans espaces entre-elles)
#Conseillé dans le cours
src_contigous = np.ascontiguousarray(src)

#on prends les dimensions de l'image
height, width, _ = src.shape

@cuda.jit
def computeImage(img,dst):
    #on prend l'indice global
    x,y = cuda.grid(2)
    #On verifie si on est pas hors limite
    if x < dst.shape[0] and y < dst.shape[1]:
        #on compute
        red = img[x, y, 0]
        green = img[x, y, 1]
        blue = img[x, y, 2]
        gray = 0.3 * red + 0.59 * green + 0.11 * blue
        dst[x, y] = gray

#un thread par pixel
block_size = (1,1)
#On calcule la taille de notre grille --> taille de l'image
grid_size = (math.ceil(height / block_size[0]), math.ceil(width / block_size[1]))

# on copie le tableau contigue
input_cuda_rgb_img = cuda.to_device(src_contigous)
output_cuda_gray_img = cuda.device_array((height, width))

# On compute l'image
computeImage[grid_size, block_size](input_cuda_rgb_img, output_cuda_gray_img)

#on copie le résultat vers le cpu
grayscale_image = output_cuda_gray_img.copy_to_host()

# on enregistre l'image
grayscale_pil_image = Image.fromarray(grayscale_image.astype(np.uint8))
grayscale_pil_image.save('grayscale_image.jpg')