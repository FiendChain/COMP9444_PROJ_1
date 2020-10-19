import numpy as np

def encode_image(im):
    h, w = im.shape

    nb_hbits = h-1
    nb_wbits = w-1

    points = []

    for y in np.arange(0, h):
        for x in np.arange(0, w):
            if not im[y,x]:
                continue
            row = np.concatenate([
                np.ones(x), np.zeros(nb_wbits-x),
                np.ones(y), np.zeros(nb_hbits-y)
            ])
            points.append(row)
    
    return np.array(points)

