

'''
This is a helper script to generate a dataset where each sample contains
features of each contour extracted on images in the captcha dataset using
the method find_contours() in the file contours.py

A sample will have the next data attributes:
- Bounding box width: Width of the bounding box that encloses the contour
- Bounding box height: Height of the bounding " "
- Bounding box fill ratio: Ratio of white pixels inside the bounding box " "
- Area: Area covered by the contour (not its bounding box)
- Extent: Coefficient between the contour area and its bounding box area
- Perimiter: Perimiter of the contour
- Num chars: Number of characters that the contour encloses (the user must indicate this attribute
manually)

This tabulated data is stored on a csv file
'''

from config import global_config
import pandas as pd
from os.path import join

DATASET_FILE = join(global_config.HOME_DIR, '.contours-data.csv')

def load_data():
    try:
        return pd.read_csv(DATASET_FILE)
    except:
        return pd.DataFrame(columns=['bbox_width', 'bbox_height', 'bbox_fill_ratio', 'area', 'extent', 'perimeter', 'num_chars'])

def save_data(data):
    data.to_csv(DATASET_FILE, index=False)

def save_sample(attrs):
    save_data(load_data().append(attrs, ignore_index=True))



if __name__ == '__main__':
    from input import InputFlow
    from dataset import CaptchaDataset
    from utils import waitKey
    from contours import find_contours
    import cv2 as cv
    import numpy as np

    dataset = CaptchaDataset()
    batch_generator = iter(InputFlow(dataset.X, dataset.y, batch_size=1))

    while True:
        X_batch, y_batch = next(batch_generator)
        img = X_batch[0, :, :, 0]
        contours = find_contours(img)

        for contour in contours:
            cv.imshow('Contour', contour.draw_bbox(cv.cvtColor((img * 255).astype(np.uint8), cv.COLOR_GRAY2RGB)))

            while True:
                ch = waitKey()
                if ch == 'q':
                    exit()
                elif ord(ch) >= ord('0') and ord(ch) <= ord('5'):
                    num_chars = ord(ch) - ord('0')
                    attrs = contour.properties
                    attrs.update({'num_chars': num_chars})
                    save_sample(attrs)
                    break
