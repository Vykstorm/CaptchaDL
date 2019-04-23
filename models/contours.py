
import cv2 as cv
import numpy as np
from itertools import islice

class Contour:
    '''
    Represents a contour extracted from an image
    '''
    def __init__(self, points, children=[]):
        self.points = points
        self.children = []

    @property
    def bbox(self):
        '''
        Returns a non rotated rectangle that encapsulates this contour
        '''
        left, top, width, height = cv.boundingRect(self.points)
        return AABBRect(left, left + width, top, top + height)

    def bbox_width(self):
        '''
        Its the same as contour.bbox.width
        '''
        return self.bbox.width

    def bbox_height(self):
        '''
        Its the same as contour.bbox.height
        '''
        return self.bbox.height

    def bbox_area(self):
        '''
        Its the same as contour.bbox.area
        '''
        return self.bbox.area

    def bbox_ratio(self):
        '''
        Its the same as contour.bbox.ratio
        '''
        return self.bbox.ratio


    def bbox_pixels(self, img):
        '''
        Its the same as contour.bbox.bbox_pixels()
        '''
        return self.bbox.bbox_pixels(img)


    def bbox_fill_ratio(self, img):
        '''
        Its the same as contour.bbox.fill_ratio()
        '''
        return self.bbox.fill_ratio(img)

    def area(self):
        '''
        Returns the area covered by this contour
        '''
        return cv.contourArea(self.points)

    def extent(self):
        '''
        Returns the ratio between the areas of the contour and its bounding rectangle
        '''
        return self.area / self.bbox_area


    def extent(self):
        '''
        Returns the extent of this contour. The ratio of the area defined by the
        contour and the area of its bounding box
        '''
        return float(cv.contourArea(self.points)) / self.bbox_area


    def draw(self, img, show_children=False):
        '''
        Draws this contour over the image specified.  Must be an RGB image with uint8 data type
        :param show_children: When this is set to True, it also draws children contours
        '''
        if show_children:
            for child in self.children:
                img = child.draw(img, show_children)
        return self._draw(img)


    def draw_bbox(self, img, show_children=False):
        '''
        Its the same as self.bbox.draw(). Also if show_children is True, it calls
        draw_bbox() on children contours
        '''
        if show_children:
            for child in self.children:
                img = child.draw_bbox(img, show_children)
        return self.bbox.draw(img)

    def _draw(img, color=(0, 255, 0)):
        return cv.drawContours(img, self.points, -1, tuple(color))



class AABBRect:
    '''
    Represents an AABB Rectangle
    '''
    def __init__(self, left, right, top, bottom):
        '''
        :param left: Left side of the rectangle
        :param right: Right side of the rectangle
        :param top: Top side of the rectangle
        :param bottom: Bottom side of the rectangle
        '''
        self.left, self.right = left, right
        self.top, self.bottom = top, bottom

    @property
    def width(self):
        '''
        Returns the width of the rectangle
        '''
        return self.right - self.left

    @property
    def height(self):
        '''
        Returns the height of the rectangle
        '''
        return self.bottom - self.top

    @property
    def area(self):
        '''
        Returns the area of the rectangle
        '''
        return self.width * self.height

    @property
    def ratio(self):
        '''
        Returns the ratio: width/height of the rectangle
        '''
        return self.width / self.height


    def pixels(self, img):
        '''
        Returns the pixels of the specified image that are inside this bounding
        box limits
        '''
        return img[self.top:self.top+self.height, self.left:self.left+self.width]

    def fill_ratio(self, img):
        '''
        Returns the ratio of the number of non-black pixels (intensity greater than 0)
        in the specified image (inside the limits of the bounding box)
        '''
        return (self.pixels(img) > 0).flatten().mean()

    def __str__(self):
        return '({}, {}), {} x {}'.format(self.left, self.top, self.width, self.height)

    def draw(self, img, color=(0, 255, 0)):
        '''
        Draws this rectangle over the specified image. Must be an RGB image with uint8 data type
        '''
        img = cv.rectangle(img,
                        (self.left,self.top), (self.left+self.width,self.top+self.height),
                        tuple(color), 1)
        return img

def find_contours(img):
    '''
    This method finds the contours in the given image
    It is optimized for the captcha dataset images
    :param img: Must be an image on gray format (2D array of float32 values with the pixel intensities
    in the range [0, 1])
    :return Returns a list of contour instances
    '''
    img = (img * 255).astype(np.uint8)

    # Invert the image
    inverted = 255 - img

    # Apply thresholding
    ret, thresholded = cv.threshold(inverted, 180, 255, cv.THRESH_BINARY)

    # Apply morphological transformation to clean image
    kernel = np.array([
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0]
    ]).astype(np.uint8)

    processed = cv.morphologyEx(thresholded, cv.MORPH_ERODE, kernel)
    processed = cv.morphologyEx(processed, cv.MORPH_DILATE, kernel)

    # Find contours

    contours, hierachy = cv.findContours(processed, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

    n = len(contours)
    contour_parent = hierachy[0, :, 3]
    contour_children = [set() for k in range(0, n)]

    for k in range(0, n):
        parent = contour_parent[k]
        if parent != -1:
            contour_children[parent].add(k)

    # Return all the contours (instances of class Contour)

    items = {}
    A = set([k for k in range(0, n) if len(contour_children[k]) == 0])
    B = set(range(0, n)) - A
    for k in A:
        items[k] = Contour(contours[k])

    while len(B) > 0:
        C = B & set([contour_parent[k] for k in A])
        for j in C:
            items[j] = Contour(contours[j], [items[k] for k in contour_children[j]])
        B -= C

    return [items[k] for k in range(0, n) if contour_parent[k] == -1]


def draw_contours(img, contours, show_children=False):
    '''
    Draws all the contours over the specified image (must be a 2D array
    with float32 values with the intensities of the pixels in the range [0, 1])
    :param show_children: If this is disabled, only top level contours are drawn
    '''
    for contour in contours:
        img = contour.draw(img, show_children)
    return img

def draw_bbox_contours(img, contours, show_children=False):
    for contour in contours:
        img = contour.draw_bbox(img, show_children)
    return img


if __name__ == '__main__':
    # Unitary test
    # The next code loads a couple of images and extract their contours (they will be
    # printed over the images)

    from input import InputFlow
    from dataset import CaptchaDataset
    import matplotlib.pyplot as plt
    import pandas as pd

    # Get a batch of samples of size n
    n = 9
    dataset = CaptchaDataset()
    input = iter(InputFlow(dataset.X, dataset.y, batch_size=n))

    X_batch, y_batch = next(input)

    # Extract a list of contours for each sample image

    contours = []
    for k in range(0, n):
        contours.append(find_contours(X_batch[k, :, :, 0]))

    # Show contour properties

    # Show the images and draw the bounding boxes of the contours
    rows, cols = n // 2, 2

    fig, ax = plt.subplots(rows, cols, figsize=(20, 10))
    for i in range(0, rows):
        for j in range(0, cols):
            k = i * cols + j
            plt.sca(ax[i, j])

            img = X_batch[k, :, :, 0]
            img = draw_bbox_contours(cv.cvtColor((img * 255).astype(np.uint8), cv.COLOR_GRAY2RGB), contours[k], show_children=False)
            plt.imshow(img, cmap='gray')

            plt.xticks([])
            plt.yticks([])
    plt.show()
