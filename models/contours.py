
import cv2 as cv
import numpy as np
from itertools import islice
from functools import lru_cache
import pickle



# This loads a support vector machine to classify contours and predict the number
# of characters inside them
with open('.contour-classifier', 'rb') as file:
    contour_clasifier = pickle.load(file)

with open('.contour-classifier-preprocessor', 'rb') as file:
    contour_classifier_preprocessor = pickle.load(file)


class Contour:
    '''
    Represents a contour extracted from an image
    '''
    def __init__(self, points, children=[]):
        '''
        Initializes this instance
        :param points: Must be a list the list of points that defines the contour
        See OpenCV contours
        :param children: A list of optional inner contours
        '''
        self.points = points
        self.children = []


    @property
    @lru_cache(maxsize=1)
    def bbox(self):
        '''
        Returns a non rotated rectangle that encapsulates this contour
        '''
        return ContourBBox(self)

    @property
    def bbox_width(self):
        '''
        Its the same as contour.bbox.width
        '''
        return self.bbox.width

    @property
    def bbox_height(self):
        '''
        Its the same as contour.bbox.height
        '''
        return self.bbox.height

    @property
    def bbox_area(self):
        '''
        Its the same as contour.bbox.area
        '''
        return self.bbox.area

    @property
    def bbox_ratio(self):
        '''
        Its the same as contour.bbox.ratio
        '''
        return self.bbox.ratio


    @property
    def area(self):
        '''
        Returns the area covered by this contour
        '''
        return cv.contourArea(self.points)

    @property
    def extent(self):
        '''
        Returns the ratio between the areas of the contour and its bounding rectangle
        '''
        return self.area / self.bbox_area

    @property
    def perimeter(self):
        '''
        Returns the perimeter of this contour
        '''
        return cv.arcLength(self.points, True)


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

    def extract_bbox_pixels(self, img):
        '''
        Its an alias of self.bbox.extract_pixels
        '''
        return self.bbox.extract_pixels(img)


    @property
    @lru_cache(maxsize=1)
    def num_chars(self):
        '''
        :return Returns an intenger number in the range [0, 5] indicating the
        predicted number of chars inside this contour.
        '''
        return self.num_chars_proba.argmax().item()

    @property
    def num_chars_proba(self):
        '''
        Returns an array of size m where the element at index k is the predicted probability
        of this contour of having k characters inside it
        All elements are in the interval [0, 1] and their sum gives 1
        The probability of this contour of having more than m characters is considered 0
        '''
        model = contour_clasifier
        scaler = contour_classifier_preprocessor

        X = np.array([
            self.bbox_width, self.bbox_height,
            self.area, self.extent, self.perimeter
        ], dtype=np.float32).reshape([1, -1])

        X = scaler.transform(X)

        y = model.predict_proba(X)[0]
        return y

    @property
    @lru_cache(maxsize=1)
    def properties(self):
        '''
        Returns a dictionary carrying information about this contour
        '''
        return {
            'bbox_width': self.bbox_width,
            'bbox_height': self.bbox_height,
            'area': self.area,
            'extent': self.extent,
            'perimeter': self.perimeter
        }


    def _draw(img, color=(0, 255, 0)):
        return cv.drawContours(img, self.points, -1, tuple(color))



class ContourBBox:
    '''
    Represents an AABB Rectangle bounding box of a contour
    '''
    def __init__(self, contour):
        '''
        :param contour: Is the contour that is surrounded by this rectangle box
        '''
        self.inner_contour = contour
        self.left, self.top, self.width, self.height = cv.boundingRect(contour.points)


    @property
    def right(self):
        '''
        Returns the right side of the rectangle
        '''
        return self.left + self.width

    @property
    def bottom(self):
        '''
        Returns the bottom of the rectangle
        '''
        return self.top + self.height

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


    def extract_pixels(self, img):
        '''
        Returns an image of the same size as this bounding box extracting the pixels
        from the image
        '''
        return img[self.top:self.top+self.height, self.left:self.left+self.width]


    def draw(self, img, color=(0, 255, 0)):
        '''
        Draws this rectangle over the specified image. Must be an RGB image with uint8 data type
        '''
        img = cv.rectangle(img,
                        (self.left,self.top), (self.left+self.width,self.top+self.height),
                        tuple(color), 1)
        return img

    def __str__(self):
        return '({}, {}), {} x {}'.format(self.left, self.top, self.width, self.height)


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
    ret, thresholded = cv.threshold(inverted, 140, 255, cv.THRESH_BINARY)

    # Apply median blur to reduce image noise
    blurred = cv.medianBlur(thresholded, 3)

    # Apply a mask to eliminate tiny objects and horizontal lines
    kernel = np.array([
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
    ]).astype(np.uint8)
    kernel2 = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0]
    ]).astype(np.uint8)

    mask = cv.morphologyEx(blurred, cv.MORPH_OPEN, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_DILATE, kernel2)
    processed = cv.bitwise_and(blurred, mask)

    # Detect edges
    #edges = cv.Laplacian(processed, 5).clip(0, 255).astype(np.uint8)

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
        items[k] = Contour(contours[k], processed)

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
    import operator
    from itertools import product
    from functools import reduce

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
    contour_bbox_width, contour_bbox_height = [], []
    contour_extent, contour_area = [], []
    contour_perimeter, contour_num_chars = [], []
    contour_id = []
    for k in range(0, n):
        for i in range(0, len(contours[k])):
            contour = contours[k][i]
            contour_id.append('Sample {}, index {}'.format(k, i))
            contour_bbox_width.append(contour.bbox_width)
            contour_bbox_height.append(contour.bbox_height)
            contour_extent.append(contour.extent)
            contour_area.append(contour.area)
            contour_perimeter.append(contour.perimeter)
            contour_num_chars.append(contour.num_chars)

    df = pd.DataFrame.from_dict({
        'ids': contour_id,
        'bbox_width': contour_bbox_width,
        'bbox_height': contour_bbox_height,
        'extent': np.round(contour_extent, 2),
        'area': np.round(contour_area, 2),
        'perimeter': np.round(contour_perimeter, 2),
        'num_chars_predicted': contour_num_chars
    })
    print(df)

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

            plt.title('Num chars predicted: {}'.format(', '.join([str(contour.num_chars) for contour in contours[k]])))
            plt.xticks([])
            plt.yticks([])
    plt.tight_layout()
    plt.show()
