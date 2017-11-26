import numpy as np
from IPython.display import Image
from IPython.display import display
import matplotlib.pylab as plt
import cv2

def flatten(lists):
    if type(lists)==str:
        return [lists]
    else:
        return lists
            

def show_images(imnames, titles=None, rows = 1, figsize=(30,8)):

    images = [plt.imread(imname) for imname in flatten(imnames)]
    n_images = len(images)
    
    if titles is None: titles = ['Image (%d)' % i for i in range(n_images + 1)]
    elif type(titles)==str: titles = [titles+'(%d)'%i for i in range(n_images + 1)]
        
    fig = plt.figure(figsize=figsize)
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(rows, np.ceil(n_images/float(rows)), n + 1)
        image = np.squeeze(image)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        plt.axis('off')
        a.set_title(title, fontsize=20)
    # fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap

def apply_threshold(heatmap, threshold):
    heatmap[heatmap<=threshold] = 0
    return heatmap


def draw_labeled_bboxes(img, labels, color=(1.0, 0, 0)):
    img = np.copy(img)
    # Iterate through all detected cars
    for car_number in range(labels[1]+1):
        # Find pixels with each car_number label value
        nonzero=(labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], color, 6)
    # Return the image
    return img

