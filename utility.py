import numpy as np
from IPython.display import Image
from IPython.display import display
import matplotlib.pylab as plt

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

