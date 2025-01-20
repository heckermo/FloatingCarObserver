import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import importlib


def get_image_vector(vector, radius, image_size):
    """
    This function scales the vector to the image size
    """
    scaling_factor = image_size / (2 * radius)
    return [int(x * scaling_factor) for x in vector]



if __name__ == '__main__':
    dataset_path = "data/i3040_newdataset_100m"
    dataset = pd.read_csv(os.path.join(dataset_path, "dataset.csv"))
    dataset['vector'] = dataset.vector.apply(lambda x: eval(x))
    image_pointers = dataset.image_pointer.unique()
    # load the radius the config cv file
    radius = 50
    for image_pointer in image_pointers:
        vectors = dataset[dataset.image_pointer == image_pointer].vector.values.tolist()

        # load the image to plot the vectors on and add the matplotlib coordinate system
        image = plt.imread(os.path.join(dataset_path,'bev', image_pointer))
        height, width, _ = image.shape

        center_x, center_y = width / 2, height / 2
        scaled_vectors = []
        for vector in vectors:
            scaled_vectors.append(get_image_vector(vector, radius, height))


        plt.figure()
        plt.imshow(image)
        plt.scatter(center_x, center_y, color='red', s=5)
        for vector in scaled_vectors:
            plt.arrow(center_x, center_y, vector[0], -vector[1], width=0.5, color='yellow', head_width=1)
        plt.grid(True)  # To show grid which represents the coordinate system
        plt.xlabel('X axis')  # Label for X axis
        plt.ylabel('Y axis')  # Label for Y axis
        plt.savefig('vector_image.png')
