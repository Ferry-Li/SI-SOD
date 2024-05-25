import numpy as np

def binarize(x, threshold=0.5):
    return np.where(x > threshold, 1, 0)

def create_colormap(num_colors):
    # Generate a list of colors for the colormap
    colors = np.random.randint(0, 256, size=(num_colors, 3), dtype=np.uint8)
    return colors
'''
def apply_colormap(arr, colormap):
    # Normalize the input array to the range of the colormap
    normalized_arr = (arr - arr.min()) / (arr.max() - arr.min())
    
    # Convert the normalized array to indices for the colormap
    indices = (normalized_arr * (len(colormap) - 1)).astype(np.uint8)
    
    # Map the indices to colors from the colormap
    colored_image = colormap[indices]

    return colored_image
'''
def apply_colormap(arr, colormap):
    # Normalize the input array to the range of the colormap
    colored_image = np.expand_dims(np.zeros_like(arr), axis=2).repeat(3, axis=2)
    step_list = np.unique(arr)
    for idx, step in enumerate(step_list):
        index = (arr == step)
        colored_image += colormap[idx] * np.expand_dims(index, axis=2).repeat(3, axis=2)
  
    return colored_image

def image_filter(name):
    postfix = name.rsplit('.')[-1]
    if not postfix.lower() in ['png', 'jpg', 'jpeg', 'svg', 'mat']:
        return False
    else:
        return True
