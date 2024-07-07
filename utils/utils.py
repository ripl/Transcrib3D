import numpy as np


def function_name(param1, param2):
    """
    One-line summary of the function's purpose.

    Detailed description explaining what the function does, the input parameters,
    the return value, and any side effects.

    Parameters:
        param1 (type): Description of param1
        param2 (type): Description of param2
        ...

    Returns:
        type: Description of the return value

    Raises:
        ExceptionType: Description of the circumstances in which the exception is thrown

    Examples:
        Provide an example of how to use the function.

    """
    # Function implementation
    pass


def round_list(lst:list, length:float):
    """
    Round every element in a list with certain length.

    Parameters:
        lst (list): The input list.
        length (int): The the number of decimal places retained.

    Returns:
        list: The rounded list

    Examples:
        round_list([1.23, 4.56], 1) returns [1.2, 4.6]
    """
    for idx, num in enumerate(lst):
        lst[idx] = round(num, length)
    return list(lst)

def remove_spaces(s: str):
    """
    Remove spaces in a string.
    """
    return s.replace(' ', '')

def center_size_to_extension(box_center_size:list):
    """
    Convert a bounding box in center-size format to extension format.

    Parameters:
        box_center_size (list): A box in in center-size format: [cx, cy, cz, sx, sy, sz].

    Returns:
        list: A box in in extension format: [xmin, ymin, zmin, xmax, ymax, zmax]

    Examples:
        A cube box at [0, 0, 0] with side length 1 -- [0, 0, 0, 1, 1, 1] will be converted to [-1, -1, -1, 1, 1, 1]
    """
    cx, cy, cz, sx, sy, sz = box_center_size
    xmin = cx - sx / 2
    xmax = cx + sx / 2
    ymin = cy - sy / 2
    ymax = cy + sy / 2
    zmin = cz - sz / 2
    zmax = cz + sz / 2
    return [xmin, ymin, zmin, xmax, ymax, zmax]

def extension_to_center_size(extension:list):
    """
    Convert a bounding box in extension format to center-size format.
    """
    xmin, ymin, zmin, xmax, ymax, zmax = extension
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2
    cz = (zmin + zmax) / 2
    sx = xmax - xmin
    sy = ymax - ymin
    sz = zmax - zmin
    return [cx, cy, cz, sx, sy, sz]

def calc_iou(box1:list, box2:list):
    """
    Calculate the IoU of two bounding boxes.

    Parameters:
        box1 (list): A box in in extension format: [xmin, ymin, zmin, xmax, ymax, zmax].
        box2 (list): Another box in in extension format: [xmin, ymin, zmin, xmax, ymax, zmax].

    Returns:
        float: IoU value.
    """
    x1_min, y1_min, z1_min, x1_max, y1_max, z1_max = box1
    x2_min, y2_min, z2_min, x2_max, y2_max, z2_max = box2
    # itersection volume
    x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    z_overlap = max(0, min(z1_max, z2_max) - max(z1_min, z2_min))
    intersection_volume = x_overlap * y_overlap * z_overlap
    # volume of two boxes
    volume_box1 = (x1_max - x1_min) * (y1_max - y1_min) * (z1_max - z1_min)
    volume_box2 = (x2_max - x2_min) * (y2_max - y2_min) * (z2_max - z2_min)
    # calculate IoU
    union_volume = volume_box1 + volume_box2 - intersection_volume
    iou = intersection_volume / union_volume if union_volume > 0 else 0.0
    return iou

def get_scene_center(objects:list):
    """
    Calculate the center of a scene containing multiple objects.

    Parameters:
        objests (list): A list of objects. Each object is dictionary.

    Returns:
        list: The center coordinate [x, y, z].
    """
    xmin, ymin, zmin = float('inf'), float('inf'), float('inf')
    xmax, ymax, zmax = float('-inf'), float('-inf'), float('-inf')
    for obj in objects:
        x, y, z = obj['center_position']
        if x < xmin:
            xmin = x
        if x > xmax:
            xmax = x
        if y < ymin:
            ymin = y
        if y > ymax:
            ymax = y
        if z < zmin:
            zmin = z
        if z > zmax:
            zmax = z
    return round_list([(xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2], 2)

def rgb_to_hsv(rgb):
    """
    Convert color value in RGB format to color value in HSV format.

    Parameters:
        rgb (list): RGB value: [r, g, b], 0-255.

    Returns:
        list: HSV value: [H, S, V], H: 0-360, S: 0-100, V: 0-100.
    """
    # Convert RGB to [0, 1] range
    r, g, b = [x / 255.0 for x in rgb]
    # Calculate chroma
    chroma = max(r, g, b) - min(r, g, b)
    # Calculate hue
    hue = 0
    if chroma == 0:
        hue = 0
    elif max(r, g, b) == r:
        hue = (g - b) / chroma % 6
    elif max(r, g, b) == g:
        hue = (b - r) / chroma + 2
    elif max(r, g, b) == b:
        hue = (r - g) / chroma + 4
    hue *= 60  # Convert to degrees
    # Calculate value
    value = max(r, g, b)
    # Calculate saturation
    saturation = 0 if value == 0 else chroma / value
    return [hue, saturation * 100, value * 100]  # Return HSV as percentages

def rgb_to_hsl(rgb):
    """
    Convert color value in RGB format to color value in HSL format.

    Parameters:
        rgb (list): RGB value: [r, g, b], 0-255.

    Returns:
        list: HSV value: [H, S, L], H: 0-360, S: 0-1, L: 0-1.
    """
    # Convert RGB to [0, 1] range
    r, g, b = [x / 255.0 for x in rgb]
    # Calculate min and max values of RGB to find chroma
    c_max = max(r, g, b)
    c_min = min(r, g, b)
    chroma = c_max - c_min
    # Calculate lightness
    lightness = (c_max + c_min) / 2
    # Calculate hue and saturation
    hue = 0
    saturation = 0
    if chroma != 0:
        if c_max == r:
            hue = ((g - b) / chroma) % 6
        elif c_max == g:
            hue = ((b - r) / chroma) + 2
        elif c_max == b:
            hue = ((r - g) / chroma) + 4
        hue *= 60
        # Calculate saturation
        if lightness <= 0.5:
            saturation = chroma / (2 * lightness)
        else:
            saturation = chroma / (2 - 2 * lightness)
    return [hue, saturation, lightness]

if __name__ == "__main__":
    print(round_list([1.23, 4.56], 1))