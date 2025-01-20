import cv2

def draw_3d_bounding_boxes_on_image(image, bounding_boxes, bb_color):
    """
    Draws bounding boxes directly on an image represented as a NumPy array and returns the modified array.

    Args:
    - image: NumPy array of the image (expected to be in RGB format).
    - bounding_boxes: List of bounding boxes, each box represented as a list or array of 8 points (corners).
    - bb_color: Color of the bounding box lines (R, G, B) format since we're working with an RGB image.

    Returns:
    - Modified image array with bounding boxes drawn on it.
    """

    for bbox in bounding_boxes:
        points = [(int(bbox[i, 0]), int(bbox[i, 1])) for i in range(8)]
        # generate ransom points for testing
        # points = [(int(np.random.rand() * 1248), int(np.random.rand() * 384)) for i in range(8)]
        # Draw lines on the image using cv2.line
        # Base
        cv2.line(image, points[0], points[1], bb_color, 1)
        cv2.line(image, points[1], points[2], bb_color, 1)
        cv2.line(image, points[2], points[3], bb_color, 1)
        cv2.line(image, points[3], points[0], bb_color, 1)
        # Top
        cv2.line(image, points[4], points[5], bb_color, 1)
        cv2.line(image, points[5], points[6], bb_color, 1)
        cv2.line(image, points[6], points[7], bb_color, 1)
        cv2.line(image, points[7], points[4], bb_color, 1)
        # Base to Top
        cv2.line(image, points[0], points[4], bb_color, 1)
        cv2.line(image, points[1], points[5], bb_color, 1)
        cv2.line(image, points[2], points[6], bb_color, 1)
        cv2.line(image, points[3], points[7], bb_color, 1)

    return image