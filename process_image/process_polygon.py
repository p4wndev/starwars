import cv2
import numpy as np
import math
import matplotlib.pyplot as plt


def polygon_area(vertices):
    n = len(vertices)
    if n < 3:
        raise ValueError("A polygon must have at least 3 vertices")

    area = 0.0
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        area += x1 * y2 - x2 * y1

    return abs(area) / 2.0

def crop_polygon_from_image(image, vertices):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(vertices, dtype=np.int32)], 255)
    result = cv2.bitwise_and(image, image, mask=mask)
    x, y, w, h = cv2.boundingRect(np.array(vertices, dtype=np.int32))
    cropped = result[y:y+h, x:x+w]
    return cropped

# def remove_white_pixels(image, threshold=240):
#     white_mask = cv2.inRange(image, np.array([threshold, threshold, threshold]), np.array([255, 255, 255]))
#     non_white_mask = cv2.bitwise_not(white_mask)
#     result = cv2.bitwise_and(image, image, mask=non_white_mask)
#     return result

def remove_white_pixels(image, threshold=240):
    if len(image.shape) == 2:  # Grayscale image
        white_mask = cv2.inRange(image, threshold, 255)
    elif len(image.shape) == 3:  # Color image
        if image.shape[2] == 3:  # RGB
            white_mask = cv2.inRange(image, np.array([threshold, threshold, threshold]), np.array([255, 255, 255]))
        elif image.shape[2] == 4:  # RGBA
            white_mask = cv2.inRange(image[:,:,:3], np.array([threshold, threshold, threshold]), np.array([255, 255, 255]))
        else:
            raise ValueError("Unsupported number of channels")
    else:
        raise ValueError("Unsupported image shape")

    non_white_mask = cv2.bitwise_not(white_mask)
    result = cv2.bitwise_and(image, image, mask=non_white_mask)
    return result

def compute_centroid(vertices):
    if vertices[0] != vertices[-1]:
        vertices.append(vertices[0])

    n = len(vertices) - 1
    A = 0
    C_x = 0
    C_y = 0

    for i in range(n):
        xi, yi = vertices[i]
        xi1, yi1 = vertices[i + 1]
        common_factor = (xi * yi1 - xi1 * yi)

        A += common_factor
        C_x += (xi + xi1) * common_factor
        C_y += (yi + yi1) * common_factor

    A *= 0.5
    C_x /= (6 * A)
    C_y /= (6 * A)

    return C_x, C_y

def extract_polygons(image, min_vertices=3, max_area=100000):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 200, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        if len(approx) >= min_vertices:
            flattened_vertices = [tuple(coord[0]) for coord in approx]
            area = polygon_area(flattened_vertices)
            if area < max_area:
                cropped_image = crop_polygon_from_image(image, flattened_vertices)
                cropped_image = remove_white_pixels(cropped_image, 100)
                polygons.append((flattened_vertices, cropped_image))
    return polygons

def calculate_angles(vertices):
    def polygon_interior_angles(vertices):
        def angle_between(v1, v2):
            dot = np.dot(v1, v2)
            det = np.cross(v1, v2)
            angle = np.arctan2(det, dot)
            return np.degrees(angle) % 360

        num_vertices = len(vertices)
        angles = []

        for i in range(num_vertices):
            p0 = np.array(vertices[i - 1])
            p1 = np.array(vertices[i])
            p2 = np.array(vertices[(i + 1) % num_vertices])

            v1 = p0 - p1
            v2 = p2 - p1

            angle = angle_between(v1, v2)
            if angle > 180:
                angle = 360 - angle

            angles.append(angle)

        # Normalize angles to ensure their sum is exactly (n-2) * 180
        # expected_sum = (num_vertices - 2) * 180
        # current_sum = sum(angles)
        # scale_factor = expected_sum / current_sum
        # angles = [angle * scale_factor for angle in angles] #Nhân góc với tỉ lệ của kỳ vọng/thực tế

        return angles

    polygon_angles = polygon_interior_angles(vertices)
    # print("polygon_angles: ",polygon_angles)
    # print("len(vertices): ", len(vertices))
    # print("Angels difference: ", abs(sum(polygon_angles) - ((len(vertices) - 2) * 180)))

    min_angle_index = np.argmin(polygon_angles)
    min_angle_vertex = vertices[min_angle_index]

    centroid = np.mean(vertices, axis=0)
    unique_vertices = list(dict.fromkeys(vertices))

    angles = []
    for vertex in unique_vertices:
        dx = vertex[0] - centroid[0]
        dy = vertex[1] - centroid[1]
        angle = math.atan2(dy, dx) - math.atan2(min_angle_vertex[1] - centroid[1], min_angle_vertex[0] - centroid[0])
        if angle < 0:
            angle += 2 * math.pi
        angles.append((angle, vertex))

    sorted_angles = sorted(angles, key=lambda x: x[0])
    sorted_vertices = [vertex for angle, vertex in sorted_angles]
    vertex_angles = [math.degrees(angle) for angle, vertex in sorted_angles]

    return centroid, min_angle_vertex, sorted_vertices, vertex_angles, sorted_angles, polygon_angles
