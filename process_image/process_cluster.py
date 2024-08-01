import numpy as np
import cv2
from process_image.process_polygon import compute_centroid, extract_polygons, polygon_area
import requests

def normalize_image_size(image, target_size=1000):
    """
    Chuẩn hóa kích thước ảnh.

    Tham số:
    - image: Ảnh đầu vào (numpy array)
    - target_size: Kích thước mục tiêu cho cạnh dài nhất của ảnh (mặc định: 1000)

    Trả về:
    - Ảnh đã được chuẩn hóa kích thước

    Chức năng:
    1. Lấy kích thước hiện tại của ảnh
    2. Tính toán tỷ lệ để thay đổi kích thước ảnh sao cho cạnh dài nhất bằng target_size
    3. Thay đổi kích thước ảnh giữ nguyên tỷ lệ
    """
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    return cv2.resize(image, (int(w * scale), int(h * scale)))

def normalize_polygon_orientation(vertices):
    """
    Chuẩn hóa hướng của đa giác.

    Tham số:
    - vertices: Danh sách các đỉnh của đa giác

    Trả về:
    - Danh sách các đỉnh của đa giác đã được chuẩn hóa hướng

    Chức năng:
    1. Tính trọng tâm của đa giác
    2. Dịch chuyển đa giác để trọng tâm nằm tại gốc tọa độ
    3. Tìm cạnh dài nhất của đa giác
    4. Tính góc xoay để cạnh dài nhất nằm ngang
    5. Xoay đa giác theo góc đã tính
    6. Dịch chuyển đa giác trở lại vị trí ban đầu
    """
    centroid = compute_centroid(vertices)
    rotated_vertices = [(x - centroid[0], y - centroid[1]) for x, y in vertices]

    max_dist = 0
    max_index = 0
    for i in range(len(rotated_vertices)):
        x1, y1 = rotated_vertices[i]
        x2, y2 = rotated_vertices[(i + 1) % len(rotated_vertices)]
        dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if dist > max_dist:
            max_dist = dist
            max_index = i

    x1, y1 = rotated_vertices[max_index]
    x2, y2 = rotated_vertices[(max_index + 1) % len(rotated_vertices)]
    angle = np.arctan2(y2 - y1, x2 - x1)

    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]])
    normalized_vertices = [tuple(np.dot(rotation_matrix, vertex)) for vertex in rotated_vertices]

    return [(x + centroid[0], y + centroid[1]) for x, y in normalized_vertices]

def compute_relative_angles(centroid, surrounding_centroids):
    """
    Tính toán góc tương đối giữa các đa giác xung quanh.

    Tham số:
    - centroid: Tọa độ trọng tâm của đa giác gốc
    - surrounding_centroids: Danh sách tọa độ trọng tâm của các đa giác xung quanh

    Trả về:
    - Danh sách các cặp (góc tương đối, tọa độ trọng tâm)

    Chức năng:
    1. Tính góc tuyệt đối giữa mỗi đa giác xung quanh và đa giác gốc
    2. Sắp xếp các góc theo thứ tự tăng dần
    3. Tính góc tương đối giữa các đa giác liên tiếp
    4. Trả về danh sách các cặp (góc tương đối, tọa độ trọng tâm)
    """
    angles = []
    for x, y in surrounding_centroids:
        dx = x - centroid[0]
        dy = y - centroid[1]
        angle = np.degrees(np.arctan2(dy, dx)) % 360
        angles.append((angle, (x, y)))

    angles.sort(key=lambda a: a[0])
    relative_angles = []
    for i in range(len(angles)):
        angle, coords = angles[i]
        prev_angle = angles[i-1][0]
        relative_angle = (angle - prev_angle) % 360
        relative_angles.append((relative_angle, coords))
    relative_angles.sort(key=lambda a: a[0])
    return relative_angles

def compute_angles(centroid, surrounding_centroids):
    """
    Tính góc giữa tâm và các tâm xung quanh với trục Ox.

    :param centroid: Tọa độ tâm chính
    :param surrounding_centroids: Danh sách tọa độ các tâm xung quanh
    :return: Danh sách các cặp (góc, tọa độ tâm) đã được sắp xếp theo góc
    """
    angles = []
    for x, y in surrounding_centroids:
        dx = x - centroid[0]
        dy = y - centroid[1]
        angle = np.degrees(np.arctan2(dy, dx))  # Tính góc bằng arctan2
        angle = 360 - (angle if angle >= 0 else 360 + angle)  # Chuyển đổi góc sang dạng 0-360 độ
        angles.append((angle, (x, y)))

    angles.sort(key=lambda a: a[0])  # Sắp xếp theo góc
    return angles

def extract_angle_features(angles):
    if not angles:  # Check if angles list is empty
        return np.array([])  # Return an empty array if no angles

    angle_values = np.array([angle for angle, _ in angles])
    # Calculate cos and sin for each angle
    cos_sin = np.array([ angle for angle in angle_values]).flatten()
    print(cos_sin)

    return cos_sin

def embed_polygon_cluster(polygons, angles):
    if not polygons:  # Check if polygons list is empty
        return np.array([])  # Return an empty array if no polygons
    angle_features = extract_angle_features(angles)
    return angle_features

def process_cluster(image):
    image = normalize_image_size(image)
    polygons = extract_polygons(image)

    if not polygons:
        return None, None

    # Tìm đa giác có diện tích nhỏ nhất
    min_area = float('inf')
    min_area_polygon = None
    for vertices, _ in polygons:
        area = polygon_area(vertices)
        if area < min_area:
            min_area = area
            min_area_polygon = vertices

    normalized_min_area_polygon = normalize_polygon_orientation(min_area_polygon)

    centroid = compute_centroid(normalized_min_area_polygon)
    surrounding_centroids = [compute_centroid(normalize_polygon_orientation(vertices)) for vertices, _ in polygons if vertices != min_area_polygon]
    relative_angles = compute_relative_angles(centroid, surrounding_centroids)

    embedded_vector = embed_polygon_cluster(polygons, relative_angles)
    return embedded_vector, image, polygons, centroid, relative_angles
