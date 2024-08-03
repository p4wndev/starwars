import numpy as np
import cv2
from process_image.process_polygon import compute_centroid, extract_polygons, polygon_area
import requests
from sklearn.neighbors import NearestNeighbors

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

# __________________________________________________________________________________
def calculate_angles_knn(vertices):
    def polygon_exterior_angles(vertices):
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

            v1 = p1 - p0
            v2 = p2 - p1

            angle = angle_between(v1, v2)
            exterior_angle = (360 - angle) % 360
            angles.append(exterior_angle)

        return angles

    polygon_angles = polygon_exterior_angles(vertices)
    return polygon_angles

def classify_polygon(polygon):
    """Classify polygon based on the number of concave angles (exterior angle < 180 degrees)"""
    concave_corners = 0

    angles = calculate_angles_knn(polygon)
    for angle in angles:
        if angle < 180:
            concave_corners += 1

    if concave_corners <= 3:
        return "normal_polygon"
    elif 4 <= concave_corners <= 6:
        return "special_polygon"
    else:
        return "superhero_polygon"

def find_superhero_polygon(cluster):
    """Find the most special polygon in the cluster"""
    classifications = [classify_polygon(poly) for poly in cluster]
    if "superhero_polygon" in classifications:
        return classifications.index("superhero_polygon")
    elif "special_polygon" in classifications:
        return classifications.index("special_polygon")
    else:
        return 0  # Return the first polygon if no special polygon is found

def extract_angle_features_knn(angles):
    """
    Trích xuất các đặc trưng góc từ danh sách góc.

    :param angles: Danh sách các góc
    :return: Mảng numpy chứa các đặc trưng góc
    """
    if not angles:  # Check if angles list is empty
        return np.array([])  # Return an empty array if no angles

    if isinstance(angles[0], (list, tuple)):
        angle_values = np.array([angle for angle, _ in angles])
    else:
        angle_values = np.array(angles)

    angles_features = np.array([angle for angle in angle_values]).flatten()
    return angles_features

def embed_polygon_cluster_knn(polygons, angles):
    """
    Tạo vector đặc trưng từ cụm các polygon và các góc.

    :param polygons: Danh sách các polygon
    :param angles: Danh sách các góc
    :return: Mảng numpy chứa các đặc trưng
    """
    if not polygons:  # Check if polygons list is empty
        return np.array([])  # Return an empty array if no polygons

    angle_features = extract_angle_features_knn(angles)
    return angle_features

def find_clusters(polygons, centroids, k=4):
    """
    Tìm các cụm của các đa giác trong ảnh.

    :param polygons: Danh sách các đa giác
    :param centroids: Danh sách các trọng tâm của các đa giác
    :param k: Số lượng hàng xóm gần nhất để tìm cụm (mặc định là 4)
    :return: Danh sách các cụm với các thông tin liên quan
    """
    # Lọc ra các centroid và polygon đặc biệt
    special_polygons = []
    special_centroids = []

    for i, polygon in enumerate(polygons):
        if classify_polygon(polygon) != "normal_polygon":
            special_polygons.append(polygon)
            special_centroids.append(centroids[i])

    if not special_centroids:
        return []

    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(special_centroids)

    clusters = []
    for i, centroid in enumerate(special_centroids):
        _, indices = knn.kneighbors([centroid])
        cluster = [special_polygons[idx] for idx in indices[0]]
        superhero_index = find_superhero_polygon(cluster)
        angles = calculate_angles_knn(polygons[i])
        embedded_vector = embed_polygon_cluster_knn(cluster, angles)
        clusters.append((i, superhero_index, cluster, angles, embedded_vector))

    return clusters

def extract_polygons_knn(image, min_vertices=3, max_area=100000):
    """
    Trích xuất các đa giác từ hình ảnh.

    :param image: Hình ảnh đầu vào (numpy array)
    :param min_vertices: Số đỉnh tối thiểu của đa giác (mặc định là 3)
    :param max_area: Diện tích tối đa của đa giác được xem xét (mặc định là 100000)
    :return: Danh sách các đỉnh của các đa giác
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 200, 255, 0)  # Nhị phân hóa ảnh
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # Tìm contour

    polygons = []
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)  # Xấp xỉ contour thành đa giác
        if len(approx) >= min_vertices:
            flattened_vertices = [tuple(coord[0]) for coord in approx]
            area = polygon_area(flattened_vertices)
            if area < max_area:
                polygons.append(flattened_vertices)
    return polygons