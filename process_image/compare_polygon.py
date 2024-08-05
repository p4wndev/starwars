import cv2
# import matplotlib as plt
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from process_image.process_polygon import extract_polygons, calculate_angles
import matplotlib.colors as mcolors
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

def validate_input(polygons1, polygons2, mode):
    if mode == "polygon_vs_polygon":
        if (len(polygons1)-1) > 1 or (len(polygons2)-1) > 1:
            return False, "Error: Both images must contain exactly one polygon for polygon vs polygon comparison."
    elif mode == "polygon_vs_map":
        if (len(polygons2)-1) != 1:
            return False, "Error: The polygon image must contain exactly one polygon for polygon vs map comparison."
        if(len(polygons1)-1) < 2:
            return False, "Error: The map image must contain atleast two polygons for polygon vs map comparison."
    elif mode == "cluster_vs_cluster":
        if (len(polygons1)-1) < 2 or (len(polygons2)-1) < 2:
            return False, "Error: Both images must contain at least two polygons for cluster vs cluster comparison."
    return True, ""

def plot_polygon_and_circle(image_cluster, vertices, centroid, similarity, color, is_highest_similarity):
    vertices = np.array(vertices, dtype=np.int32)
    centroid = tuple(map(int, centroid))

    # Tô màu polygon
    cv2.fillPoly(image_cluster, [vertices], color)
    
    # Vẽ đường viền polygon
    cv2.polylines(image_cluster, [vertices], isClosed=True, color=(255, 255, 255), thickness=2)

    # Tính toán vị trí để in phần trăm tương đồng
    text = f'{similarity:.2f}%'
    font_scale = 0.75  # Giảm kích thước chữ
    thickness = 1
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    
    # Tìm điểm cao nhất và thấp nhất của polygon
    min_y = min(vertices[:, 1])
    max_y = max(vertices[:, 1])
    
    # Đặt text phía trên polygon
    text_y = min_y - text_size[1] - 5
    
    # Tính toán vị trí x để text nằm giữa polygon theo chiều ngang
    min_x = min(vertices[:, 0])
    max_x = max(vertices[:, 0])
    text_x = (min_x + max_x - text_size[0]) // 2

    # Đảm bảo text không vượt ra ngoài hình ảnh
    text_x = max(0, min(text_x, image_cluster.shape[1] - text_size[0]))
    text_y = max(text_size[1], min(text_y, image_cluster.shape[0] - 5))

    # Vẽ nền cho text
    cv2.rectangle(image_cluster, (text_x - 2, text_y - text_size[1] - 2),
                  (text_x + text_size[0] + 2, text_y + 2), (255, 255, 255), -1)

    # Thêm text chính
    cv2.putText(image_cluster, text, (text_x, text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    if is_highest_similarity and is_highest_similarity > 1:
        radius = int(max(np.linalg.norm(vertices - centroid, axis=1)))
        cv2.circle(image_cluster, centroid, radius, (0, 0, 255), 3)

    return image_cluster

# def plot_polygon_and_image(vertices, centroid, min_angle_vertex, vertex_angles, sorted_angles, title):
#     fig, ax1 = plt.subplots(figsize=(10, 10))
#     x, y = zip(*vertices)
#     ax1.plot(x + (x[0],), y + (y[0],), color='blue', label='Polygon')
#     ax1.scatter(*centroid, color='red', label='Centroid')
    
#     # Extract angles from sorted_angles
#     extracted_angles = [angle for angle, vertex in sorted_angles]
    
#     # Ensure extracted_angles are numeric values
#     polygon_angles = [np.degrees(angle) for angle in extracted_angles]

#     for i, (angle, vertex) in enumerate(sorted_angles):
#         ax1.plot([centroid[0], vertex[0]], [centroid[1], vertex[1]], 'green', linestyle='--')
#         mid_x = (centroid[0] + vertex[0]) / 2
#         mid_y = (centroid[1] + vertex[1]) / 2
#         ax1.text(mid_x, mid_y, f"{vertex_angles[i]:.2f}°", fontsize=10, color='green')
    
#     for i, vertex in enumerate(vertices):
#         ax1.text(vertex[0], vertex[1], f"{polygon_angles[i]:.2f}°", fontsize=10, color='blue')
    
#     ax1.scatter(*min_angle_vertex, color='red', s=100, zorder=2)
#     ax1.plot([centroid[0], min_angle_vertex[0]], [centroid[1], min_angle_vertex[1]], color='red', linewidth=2, label='Reference Axis')
    
#     ax1.set_xlabel('X')
#     ax1.set_ylabel('Y')
#     ax1.set_title(title)
#     ax1.grid(True)
#     ax1.legend()
#     ax1.axis('equal')
    
#     plt.tight_layout()
#     return fig, polygon_angles, vertex_angles
def plot_polygon_and_image(vertices, centroid, min_angle_vertex, vertex_angles, sorted_angles, polygon_angles):
    fig, ax1 = plt.subplots(figsize=(15, 15))

    x, y = zip(*vertices)
    ax1.plot(x + (x[0],), y + (y[0],), color='blue', label='Polygon')
    ax1.scatter(*centroid, color='red', label='Centroid')

    for i, (angle, vertex) in enumerate(sorted_angles):
        ax1.plot([centroid[0], vertex[0]], [centroid[1], vertex[1]], 'green', linestyle='--')
        mid_x = (centroid[0] + vertex[0]) / 2
        mid_y = (centroid[1] + vertex[1]) / 2
        ax1.text(mid_x, mid_y, f"{vertex_angles[i]:.2f}°", fontsize=12, color='black', zorder=1)

    for i, vertex in enumerate(vertices):
        ax1.text(vertex[0], vertex[1], f"{polygon_angles[i]:.2f}°", fontsize=12, color='blue',zorder=1)

    ax1.scatter(*min_angle_vertex, color='red', s=100, zorder=0)
    ax1.plot([centroid[0], min_angle_vertex[0]], [centroid[1], min_angle_vertex[1]], color='red', linewidth=2, label='Reference Axis')

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('Polygon with Centroid, Angles, and Reference Axis')
    ax1.grid(True)
    ax1.legend()
    ax1.axis('equal')
    return fig, polygon_angles, vertex_angles

def get_color_for_similarity(similarity, max_similarity):
    normalized_similarity = similarity / (max_similarity+0.0000001)
    colors = ['red', 'orange', 'yellow', 'green', 'blue']
    cmap = mcolors.LinearSegmentedColormap.from_list("custom", colors)
    rgb_color = cmap(1-normalized_similarity)[:3]  # Đảo ngược thang đo để đỏ là cao nhất
    return tuple(int(x * 255) for x in rgb_color)

def plot_one_vs_one(image_cluster, vertices, centroid, similarity, color):
    return plot_polygon_and_circle(image_cluster, vertices, centroid, similarity, color, False)

#------------------------------------------DTW-------------------------------------------
def normalize_vector(vector):
    norm = np.linalg.norm(vector)
    return vector / norm if norm != 0 else vector

def apply_weight(vector, weight):
    return vector * weight
# Tính trọng số dựa trên sự khác biệt kích thước của vector
def compute_size_difference_weight(len_v1, len_v2):
    size_diff = abs(len_v1 - len_v2)
    weight = np.exp(size_diff)  # Trọng số exponential dựa trên sự khác biệt kích thước
    return weight

# Tính khoảng cách DTW giữa các cặp vector
def normalize_and_compute_dtw_similarity(v1, v2):
    len_v1 = len(v1)
    len_v2 = len(v2)
    size_weight = compute_size_difference_weight(len_v1, len_v2)

    # Chuẩn hóa các vector
    v1_normalized = normalize_vector(v1)
    v2_normalized = normalize_vector(v2)

    # Áp dụng trọng số
    v1_weighted = apply_weight(v1_normalized, size_weight)
    v2_weighted = apply_weight(v2_normalized, size_weight)

    # Tính khoảng cách DTW
    distance, _ = fastdtw(v1_weighted.reshape(-1, 1), v2_weighted.reshape(-1, 1), dist=euclidean)

    # Tính max_distance trên cơ sở chuẩn hóa
    max_distance = np.linalg.norm(np.ones_like(v1_weighted) - np.zeros_like(v1_weighted))
    percent_similarity = max(0, (1 - (distance / max_distance)) * 100)

    return percent_similarity, v1_weighted, v2_weighted

#----------------------------------------Subarrays-------------------------------------------

def find_largest_consecutive_matches(A, B, k1=5, k2=3):
    """
    Tìm nhóm khớp liên tiếp lớn nhất giữa hai mảng dựa trên ngưỡng k.
    
    Parameters:
    A (list): Mảng đầu tiên
    B (list): Mảng thứ hai
    k (int): Ngưỡng khớp
    
    Returns:
    tuple: Nhóm khớp liên tiếp lớn nhất dưới dạng (nhóm A, nhóm B)
    """
    def subarrays(arr):
        """Tạo ra tất cả các nhóm liên tiếp từ mảng arr."""
        result = []
        for start in range(len(arr)):
            for end in range(start + 1, len(arr) + 1):
                result.append(arr[start:end])
        return result

    def is_match(subA, subB):
        """Kiểm tra xem hai nhóm có khớp hay không dựa trên ngưỡng k."""
        if len(subA) != len(subB):
            return False
        return all(abs(a - b) < k1 for a, b in zip(subA, subB))
    
    subarrays_A = subarrays(A)
    subarrays_B = subarrays(B)
    
    largest_match = None
    largest_match_length = 0

    for subA in subarrays_A:
        if len(subA) >= k2:  # Chỉ kiểm tra nhóm có độ dài lớn hơn hoặc bằng k2
            for subB in subarrays_B:
                if len(subB) == len(subA) and is_match(subA, subB):
                    if len(subA) > largest_match_length:
                        largest_match = (subA, subB)
                        largest_match_length = len(subA)
    
    return largest_match_length

def calculate_percentage_match(A, B, k1, k2):
    largest_match_length = find_largest_consecutive_matches(A, B, k1, k2)
    
    if largest_match_length == 0:
        return 0.0
    
    total_length_A = len(A)
    total_length_B = len(B)
    
    # Tính tỷ lệ phần trăm khớp dựa trên mảng có độ dài nhỏ hơn
    percentage_match = (largest_match_length / max(total_length_A, total_length_B)) * 100
    return percentage_match

#----------------------------------------------------------------------
def process_image_and_find_similar_polygons_2(image1, image2, top_n=1, progress_callback=None, is_one_vs_one=False, compare_mode="polygon_vs_polygon", option="Subarrays", k1=5, k2=3):
    if progress_callback:
        progress_callback(1)  # Update progress to 1%
    # Extract polygons
    polygons1 = extract_polygons(image1)
    if progress_callback:
        progress_callback(20)  # Update progress to 20%
        
    polygons2 = extract_polygons(image2)
    if progress_callback:
        progress_callback(40)  # Update progress to 40%
        
    # Validate input
    is_valid, error_message = validate_input(polygons1, polygons2, compare_mode)
    if not is_valid:
        return None, 0, [], error_message
    
    combined_angles2_list = []
    for i, (vertices, cropped_image) in enumerate(polygons2):
        centroid2, min_angle_vertex2, sorted_vertices2, sorted_angle_values2, sorted_angles2, polygon_angles2 = calculate_angles(vertices)
        combined_angles2 = np.concatenate([sorted(sorted_angle_values2), sorted(polygon_angles2)])
        combined_angles2_list.append(combined_angles2)

    similarity_scores = []
    img1 = image1.copy()
    for i, (vertices, cropped_image) in enumerate(polygons1):
        centroid1, min_angle_vertex1, sorted_vertices1, sorted_angle_values1, sorted_angles1, polygon_angles1 = calculate_angles(vertices)
        combined_angles1 = np.concatenate([sorted(sorted_angle_values1), sorted(polygon_angles1)])
        
        if option == "Subarrays":
            for combined_angles2 in combined_angles2_list:
                similarity = calculate_percentage_match(combined_angles1, combined_angles2, k1, k2)
                similarity_scores.append((similarity, vertices, cropped_image, centroid1, min_angle_vertex1, sorted_angle_values1, sorted_angles1))
                
    similarity_scores = sorted(similarity_scores, key=lambda x: x[0], reverse=True)[:top_n]
    highest_similarity = similarity_scores[0][0] if similarity_scores else 0
    
    for i, (similarity, vertices, cropped_image, centroid, min_angle_vertex, sorted_angle_values, sorted_angles) in enumerate(similarity_scores):
        color = get_color_for_similarity(similarity, highest_similarity)
        is_highest_similarity = (i == 0)
        img1 = plot_polygon_and_circle(img1, vertices, centroid, similarity, color, is_highest_similarity)
    
    result_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    
    if progress_callback:
        progress_callback(100)  # Update progress to 100%
    
    return result_rgb, highest_similarity, similarity_scores, error_message

def process_image_and_find_similar_polygons(image1, image2, top_n=1, progress_callback=None, is_one_vs_one=False, compare_mode="polygon_vs_polygon", option="Subarrays", k1=5, k2=3):
    if progress_callback:
        progress_callback(1)  # Update progress to 1%
    # Extract polygons
    polygons1 = extract_polygons(image1)
    # print("\n",len(polygons1))
    if progress_callback:
        progress_callback(20)  # Update progress to 20%
        
    polygons2 = extract_polygons(image2)
    # print(len(polygons2))
    if progress_callback:
        progress_callback(40)  # Update progress to 40%
        # Validate input
    is_valid, error_message = validate_input(polygons1, polygons2, compare_mode)
    if not is_valid:
        return None, 0, [], error_message
    
    for i, (vertices, cropped_image) in enumerate(polygons2):
        centroid2, min_angle_vertex2, sorted_vertices2, sorted_angle_values2, sorted_angles2, polygon_angles2 = calculate_angles(vertices)
        combined_angles2 = np.concatenate([sorted(sorted_angle_values2), sorted(polygon_angles2)])

    similarity_scores = []
    img1 = image1.copy()
    for i, (vertices, cropped_image) in enumerate(polygons1):
        centroid1, min_angle_vertex1, sorted_vertices1, sorted_angle_values1, sorted_angles1, polygon_angles1 = calculate_angles(vertices)
        combined_angles1 = np.concatenate([sorted(sorted_angle_values1), sorted(polygon_angles1)])
        if option == "Subarrays":
            similarity = calculate_percentage_match(combined_angles1, combined_angles2, k1, k2)
            similarity_scores.append((similarity, vertices, cropped_image, centroid1, min_angle_vertex1, sorted_angle_values1, sorted_angles1))
            similarity_scores = sorted(similarity_scores, key=lambda x: x[0], reverse=True)[:top_n]

            highest_similarity = similarity_scores[0][0] if similarity_scores else 0
            
            for i, (similarity, vertices, cropped_image, centroid, min_angle_vertex, sorted_angle_values, sorted_angles) in enumerate(similarity_scores):
                color = get_color_for_similarity(similarity, highest_similarity)
                if is_one_vs_one:
                    plot_image_1 = image1.copy()
                    plot_image_1 = plot_one_vs_one(plot_image_1, vertices, centroid, similarity, color)
                else:
                    is_highest_similarity = (i == 0)  # Đánh dấu polygon có độ tương đồng cao nhất
                    plot_image_1 = image1
                    plot_image_1 = plot_polygon_and_circle(plot_image_1, vertices, centroid, similarity, color, is_highest_similarity)
            result_rgb = cv2.cvtColor(plot_image_1, cv2.COLOR_BGR2RGB)    
        
        else:
            similarity, w1, w2 = normalize_and_compute_dtw_similarity(combined_angles1, combined_angles2)
            similarity_scores.append((similarity, vertices, cropped_image, centroid1, min_angle_vertex1, sorted_angle_values1, sorted_angles1, w1, w2))
        
            if progress_callback:
                progress_callback(60)  # Update progress to 60%

            similarity_scores = sorted(similarity_scores, key=lambda x: x[0], reverse=True)[:top_n]

            highest_similarity = similarity_scores[0][0] if similarity_scores else 0
            
            for i, (similarity, vertices, cropped_image, centroid, min_angle_vertex, sorted_angle_values, sorted_angles, w1, w2) in enumerate(similarity_scores):
                color = get_color_for_similarity(similarity, highest_similarity)
                if is_one_vs_one:
                    plot_image_2 = img1.copy()
                    plot_image_2 = plot_one_vs_one(plot_image_2, vertices, centroid, similarity, color)
                else:
                    is_highest_similarity = (i == 0)  # Đánh dấu polygon có độ tương đồng cao nhất
                    plot_image_2 = img1
                    plot_image_2 = plot_polygon_and_circle(plot_image_2, vertices, centroid, similarity, color, is_highest_similarity)
            result_rgb = cv2.cvtColor(plot_image_2, cv2.COLOR_BGR2RGB)    
    if progress_callback:
        progress_callback(80)  # Update progress to 80%
        
    # Convert the result image to RGB format
    
    
    
    if progress_callback:
        progress_callback(100)  # Update progress to 100%

    return result_rgb, highest_similarity, similarity_scores, error_message

def compare_maps(input_image, image2, similarity_threshold=80.0, progress_callback=None):
    if progress_callback:
        progress_callback(1)  # Update progress to 1%
    
    # Extract polygons from the input image
    polygons_input = extract_polygons(input_image)
    input_polygon_count = len(polygons_input)
    
    if progress_callback:
        progress_callback(20)  # Update progress to 20%
    
    # Extract polygons from the current image in the set
    polygons_image = extract_polygons(image2)
    total_polygons = len(polygons_image)
    
    if progress_callback:
        progress_callback(40)  # Update progress to 40%
    
    matched_polygons = 0
    used_polygons_input = set()
    used_polygons_image = set()
    
    for i, (vertices_input, _) in enumerate(polygons_input):
        if i in used_polygons_input:
            continue
        _, _, _, sorted_angle_values_input, _, _ = calculate_angles(vertices_input)
        
        for j, (vertices_image, _) in enumerate(polygons_image):
            if j in used_polygons_image:
                continue
            centroid_image, _, _, sorted_angle_values_image, _, _ = calculate_angles(vertices_image)
            similarity, w1, w2 = normalize_and_compute_dtw_similarity(sorted_angle_values_input, sorted_angle_values_image)
            if similarity >= similarity_threshold:
                matched_polygons += 1
                used_polygons_input.add(i)
                used_polygons_image.add(j)
                color = get_color_for_similarity(similarity, 100.0)
                image2 = plot_polygon_and_circle(image2, vertices_image, centroid_image, similarity, color, False)
                break
    
    if progress_callback:
        progress_callback(80)  # Update progress to 80%
    
    # Convert the result image to RGB format
    result_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    
    if progress_callback:
        progress_callback(100)  # Update progress to 100%

    return result_rgb, matched_polygons, total_polygons, input_polygon_count, None