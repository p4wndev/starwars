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
        if (len(polygons2)-1) > 1:
            print(len(polygons2))
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
def plot_polygon_and_image(vertices, centroid, vertex_angles, sorted_angles, polygon_angles):
    fig, ax1 = plt.subplots(figsize=(15, 15))

    x, y = zip(*vertices)
    ax1.plot(x + (x[0],), y + (y[0],), color='blue', label='Đa giác')
    ax1.scatter(*centroid, color='red', label='Trọng tâm')

    for i in range(len(sorted_angles)):
        angle1, vertex1 = sorted_angles[i]
        angle2, vertex2 = sorted_angles[(i+1) % len(sorted_angles)]

        # Vẽ đường thẳng từ trọng tâm đến đỉnh
        ax1.plot([centroid[0], vertex1[0]], [centroid[1], vertex1[1]], 'green', linestyle='--')
        ax1.plot([centroid[0], vertex2[0]], [centroid[1], vertex2[1]], 'green', linestyle='--')

        # Tính toán khoảng cách đến các đỉnh
        dist1 = np.linalg.norm(np.array(vertex1) - np.array(centroid))
        dist2 = np.linalg.norm(np.array(vertex2) - np.array(centroid))
        avg_dist = (dist1 + dist2) / 2

        # Lấy kích thước góc
        angle_size = vertex_angles[(i+1) % len(vertex_angles)]

        # Điều chỉnh bán kính dựa trên kích thước góc
        if angle_size < 30:
            radius = avg_dist * 0.6  # Góc nhỏ, vẽ xa hơn
        elif angle_size < 60:
            radius = avg_dist * 0.5  # Góc trung bình
        else:
            radius = avg_dist * 0.4  # Góc lớn, vẽ gần hơn

        # Tính toán vị trí của góc
        center_angle = (angle1 + angle2) / 2

        # Hiển thị số đo góc tại vị trí thích hợp
        if i == len(sorted_angles)-1:
            # Đối với góc đầu và cuối, đặt text ở vị trí đối diện
            arc_center = (centroid[0] - radius * 1.1 * np.cos(center_angle),
                          centroid[1] - radius * 1.1 * np.sin(center_angle))
        else:
            arc_center = (centroid[0] + radius * 1.1 * np.cos(center_angle),
                          centroid[1] + radius * 1.1 * np.sin(center_angle))

        ax1.text(arc_center[0], arc_center[1], f"{angle_size:.2f}°",
                 fontsize=10, color='red', ha='center', va='center')

    # Hiển thị góc tại các đỉnh đa giác
    for i, vertex in enumerate(vertices):
        ax1.text(vertex[0] + 0.05, vertex[1] + 0.05, f"{polygon_angles[i]:.2f}°",
                 fontsize=12, color='blue', zorder=1)

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('Polygon with Centroid and Angles')
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

# def find_largest_consecutive_matches(A, B, k1=5, k2=3):
#     """
#     Tìm nhóm khớp liên tiếp lớn nhất giữa hai mảng dựa trên ngưỡng k.

#     Parameters:
#     A (list): Mảng đầu tiên
#     B (list): Mảng thứ hai
#     k (int): Ngưỡng khớp

#     Returns:
#     tuple: Nhóm khớp liên tiếp lớn nhất dưới dạng (nhóm A, nhóm B)
#     """
#     # def subarrays(arr):
#     #     """Tạo ra tất cả các nhóm liên tiếp từ mảng arr."""
#     #     result = []
#     #     for start in range(len(arr)):
#     #         for end in range(start + 1, len(arr) + 1):
#     #             result.append(arr[start:end])
#     #     return result

#     def subarrays(arr):
#         """Tạo ra tất cả các nhóm liên tiếp từ mảng arr, tính cả vòng."""
#         n = len(arr)
#         result = []
#         for start in range(n):
#             for length in range(1, n + 1):
#                 subarray = [arr[(start + i) % n] for i in range(length)] #VÒNG nè
#                 result.append(subarray)
#         return result

#     def is_match(subA, subB):
#         """Kiểm tra xem hai nhóm có khớp hay không dựa trên ngưỡng k."""
#         if len(subA) != len(subB):
#             return False
#         return all(abs(a - b) < k1 for a, b in zip(subA, subB))

#     subarrays_A = subarrays(A)
#     subarrays_B = subarrays(B)

#     largest_match = None
#     largest_match_length = 0

#     for subA in subarrays_A:
#         if len(subA) >= k2:  # Chỉ kiểm tra nhóm có độ dài lớn hơn hoặc bằng k2
#             for subB in subarrays_B:
#                 if len(subB) == len(subA) and is_match(subA, subB):
#                     if len(subA) > largest_match_length:
#                         largest_match = (subA, subB)
#                         print(largest_match)
#                         largest_match_length = len(subA)

#     return largest_match_length

# def calculate_percentage_match(A, B, k1, k2):
#     largest_match_length = find_largest_consecutive_matches(A, B, k1, k2)

#     if largest_match_length == 0:
#         return 0.0

#     total_length_A = len(A)
#     total_length_B = len(B)

#     # Tính tỷ lệ phần trăm khớp dựa trên mảng có độ dài nhỏ hơn
#     percentage_match = (largest_match_length / max(total_length_A, total_length_B)) * 100
#     return percentage_match

def find_largest_consecutive_matches(A, B, k1=5, k2=3):
    """
    Tìm nhóm khớp liên tiếp lớn nhất giữa hai mảng dựa trên ngưỡng k.

    Parameters:
    A (list): Mảng đầu tiên
    B (list): Mảng thứ hai
    k1 (int): Ngưỡng khớp cho phép

    Returns:
    float: Tổng điểm khớp lớn nhất
    """
    def subarrays(arr):
        """Tạo ra tất cả các nhóm liên tiếp từ mảng arr, tính cả vòng."""
        n = len(arr)
        result = []
        for start in range(n):
            for length in range(1, n + 1):
                subarray = [arr[(start + i) % n] for i in range(length)]  # Vòng nè
                result.append(subarray)
        return result

    def compute_matching_score(subA, subB):
        """Tính điểm khớp giữa hai nhóm"""
        total_score = 0
        for a, b in zip(subA, subB):
            if abs(a - b) <= k1:
                score = 1
            elif abs(a - b) > k1 and b > a and (a / b) >= 0.9:
                score = a / b
            elif abs(a - b) > k1 and b < a and (b / a) >= 0.9:
                score = b / a
            else:
                score = 0
            total_score += score
        return total_score

    subarrays_A = subarrays(A)
    subarrays_B = subarrays(B)

    largest_match = None
    max_total_score = 0

    for subA in subarrays_A:
        if len(subA) >= k2:  # Chỉ kiểm tra nhóm có độ dài lớn hơn hoặc bằng k2
            for subB in subarrays_B:
                if len(subB) == len(subA):
                    total_score = compute_matching_score(subA, subB)
                    if total_score > max_total_score:
                        largest_match = (subA, subB)
                        max_total_score = total_score

    return max_total_score

def calculate_percentage_match(A, B, k1, k2):
    max_total_score = find_largest_consecutive_matches(A, B, k1, k2)

    if max_total_score == 0:
        return 0.0

    total_length_A = len(A)
    total_length_B = len(B)

    # Tính tỷ lệ phần trăm khớp dựa trên mảng có độ dài lớn hơn
    percentage_match = (max_total_score / max(total_length_A, total_length_B)) * 100
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
        centroid2, sorted_vertices2, vertex_angles2, sorted_angles2, polygon_angles2 = calculate_angles(vertices)
        combined_angles2 = np.concatenate([vertex_angles2, polygon_angles2])
        combined_angles2_list.append(combined_angles2)

    similarity_scores = []
    img1 = image1.copy()
    for i, (vertices, cropped_image) in enumerate(polygons1):
        centroid1, sorted_vertices1, vertex_angles1, sorted_angles1, polygon_angles1 = calculate_angles(vertices)
        combined_angles1 = np.concatenate([vertex_angles1, polygon_angles1])


        if option == "Subarrays":
            if len(combined_angles2_list) == 2:
                for combined_angles2 in combined_angles2_list[1:]:  # Skip the first element
                    similarity = calculate_percentage_match(combined_angles1, combined_angles2, k1, k2)
                    similarity_scores.append((similarity, vertices, cropped_image, centroid1, sorted_vertices1, vertex_angles1, sorted_angles1))
            else:
                for combined_angles2 in combined_angles2_list:
                    similarity = calculate_percentage_match(combined_angles1, combined_angles2, k1, k2)
                    similarity_scores.append((similarity, vertices, cropped_image, centroid1, sorted_vertices1, vertex_angles1, sorted_angles1))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[0], reverse=True)[:top_n]
    highest_similarity = similarity_scores[0][0] if similarity_scores else 0

    for i, (similarity, vertices, cropped_image, centroid, sorted_vertices, vertex_angles, sorted_angles) in enumerate(similarity_scores):
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
        centroid2, sorted_vertices2, vertex_angles2, sorted_angles2, polygon_angles2 = calculate_angles(vertices)
        combined_angles2 = np.concatenate([vertex_angles2, polygon_angles2])
        # print(combined_angles2)
    similarity_scores = []
    img1 = image1.copy()
    for i, (vertices, cropped_image) in enumerate(polygons1):
        centroid1, sorted_vertices1, vertex_angles1, sorted_angles1, polygon_angles1 = calculate_angles(vertices)
        combined_angles1 = np.concatenate([vertex_angles1, polygon_angles1])
        # print(combined_angles1)
        if option == "Subarrays":
            similarity_1 = calculate_percentage_match(vertex_angles1, vertex_angles2, k1, k2)
            similarity_2 = calculate_percentage_match(polygon_angles1, polygon_angles2, k1, k2)
            similarity = (similarity_1 * 0.7 + similarity_2 * 0.3)
            similarity_scores.append((similarity, vertices, cropped_image, centroid1, sorted_vertices1, vertex_angles1, sorted_angles1))
            similarity_scores = sorted(similarity_scores, key=lambda x: x[0], reverse=True)[:top_n]

            highest_similarity = similarity_scores[0][0] if similarity_scores else 0

            for i, (similarity, vertices, cropped_image, centroid, sorted_vertices, vertex_angles, sorted_angles) in enumerate(similarity_scores):
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
            similarity_scores.append((similarity, vertices, cropped_image, centroid1, sorted_vertices1, vertex_angles1, sorted_angles1, w1, w2))

            if progress_callback:
                progress_callback(60)  # Update progress to 60%

            similarity_scores = sorted(similarity_scores, key=lambda x: x[0], reverse=True)[:top_n]

            highest_similarity = similarity_scores[0][0] if similarity_scores else 0

            for i, (similarity, vertices, cropped_image, centroid, sorted_vertices, vertex_angles, sorted_angles, w1, w2) in enumerate(similarity_scores):
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
        _, _, vertex_angles_input, _, _ = calculate_angles(vertices_input)

        for j, (vertices_image, _) in enumerate(polygons_image):
            if j in used_polygons_image:
                continue
            centroid_image, _, vertex_angles_image, _, _ = calculate_angles(vertices_image)
            similarity= calculate_percentage_match(vertex_angles_input, vertex_angles_image, 15, 5)
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

def plot_polygon_and_circle_2(image_cluster, results):
    if not isinstance(image_cluster, np.ndarray):
        image_cluster = np.array(image_cluster)

    # Tính toán kích thước phù hợp cho chữ và độ dày đường vẽ dựa trên kích thước ảnh
    image_height, image_width = image_cluster.shape[:2]
    font_scale = min(image_width, image_height) / 1000
    thickness_scale = min(image_width, image_height) / 500  # Hệ số mới cho độ dày

    for res in results:
        _, _, similarity, vertices, centroid = res
        print(f"Plotting similarity: {similarity}")

        vertices = np.array(vertices, dtype=np.int32)
        centroid = tuple(map(int, centroid))

        # Tính toán độ dày phù hợp
        polygon_thickness = max(1, int(thickness_scale) )
        circle_thickness = max(2, int(thickness_scale * 2))
        text_thickness = max(1, int(thickness_scale))

        # Vẽ đa giác
        cv2.polylines(image_cluster, [vertices], isClosed=True, color=(0, 255, 0), thickness=polygon_thickness)

        # Vẽ vòng tròn quanh tâm
        radius = int(max(np.linalg.norm(vertices - centroid, axis=1)))
        radius = 2*radius
        cv2.circle(image_cluster, centroid, radius, color=(0, 0, 255), thickness=circle_thickness)

        # Thêm text độ tương đồng phía trên vòng tròn
        text = f'{similarity:.2f}%'
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
        text_x = centroid[0] - text_size[0] // 2
        text_y = centroid[1] - radius - 10
        cv2.putText(image_cluster, text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), text_thickness, cv2.LINE_AA)
    result_img_hsv = cv2.cvtColor(image_cluster, cv2.COLOR_BGR2HSV)
    # Convert back to BGR for display
    result_img_enhanced = cv2.cvtColor(result_img_hsv, cv2.COLOR_HSV2BGR)
    return result_img_enhanced

def compare_polygon(polygon1, polygon2, k1=15, k2=5):
    # Tính toán các góc cho polygon1
    centroid1, sorted_vertices1, vertex_angles1, sorted_angles1, polygon_angles1 = calculate_angles(polygon1)
    # Tính toán các góc cho polygon2
    centroid2, sorted_vertices2, vertex_angles2, sorted_angles2, polygon_angles2 = calculate_angles(polygon2)
    # Tính độ tương đồng
    similarity1 = calculate_percentage_match(polygon_angles1, polygon_angles2, k1, k2)
    similarity2 = calculate_percentage_match(vertex_angles1, vertex_angles2, k1, k2)
    similarity = (similarity1 * 0.5 + similarity2 * 0.5)
    return similarity, sorted_vertices2, centroid2

def find_polygon(image2, polygons1, polygons2, top_n=5, stop_threshold=95, k1=15, k2=3):
    # Debugging: print number of polygons found
    print(f"Number of polygons in image1: {len(polygons1)}")
    print(f"Number of polygons in image2: {len(polygons2)}")

    result = []
    i = 0
    for polygon1 in polygons1:
        if len(polygon1[0]) <= 4:
            continue
        for polygon2 in polygons2:
            similarity, vertices, centroid = compare_polygon(polygon1[0], polygon2[0], k1=k1, k2=k2)
            print(f"Similarity {i}: {similarity}")
            i += 1
            # Append to results if conditions are met
            result.append((polygon1, polygon2, similarity, vertices, centroid))

            # Sort and keep only the top_n best results
            result = sorted(result, key=lambda x: x[2], reverse=True)[:top_n]

            # Check if the stop threshold is reached
            if result and result[0][2] >= stop_threshold:
                print(f"Stopping early as similarity reached {result[0][2]} which is above threshold {stop_threshold}")
                break
        else:
            continue
        break

    result_image = image2.copy()

    # Plot all top_n results on the image
    result_image = plot_polygon_and_circle_2(result_image, result)

    return result, result_image
