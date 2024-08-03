import matplotlib.pyplot as plt
import cv2
from process_image.process_cluster import process_cluster, embed_polygon_cluster_knn, extract_angle_features_knn, extract_polygons_knn, find_clusters
from process_image.compare_polygon import normalize_and_compute_dtw_similarity, validate_input
from process_image.process_polygon import compute_centroid
import os
import numpy as np

def display_results(image, polygons, centroid, relative_angles):
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    for vertices, _ in polygons:
        poly = plt.Polygon(vertices, fill=None, edgecolor='r')
        ax.add_patch(poly)

    ax.plot(centroid[0], centroid[1], 'bo')

    for angle, (x, y) in relative_angles:
        ax.plot([centroid[0], x], [centroid[1], y], 'grey')
        ax.plot(x, y, 'go', markersize=2)
        ax.text(x, y, f"{angle:.2f}°", color='black', fontsize=10)

    ax.axhline(y=centroid[1], color='blue', linestyle='--')
    ax.axvline(x=centroid[0], color='blue', linestyle='--')
    return fig, relative_angles

def process_image_and_find_similar_cluster(image1, image2_list, top_n=5, progress_callback=None):
    if progress_callback:
        progress_callback(0.01)  # Start with 1% progress

    # Process first cluster image
    cluster1, image_cluster1, _, _, _ = process_cluster(image1)
    
    if cluster1 is None:
        return None, 0, [], "The system is unable to process the input image."

    if progress_callback:
        progress_callback(0.2)  # 20% progress
        
    similarity_scores = []
    
    total_images = len(image2_list)
    for i, image2 in enumerate(image2_list):
        cluster2, image_cluster2, _, _, _ = process_cluster(image2)
        if cluster2 is not None:
            similarity, w1, w2 = normalize_and_compute_dtw_similarity(cluster1, cluster2)
            similarity_scores.append((similarity, f"Image {i+1}", image_cluster2,w1,w2))
        
        if progress_callback:
            progress = 0.2 + (0.6 * (i + 1) / total_images)
            progress_callback(progress)

    similarity_scores = sorted(similarity_scores, key=lambda x: x[0], reverse=True)[:top_n]

    highest_similarity = similarity_scores[0][0] if similarity_scores else 0
    
    if progress_callback:
        progress_callback(1.0)  # 100% progress

    return image_cluster1, highest_similarity, similarity_scores, None

# ___________________________________________________________________________________

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors

def compute_centroids(polygons):
    """Compute centroids of polygons"""
    centroids = []
    for poly in polygons:
        M = cv2.moments(np.array(poly))
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centroids.append([cX, cY])
    return np.array(centroids)

def compare_clusters_knn(cluster1, cluster2):
    """
    So sánh hai cụm polygon và trả về mức độ tương đồng giữa chúng.

    :param cluster1: Cụm polygon đầu tiên (gồm các polygon và các đặc trưng góc)
    :param cluster2: Cụm polygon thứ hai (gồm các polygon và các đặc trưng góc)
    :return: Mức độ tương đồng giữa hai cụm (phần trăm)
    """
    _, _, _, angles1, embedded_vector1 = cluster1
    _, _, _, angles2, embedded_vector2 = cluster2

    # Tính mức độ tương đồng giữa các đặc trưng góc của hai cụm
    similarity, _, _ = normalize_and_compute_dtw_similarity(embedded_vector1, embedded_vector2)
    return similarity

def compare_images_knn(image1, image2, k):
    """
    So sánh các cụm polygon trong hai ảnh và trả về các cụm giống nhau và tổng số cụm trong ảnh 1 và ảnh 2.
    Mỗi cụm trong ảnh 1 chỉ được ghép cặp với một cụm duy nhất trong ảnh 2.

    :param image1: Ảnh đầu tiên (numpy array)
    :param image2: Ảnh thứ hai (numpy array)
    :param k: Số lượng cụm
    :return: Danh sách các cặp cụm giống nhau giữa ảnh 1 và ảnh 2, tổng số cụm trong ảnh 1 và ảnh 2
    """
    # Trích xuất các cụm polygon từ hai ảnh
    polygons1 = extract_polygons_knn(image1)
    polygons2 = extract_polygons_knn(image2)

    if not polygons1 or not polygons2:
        return [], 0, 0

    centroids1 = compute_centroids(polygons1)
    centroids2 = compute_centroids(polygons2)

    clusters1 = find_clusters(polygons1, centroids1, k)
    clusters2 = find_clusters(polygons2, centroids2, k)

    matching_clusters = []
    used_cluster_indices = []

    for i, cluster1 in enumerate(clusters1):
        best_match_index = -1
        best_similarity = 0
        for j, cluster2 in enumerate(clusters2):
            if j not in used_cluster_indices:
                similarity = compare_clusters_knn(cluster1, cluster2)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_index = j

        if best_match_index != -1 and best_similarity > 90:  # Ngưỡng mức độ tương đồng để xác định hai cụm là giống nhau
            matching_clusters.append((cluster1, clusters2[best_match_index]))
            used_cluster_indices.append(best_match_index)

    return matching_clusters, len(clusters1), len(clusters2)

def plot_matching_clusters_knn(image1, image2, matching_clusters):
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # Tạo nền trắng
    # white_background1 = np.ones_like(image1) * 255
    # white_background2 = np.ones_like(image2) * 255

    # Vẽ nền trắng
    axes[0].imshow(image1)
    axes[0].set_title("Image 1")
    axes[1].imshow(image2)
    axes[1].set_title("Image 2")

    # Tạo danh sách các màu sắc
    colors = list(mcolors.TABLEAU_COLORS.values())

    def draw_cluster(ax, cluster, color):
            # Vẽ các đa giác của cụm
            for polygon in cluster[2]:
                poly = patches.Polygon(polygon, closed=True, fill=False, edgecolor=color, linewidth=1, facecolor=color, alpha=0.1)
                ax.add_patch(poly)

            # Vẽ vòng tròn, đường nối và tâm
            centroid = compute_centroid(cluster[2][0])
            all_centroids = np.array([compute_centroid(poly) for poly in cluster[2]])
            max_distance = np.max(np.linalg.norm(all_centroids - centroid, axis=1))
            circle = plt.Circle(centroid, max_distance, color=color,alpha=0.1, fill=True, linewidth=1)
            ax.add_patch(circle)
            ax.plot(centroid[0], centroid[1], 'o', color=color, markersize=8)  # Tâm cùng màu với cụm
            for poly in cluster[2]:
                poly_centroid = compute_centroid(poly)
                ax.plot([centroid[0], poly_centroid[0]], [centroid[1], poly_centroid[1]], '--', linewidth=1, color=color)

    # Vẽ các cụm giống nhau với cùng màu sắc
    for i, (cluster1, cluster2) in enumerate(matching_clusters):
        color = colors[i % len(colors)]
        # print(f"cluster1[{i}]: ",cluster1)
        # print(f"cluster2[{i}]: ",cluster2)
        draw_cluster(axes[0], cluster1, color)
        draw_cluster(axes[1], cluster2, color)
    return fig