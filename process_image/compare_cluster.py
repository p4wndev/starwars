import matplotlib.pyplot as plt
import cv2
from process_image.process_cluster import process_cluster
from process_image.compare_polygon import normalize_and_compute_cosine_similarity, validate_input
import os
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
            similarity, w1, w2 = normalize_and_compute_cosine_similarity(cluster1, cluster2)
            similarity_scores.append((similarity, f"Image {i+1}", image_cluster2,w1,w2))
        
        if progress_callback:
            progress = 0.2 + (0.6 * (i + 1) / total_images)
            progress_callback(progress)

    similarity_scores = sorted(similarity_scores, key=lambda x: x[0], reverse=True)[:top_n]

    highest_similarity = similarity_scores[0][0] if similarity_scores else 0
    
    if progress_callback:
        progress_callback(1.0)  # 100% progress

    return image_cluster1, highest_similarity, similarity_scores, None
    