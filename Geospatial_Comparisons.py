import streamlit as st
import cv2
import numpy as np
from PIL import Image
from process_image.compare_polygon import process_image_and_find_similar_polygons, process_image_and_find_similar_polygons_2, compare_maps, plot_polygon_and_image, extract_polygons, calculate_angles
from process_image.compare_cluster import process_image_and_find_similar_cluster, process_cluster, display_results, compare_images_knn, plot_matching_clusters_knn
import io, base64
import os
import matplotlib.pyplot as plt
import streamlit_shadcn_ui as ui
from htbuilder import HtmlElement, div, ul, li, br, hr, a, p, img, styles, classes, fonts
from htbuilder.units import percent, px
from htbuilder.funcs import rgba, rgb
import pandas as pd
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

st.set_page_config(page_title="Starwar Project", page_icon="🌏") #layout="wide"

def image_to_base64(uploaded_file):
    image = Image.open(uploaded_file)
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def display_selected_image(uploaded_file, container_id, index):
    st.markdown(f"""
    <div class="image-container" id="{container_id}">
        <img src="data:image/png;base64,{image_to_base64(uploaded_file)}" alt="{uploaded_file.name}">
    </div>
    """, unsafe_allow_html=True)

def add_white_background(image):
    # Kiểm tra xem ảnh có kênh alpha không
    if image.shape[2] == 4:
        # Tạo một ảnh nền trắng
        white_background = np.ones(image.shape[:3], dtype=np.uint8) * 255

        # Trích xuất kênh alpha
        alpha = image[:, :, 3] / 255.0

        # Thực hiện alpha blending
        for c in range(3):
            white_background[:, :, c] = (1 - alpha) * white_background[:, :, c] + alpha * image[:, :, c]

        return white_background.astype(np.uint8)
    else:
        # Nếu ảnh không có kênh alpha, trả về ảnh gốc
        return image

def read_image(uploaded_file):
    if uploaded_file is not None:
        uploaded_file.seek(0)
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)  # Đọc cả kênh alpha nếu có

        # Chuyển đổi từ BGR sang RGB
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

        return add_white_background(image)
    return None


# def read_image(uploaded_file):
#     if uploaded_file is not None:
#         uploaded_file.seek(0)
#         file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
#         image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
#         return image
#     return None

def main():
    st.markdown("""
    <style>
        .st-emotion-cache-13ln4jf{
            padding: 2rem;
        }
        .st-emotion-cache-1jicfl2 {
            padding: 1rem;
        }
        .image-container {
            width: 5vw;
            height: 5vw;
            border: 1px solid #cccccc;
            border-radius: 5px;
            overflow: hidden;
            display: flex;
            justify-content: center;
            align-items: center;
            margin-bottom: 10px;
            # box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2);
        }
        .image-container img {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
        }
        .stacked-images {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }
        .upload-section {
            margin-bottom: 20px;
        }
        .center-tabs {
            display: flex;
            justify-content: center;
        }
    </style>
    """, unsafe_allow_html=True)


    current_dir = os.path.dirname(__file__)
    image_path_logo2 = os.path.join(current_dir, "images", "LOGO2.png")
    image_path_logo1 = os.path.join(current_dir, "images", "logo1.jpg")
    image_path_eu = os.path.join(current_dir, "images", "eu.png")
    # st.markdown("<h1 style='text-align: center; color: #2563EB;'> Project STARWARS</h1>", unsafe_allow_html=True)
    logo = st.columns([0.2,0.5,0.1,0.2])
    with logo[1]:
        st.write("")
        st.image(image_path_logo2)
    with logo[2]:
        st.write("")
        st.write("")
        st.write("")
        st.write("")

        st.image(image_path_eu)
    st.logo(image_path_logo1)
    st.sidebar.markdown(" ")
    left_co1, cent_co1,last_co1 = st.columns([0.0001,0.9,0.0001])
    with cent_co1:
        tabs = ui.tabs(options=['Polygon vs Polygon', 'Polygon vs Map', 'Cluster vs Cluster', 'Cluster vs Map', 'Map vs Map'], default_value='Polygon vs Polygon', key="tabs")
    if tabs == "Polygon vs Polygon":
        polygon_vs_polygon()
    elif tabs == "Polygon vs Map":
        polygon_vs_map()
    elif tabs == "Cluster vs Cluster":
        cluster_vs_cluster()
    elif tabs == "Cluster vs Map":
        cluster_vs_map()
    else:
        map_vs_map()
    footer()
def find_polygon_with_most_vertices(polygons):
    """Finds the polygon with the most vertices."""
    return max(polygons, key=lambda poly: len(poly[0]))

def polygon_vs_polygon():
    error_container = st.empty()  # Placeholder for error messages
    container = st.container(border=True)
    compare_button = st.button("Compare", type="primary", key="polygon_vs_polygon_button", use_container_width=True)
    with st.expander("Upload Images", expanded=True):
        option = st.selectbox(
            "Select a comparison approach",
            ("Subarrays", "DTW"),
        )
        if option == "Subarrays":
            col = st.columns(2)
            with col[0]:
                k1 = st.number_input("Choose the tolerance angle (Degrees)", min_value=1, max_value=50, value=5, step=1, key="polygon_vs_polygon_number_input")
            with col[1]:
                k2 = st.number_input("Choose the minimum subsequence length", min_value=2, max_value=50, value=3, step=1, key="min_subsequence_length")
            vertex_angle_weight = st.slider("Vertex Angles Weight", min_value=0.0, max_value=1.0, value=0.5, step=0.1, key='pvp_vertex_angle_weight')
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                """
                <h4 style='color: #2563EB;'>First Polygon</h4>
                """,
                unsafe_allow_html=True)
            uploaded_file1 = st.file_uploader("Select first polygon image", type=['png', 'jpg', 'jpeg'], key="polygon_vs_polygon_uploader1")
            if uploaded_file1:
                display_selected_image(uploaded_file1, "polygon_container1", 0)

        with col2:
            st.markdown(
                """
                <h4 style='color: #2563EB;'>Second Polygon</h4>
                """,
                unsafe_allow_html=True)
            uploaded_file2 = st.file_uploader("Select second polygon image", type=['png', 'jpg', 'jpeg'], key="polygon_vs_polygon_uploader2")
            if uploaded_file2:
                display_selected_image(uploaded_file2, "polygon_container2", 0)

    if compare_button:
        if not uploaded_file1 or not uploaded_file2:
            error_container.error("Error: Please upload an image for both polygons.")
        else:
            with st.spinner("Comparing images..."):
                image1 = read_image(uploaded_file1)
                img1 = read_image(uploaded_file1)
                img11 = read_image(uploaded_file1)
                image2 = read_image(uploaded_file2)
                img2 = read_image(uploaded_file2)
                img22 = read_image(uploaded_file2)
                if image1 is None or image2 is None:
                    error_container.error("Error: One or both images could not be read.")
                else:
                    if option == "Subarrays":
                        result_rgb, highest_similarity, similarity_scores, error_message = process_image_and_find_similar_polygons(image1, image2, top_n=1, progress_callback=None, is_one_vs_one=True, option=option, k1=k1, k2=k2, vertex_angle_weight=vertex_angle_weight)
                        img22,_,_,_ = process_image_and_find_similar_polygons(img22,img11 , top_n=1, progress_callback=None, is_one_vs_one=True, option=option, k1=k1, k2=k2,vertex_angle_weight=vertex_angle_weight)
                        if error_message:
                            error_container.error(error_message)
                        else:
                            with container:
                                if highest_similarity > 1:
                                    st.subheader("Comparison Result")
                                    ui.badges([(f"Highest Similarity: {highest_similarity:.2f}%",'default')])
                                    col = st.columns(2)
                                    with col[0]:
                                        st.image(result_rgb, caption="Polygon 1", use_column_width=True)
                                    with col[1]:
                                        st.image(img22, caption="Polygon 2")
                                    with st.expander("View Explanation"):
                                        col1, col2 = st.columns(2)
                                        polygon1 = extract_polygons(img1)
                                        polygon2 = extract_polygons(img2)
                                        centroid1, sorted_vertices1, vertex_angles1, sorted_angles1, polygon_angles1 = calculate_angles(polygon1[0][0])
                                        fig1, polygon_angles1, vertex_angles1 = plot_polygon_and_image(polygon1[0][0], centroid1, vertex_angles1, sorted_angles1, polygon_angles1)
                                        centroid2, sorted_vertices2, vertex_angles2, sorted_angles2, polygon_angles2 = calculate_angles(polygon2[0][0])
                                        # vertices, centroid, vertex_angles, sorted_angles, polygon_angles
                                        fig2, polygon_angles2, vertex_angles2 = plot_polygon_and_image(polygon2[0][0], centroid2, vertex_angles2, sorted_angles2, polygon_angles2)
                                        # st.code("combined_angles = np.concatenate([vertex_angles, sorted(polygon_angles)])")
                                        # value_7 = np.concatenate([vertex_angles1, polygon_angles1])
                                        # value_8 = np.concatenate([vertex_angles2, polygon_angles2])

                                        # df = pd.DataFrame({
                                        #     'Polygon': [1, 2],
                                        #     'combined_angles': [value_7, value_8]
                                        # })
                                        # st.table(df)
                                        st.markdown(
                                        """
                                        <h4 style='color: #2563EB;'>Enumerating Sub-arrays (denoted as SArr):</h4>
                                        """,
                                        unsafe_allow_html=True)
                                        st.latex(r"""SArr(A) = \{A[i:j] \mid 0\leq i\leq j \leq |A|\}""")
                                        st.latex(r"""SArr(B) = \{B[i:j] \mid 0\leq i\leq j \leq |B|\}""")
                                        st.code("""def subarrays(arr):
        n = len(arr)
        result = []
        for start in range(n):
            for length in range(1, n + 1):
                subarray = [arr[(start + i) % n] for i in range(length)]
                result.append(subarray)
        return result""")
                                        st.markdown(
                                        """
                                        <h4 style='color: #2563EB;'>Compute matching score:</h4>
                                        """,
                                        unsafe_allow_html=True)
                                        st.latex(r"""
\text{score}_i =
\begin{cases}
1, & \text{if } |a_i - b_i| \leq k_1, \\
\frac{a_i}{b_i}, & \text{if } |a_i - b_i| > k_1, \, b_i > a_i, \, \frac{a_i}{b_i} \geq 0.5, \\
0, & \text{otherwise.}
\end{cases}
""")
                                        st.code("""def compute_matching_score(subA, subB):
        total_score = 0
        for a, b in zip(subA, subB):
            if abs(a - b) <= k1:
                score = 1
            elif abs(a - b) > k1 and b > a and (a / b) >= 0.5:
                score = a / b
            else:
                score = 0
            total_score += score
        return total_score""")
                                        st.markdown(
                                        """
                                        <h4 style='color: #2563EB;'>Finding the largest match:</h4>
                                        """,
                                        unsafe_allow_html=True)
                                        st.latex(r"""score_{\text{max}} = \max_{\substack{a \in SArr_A, b \in SArr_B \\ |a| \geq k_2, \, |a| = |b|}} \left\{ \text{compute\_matching\_score}(a, b) \right\}.""")
                                        st.code("""def find_largest_consecutive_matches(A, B, k1, k2):
    subarrays_A = subarrays(A)
    subarrays_B = subarrays(B)

    largest_match = None
    max_total_score = 0

    for subA in subarrays_A:
        if len(subA) >= k2:
            for subB in subarrays_B:
                if len(subB) == len(subA):
                    total_score = compute_matching_score(subA, subB)
                    if total_score > max_total_score:
                        largest_match = (subA, subB)
                        max_total_score = total_score

    return max_total_score""")
                                        st.markdown(
                                        """
                                        <h4 style='color: #2563EB;'>Matching Percentage (P) between two Sub-Arrays:</h4>
                                        """,
                                        unsafe_allow_html=True)
                                        st.latex(r"""\mathcal{P} = \Big(\frac{score_{\text{max}}}{\max(|A|,|B|)}\Big)\times 100""")
                                        st.code("""def calculate_percentage_match(A, B, k1, k2):
    max_total_score = find_largest_consecutive_matches(A, B, k1, k2)

    if max_total_score == 0:
        return 0.0

    total_length_A = len(A)
    total_length_B = len(B)

    percentage_match = (max_total_score / max(total_length_A, total_length_B)) * 100
    return percentage_match
""")
                                        with col1:
                                            ui.badges([(f"Polygon 1",'outline')])
                                            st.pyplot(fig1)
                                            ui.badges([(f"Polygon Angles 1",'secondary')])
                                            st.write(polygon_angles1)
                                            ui.badges([(f"Vertex Angles 1",'secondary')])
                                            st.write(vertex_angles1)
                                            # st.write(f"{similarity_scores[0][7]}")
                                        with col2:
                                            ui.badges([(f"Polygon 2",'outline')])
                                            st.pyplot(fig2)
                                            ui.badges([(f"Polygon Angles 2",'secondary')])
                                            st.write(polygon_angles2)
                                            ui.badges([(f"Vertex Angles 2",'secondary')])
                                            st.write(vertex_angles2)
                                            # st.write(similarity_scores[0][8])

                                        plt.close(fig1)  # Close the figure to free up memory
                                        plt.close(fig2)  # Close the figure to free up memory
                                        if len(similarity_scores) == 0:
                                            st.write("No polygons found for explanation.")
                    else:
                        result_rgb, highest_similarity, similarity_scores, error_message = process_image_and_find_similar_polygons(image1, image2, top_n=1, progress_callback=None, is_one_vs_one=True, option=option,vertex_angle_weight=vertex_angle_weight)
                        img22,_,_,_ = process_image_and_find_similar_polygons(img22,img11 , top_n=1, progress_callback=None, is_one_vs_one=True, option=option,vertex_angle_weight=vertex_angle_weight)
                        if error_message:
                            error_container.error(error_message)
                        else:
                            with container:
                                if highest_similarity > 1:
                                    st.subheader("Comparison Result")
                                    ui.badges([(f"Highest Similarity: {highest_similarity:.2f}%",'default')])
                                    col = st.columns(2)
                                    with col[0]:
                                        st.image(result_rgb, caption="Polygon 1", use_column_width=True)
                                    with col[1]:
                                        st.image(img22, caption="Polygon 2")
                                    with st.expander("View Explanation"):
                                        # Display input polygons side by side
                                        col1, col2 = st.columns(2)
                                        value_7 = similarity_scores[0][7]
                                        value_8 = similarity_scores[0][8]
                                        df = pd.DataFrame({
                                            'Polygon': [1, 2],
                                            'Embedded vector': [value_7, value_8]
                                        })
                                        st.table(df)
                                        distance, _ = fastdtw(value_7.reshape(-1, 1), value_8.reshape(-1, 1), dist=euclidean)
                                        # ui.badges([(f"DTW Distance: {distance}",'secondary')])
                                        st.write("DTW Distance: ",distance )
                                        max_distance = np.linalg.norm(np.ones_like(value_7) - np.zeros_like(value_7))
                                        st.write("Max DTW Distance: ",max_distance )
                                        percent_similarity = max(0, (1 - (distance / max_distance)) * 100)
                                        st.write("Percent Similarity: ",percent_similarity )
                                        st.latex(r"""Percent Similarity = \max \left( 0, \left( 1 - \frac{\text{distance}}{\text{max\_distance}} \right) \times 100 \right)""")
                                        # st.write()
                                        polygon1 = extract_polygons(img1)
                                        polygon2 = extract_polygons(img2)
                                        # vertices, centroid, min_angle_vertex, vertex_angles, sorted_angles, polygon_angles
                                        centroid1, sorted_vertices1, vertex_angles1, sorted_angles1, polygon_angles1 = calculate_angles(polygon1[0][0])
                                        fig1, polygon_angles1, vertex_angles1 = plot_polygon_and_image(polygon1[0][0], centroid1, sorted_vertices1, vertex_angles1, sorted_angles1, polygon_angles1)
                                        centroid2, sorted_vertices2, vertex_angles2, sorted_angles2, polygon_angles2 = calculate_angles(polygon2[0][0])
                                        fig2, polygon_angles2, vertex_angles2 = plot_polygon_and_image(polygon2[0][0], centroid2, sorted_vertices2, vertex_angles2, sorted_angles2, polygon_angles2)

                                        with col1:
                                            ui.badges([(f"Polygon 1",'outline')])
                                            st.pyplot(fig1)
                                            ui.badges([(f"Polygon Angles 1",'secondary')])
                                            st.write(polygon_angles1)
                                            ui.badges([(f"Vertex Angles 1",'secondary')])
                                            st.write(vertex_angles1)
                                            # st.write(f"{similarity_scores[0][7]}")

                                        with col2:
                                            ui.badges([(f"Polygon 2",'outline')])
                                            st.pyplot(fig2)
                                            ui.badges([(f"Polygon Angles 2",'secondary')])
                                            st.write(polygon_angles2)
                                            ui.badges([(f"Vertex Angles 2",'secondary')])
                                            st.write(vertex_angles2)
                                            # st.write(similarity_scores[0][8])

                                        plt.close(fig1)  # Close the figure to free up memory
                                        plt.close(fig2)  # Close the figure to free up memory
                                        if len(similarity_scores) == 0:
                                            st.write("No polygons found for explanation.")
                                else:
                                    st.subheader("Comparison Result")
                                    ui.badges([(f"Highest Similarity: {highest_similarity:.2f}%",'default'),("Two polygons are totally different",'destructive')], class_name="flex gap-2")
                                    with st.expander("View Explanation"):
                                        # Display input polygons side by side
                                        col1, col2 = st.columns(2)
                                        value_7 = similarity_scores[0][7]
                                        value_8 = similarity_scores[0][8]
                                        # df = pd.DataFrame({
                                        #     'Polygon': [1, 2],
                                        #     'Embedded Vector': [value_7, value_8]
                                        # })
                                        # st.table(df)
                                        distance, _ = fastdtw(value_7.reshape(-1, 1), value_8.reshape(-1, 1), dist=euclidean)
                                        # ui.badges([(f"DTW Distance: {distance}",'secondary')])
                                        st.write("DTW Distance: ",distance )
                                        max_distance = np.linalg.norm(np.ones_like(value_7) - np.zeros_like(value_7))
                                        st.write("Max DTW Distance: ",max_distance )
                                        percent_similarity = max(0, (1 - (distance / max_distance)) * 100)
                                        st.write("Percent Similarity: ",percent_similarity )
                                        st.latex(r"""Percent Similarity = \max \left( 0, \left( 1 - \frac{\text{distance}}{\text{max\_distance}} \right) \times 100 \right)""")
                                        polygon1 = extract_polygons(img1)
                                        polygon2 = extract_polygons(image2)
                                        # vertices, centroid, min_angle_vertex, vertex_angles, sorted_angles, polygon_angles
                                        centroid1, sorted_vertices1, vertex_angles1, sorted_angles1, polygon_angles1 = calculate_angles(polygon1[1][0])
                                        fig1, polygon_angles1, vertex_angles1 = plot_polygon_and_image(polygon1[1][0], centroid1, sorted_vertices1, vertex_angles1, sorted_angles1, polygon_angles1)
                                        centroid2, sorted_vertices2, vertex_angles2, sorted_angles2, polygon_angles2 = calculate_angles(polygon2[1][0])
                                        fig2, polygon_angles2, vertex_angles2 = plot_polygon_and_image(polygon2[1][0], centroid2, sorted_vertices2, vertex_angles2, sorted_angles2, polygon_angles2)

                                        with col1:
                                            ui.badges([(f"Polygon 1",'outline')])
                                            st.pyplot(fig1)
                                            ui.badges([(f"Polygon Angles 1",'secondary')])
                                            st.write(polygon_angles1)
                                            ui.badges([(f"Vertex Angles 1",'secondary')])
                                            st.write(vertex_angles1)

                                        with col2:
                                            ui.badges([(f"Polygon 2",'outline')])
                                            st.pyplot(fig2)
                                            ui.badges([(f"Polygon Angles 2",'secondary')])
                                            st.write(polygon_angles2)
                                            ui.badges([(f"Vertex Angles 2",'secondary')])
                                            st.write(vertex_angles2)

                                        plt.close(fig1)  # Close the figure to free up memory
                                        plt.close(fig2)  # Close the figure to free up memory
                                        if len(similarity_scores) == 0:
                                            st.write("No polygons found for explanation.")
def polygon_vs_map():
    # st.subheader("Polygon vs Map 🗺️")
    error_container = st.empty()  # Placeholder for error messages
    container = st.container(border=True)

    compare_button = st.button("Compare", type="primary", key="polygon_vs_map_button",use_container_width=True)
    with st.expander("Upload Images", expanded=True):
        option = st.selectbox(
            "Select a comparison approach",
            ("Subarrays", "DTW"),
        )
        if option == "Subarrays":
            col = st.columns(3)
            with col[0]:
                k1 = st.number_input("Tolerance angle (Degrees)", min_value=1, max_value=50, value=5, step=1, key="polygon_vs_map_number_input")
            with col[1]:
                k2 = st.number_input("Minimum subsequence length", min_value=2, max_value=50, value=3, step=1, key="min_subsequence_length_2")
            with col[2]:
                top_n_1 = st.number_input("Select top N for comparison", min_value=1, max_value=10, value=1, step=1, key="polygon_vs_map_slider")
            vertex_angle_weight = st.slider("Vertex Angles Weight", min_value=0.0, max_value=1.0, value=0.5, step=0.1, key='pvm_vertex_angle_weight')
        # top_n = st.select_slider("Select top N for comparison", options=[1, 2, 3, 4, 5], value=1, key="polygon_vs_map_slider")
        if option == "DTW":
            top_n_2 = st.number_input("Select top N for comparison", min_value=1, max_value=10, value=1, step=1, key="polygon_vs_map_slider_2")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                """
                <h4 style='color: #2563EB;'>Polygon Image</h4>
                """,
                unsafe_allow_html=True)
            uploaded_file1 = st.file_uploader("Select polygon image", type=['png', 'jpg', 'jpeg'], key="polygon_vs_map_uploader1")
            if uploaded_file1:
                display_selected_image(uploaded_file1, "polygon_container_1", 0)

        with col2:
            st.markdown(
                """
                <h4 style='color: #2563EB;'>Map Image</h4>
                """,
                unsafe_allow_html=True)
            uploaded_file2 = st.file_uploader("Select map image", type=['png', 'jpg', 'jpeg'], key="polygon_vs_map_uploader2")
            if uploaded_file2:
                display_selected_image(uploaded_file2, "map_container_1", 0)
    if compare_button:
        if not uploaded_file1 or not uploaded_file2:
            error_container.error("Error: Please upload an image for both the polygon and the map.")
        else:
            with st.spinner("Comparing images..."):
                uploaded_file1_image = read_image(uploaded_file1)
                uploaded_file2_image = read_image(uploaded_file2)

                if uploaded_file1_image is None or uploaded_file2_image is None:
                    error_container.error("Error: One or both images could not be read.")
                else:
                    progress_text = "Operation in progress. Please wait..."
                    my_bar = st.progress(0, text=progress_text)

                    def update_progress(progress):
                        my_bar.progress(progress, text=progress_text)
                    if option == "Subarrays":
                        result, highest_similarity, similarity_scores, error_message = process_image_and_find_similar_polygons_2(
                            uploaded_file2_image, uploaded_file1_image,top_n_1, progress_callback=update_progress, compare_mode="polygon_vs_map", option=option, k1=k1, k2=k2
                        )
                    else:
                        result, highest_similarity, similarity_scores, error_message = process_image_and_find_similar_polygons(
                            uploaded_file2_image, uploaded_file1_image,top_n_2, progress_callback=update_progress, compare_mode="polygon_vs_map", option=option, vertex_angle_weight=vertex_angle_weight
                        )
                    if error_message:
                        error_container.error(error_message)
                    else:
                        with container:
                            st.subheader("Comparison Result")
                            ui.badges([(f"Highest Similarity: {highest_similarity:.2f}%",'default')])
                            st.image(result, caption="Result", use_column_width=True)
                            with st.expander("View Explanation"):
                                st.write("Identify the polygon with the closest match to the input polygon, determined by comparing their sets of characteristic angles.")
                    my_bar.empty()

def cluster_vs_cluster():
    error_container = st.empty()  # Placeholder for error messages
    result_container = st.empty()  # Placeholder for the comparison result
    container = st.container(border=True)
    compare_button = st.button("Compare", type="primary", key="cluster_vs_cluster_button",use_container_width=True)
    with st.expander("Upload Images", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                """
                <h4 style='color: #2563EB;'>First Cluster</h4>
                """,
            unsafe_allow_html=True)
            uploaded_file1 = st.file_uploader("Select first cluster image", type=['png', 'jpg', 'jpeg'], key="cluster_vs_cluster_uploader1")
            if uploaded_file1:
                display_selected_image(uploaded_file1, "cluster_container1", 0)

        with col2:
            st.markdown(
                """
                <h4 style='color: #2563EB;'>Second Cluster</h4>
                """,
            unsafe_allow_html=True)
            uploaded_files2 = st.file_uploader("Select second cluster images", type=['png', 'jpg', 'jpeg'], key="cluster_vs_cluster_uploader2", accept_multiple_files=True)
            if uploaded_files2:
                st.markdown('<div class="stacked-images">', unsafe_allow_html=True)
                for i, file in enumerate(uploaded_files2):
                    display_selected_image(file, f"cluster_container2_{i}", i)
                st.markdown('</div>', unsafe_allow_html=True)


    if compare_button:
        if not uploaded_file1 or not uploaded_files2:
            error_container.error("Error: Please upload an image to the first cluster and at least one image to the second cluster.")
        else:
            with st.spinner("Comparing images..."):
                image1 = read_image(uploaded_file1)
                if image1 is None:
                    error_container.error(f"Error: Could not read image {uploaded_file1.name}")
                else:
                    image2_list = [read_image(file2) for file2 in uploaded_files2 if read_image(file2) is not None]
                    result, highest_similarity, similarity_scores, error_message = process_image_and_find_similar_cluster(image1, image2_list, top_n=5)

                    if error_message:
                        error_container.error(error_message)
                    elif result is not None and similarity_scores:
                        with result_container.container():
                            with container:
                                st.subheader("Comparison Result")
                                ui.badges([(f"Highest Similarity: {highest_similarity:.2f}%",'default')])
                                col1, col2 = st.columns(2)
                                with col1:
                                    with st.container(border=True):
                                        st.image(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB), caption="First Cluster Image", use_column_width=True)
                                with col2:
                                    with st.container(border=True):
                                        st.image(cv2.cvtColor(similarity_scores[0][2], cv2.COLOR_BGR2RGB), caption="Most Similar Cluster Image", use_column_width=True)
                                with st.expander("View Explanation"):
                                    col1, col2 = st.columns(2)
                                    value_7 = similarity_scores[0][3]
                                    value_8 = similarity_scores[0][4]
                                    df = pd.DataFrame({
                                        'Cluster': [1, 2],
                                        'Embedded vector': [value_7, value_8]
                                    })
                                    st.table(df)
                                    distance, _ = fastdtw(value_7.reshape(-1, 1), value_8.reshape(-1, 1), dist=euclidean)
                                    st.write("DTW Distance: ",distance )
                                    max_distance = np.linalg.norm(np.ones_like(value_7) - np.zeros_like(value_7))
                                    st.write("Max DTW Distance: ",max_distance )
                                    percent_similarity = max(0, (1 - (distance / max_distance)) * 100)
                                    st.write("Percent Similarity: ",percent_similarity )
                                    st.latex(r"""Percent Similarity = \max \left( 0, \left( 1 - \frac{\text{distance}}{\text{max\_distance}} \right) \times 100 \right)""")
                                    # st.write()
                                    # embedded_vector, image, polygons, centroid, relative_angles
                                    embedded_vector1, img1, polygons1, centroid1, relative_angles1 = process_cluster(image1)
                                    embedded_vector2, img2, polygons2, centroid2, relative_angles2 = process_cluster(image2_list[0])
                                    # vertices, centroid, min_angle_vertex, vertex_angles, sorted_angles, polygon_angles
                                    fig1, relative_angles1 = display_results(img1, polygons1, centroid1, relative_angles1)
                                    fig2, relative_angles2 = display_results(img2, polygons2, centroid2, relative_angles2)

                                    with col1:
                                        ui.badges([(f"Cluster 1",'outline')])
                                        st.pyplot(fig1)
                                        ui.badges([(f"Relative Angles 1",'secondary')])
                                        relative_angles1 = [angle for angle, _ in relative_angles1]
                                        st.write(relative_angles1)

                                    with col2:
                                        ui.badges([(f"Cluster 2",'outline')])
                                        st.pyplot(fig2)
                                        ui.badges([(f"Relative Angles 2",'secondary')])
                                        relative_angles2 = [angle for angle, _ in relative_angles2]
                                        st.write(relative_angles2)

                                    plt.close(fig1)  # Close the figure to free up memory
                                    plt.close(fig2)  # Close the figure to free up memory
                                with st.expander("View Other Results"):
                                    for similarity, filename, image, w1, w2 in similarity_scores[1:]:
                                        with st.container():

                                            if similarity < 1:
                                                ui.badges([(f"Similarity: {similarity:.2f}%",'default'), ("Two cluster are totally different",'destructive')], class_name="flex gap-2", key=f"badge_{filename}")
                                            else:
                                                ui.badges([(f"Similarity: {similarity:.2f}%",'secondary')], key=f"badge{filename}")
                                            col1, col2 = st.columns(2)
                                            with col1:
                                                with st.container(border=True):
                                                    st.image(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB), caption="First Cluster Image", use_column_width=True)
                                            with col2:
                                                with st.container(border=True):
                                                    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption=f"{filename}", use_column_width=True)
                                            st.write("---")
                    else:
                        error_container.error("No valid comparisons could be made.")

def cluster_vs_map():
    error_container = st.empty()  # Placeholder for error messages
    container = st.container(border=True)

    compare_button = st.button("Compare", type="primary", key="cluster_vs_map_button", use_container_width=True)
    with st.expander("Upload Images", expanded=True):
        k = st.number_input("Number of nearest neighbors (KNN)", min_value=3, max_value=20, value=3, step=1, key="cluster_vs_map_number_input")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                """
                <h4 style='color: #2563EB;'>Cluster Image</h4>
                """,
                unsafe_allow_html=True)
            uploaded_file1 = st.file_uploader("Select cluster image", type=['png', 'jpg', 'jpeg'], key="cluster_vs_map_uploader1")
            if uploaded_file1:
                display_selected_image(uploaded_file1, "cluster_container_1", 0)

        with col2:
            st.markdown(
                """
                <h4 style='color: #2563EB;'>Map Image</h4>
                """,
                unsafe_allow_html=True)
            uploaded_file2 = st.file_uploader("Select map image", type=['png', 'jpg', 'jpeg'], key="cluster_vs_map_uploader2")
            if uploaded_file2:
                display_selected_image(uploaded_file2, "map_container_1", 0)
    if compare_button:
        if not uploaded_file1 or not uploaded_file2:
            error_container.error("Error: Please upload an image for both the cluster and the map.")
        else:
            with st.spinner("Comparing images..."):
                uploaded_file1_image = read_image(uploaded_file1)
                uploaded_file2_image = read_image(uploaded_file2)

                if uploaded_file1_image is None or uploaded_file2_image is None:
                    error_container.error("Error: One or both images could not be read.")
                else:
                    matching_clusters, num_clusters1, num_clusters2 = compare_images_knn(uploaded_file1_image, uploaded_file2_image, k)
                    result = plot_matching_clusters_knn(uploaded_file1_image, uploaded_file2_image, matching_clusters)
                    with container:
                        st.subheader("Comparison Result")
                        ui.badges([(f"Matching Clusters: {len(matching_clusters)}",'default')])
                        st.pyplot(result)
                        with st.expander("View Explanation"):
                            st.write("")
                            # st.write("Identify the polygon with the closest match to the input polygon, determined by comparing their sets of characteristic angles.")


def map_vs_map():
    error_container = st.empty()  # Placeholder for error messages
    result_container = st.empty()  # Placeholder for the comparison result
    container = st.container(border=True)

    compare_button = st.button("Compare", type="primary", key="map_vs_map_button", use_container_width=True)
    spinner = st.spinner("Comparing maps...")
    with st.expander("Upload Images", expanded=True):
        similarity_threshold = st.slider("Similarity Threshold (%)", min_value=0, max_value=100, value=80, step=1)
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<h4 style='color: #2563EB;'>First Map</h4>", unsafe_allow_html=True)
            uploaded_file1 = st.file_uploader("Select first map image", type=['png', 'jpg', 'jpeg'], key="map_vs_map_uploader1")
            if uploaded_file1:
                display_selected_image(uploaded_file1, "map_container1", 0)

        with col2:
            st.markdown("<h4 style='color: #2563EB;'>Second Maps</h4>", unsafe_allow_html=True)
            uploaded_files2 = st.file_uploader("Select second map images", type=['png', 'jpg', 'jpeg'], key="map_vs_map_uploader2", accept_multiple_files=True)
            if uploaded_files2:
                st.markdown('<div class="stacked-images">', unsafe_allow_html=True)
                for i, file in enumerate(uploaded_files2):
                    display_selected_image(file, f"map_container2_{i}", i)
                st.markdown('</div>', unsafe_allow_html=True)

    if compare_button:
        if not uploaded_file1 or not uploaded_files2:
            error_container.error("Error: Please upload an image to the first map and at least one image to the second maps.")
        else:
            with spinner:
                input_image = read_image(uploaded_file1)
                if input_image is None:
                    error_container.error(f"Error: Could not read image {uploaded_file1.name}")
                else:
                    image_set = [read_image(file2) for file2 in uploaded_files2 if read_image(file2) is not None]
                    comparison_results = []

                    for i, image2 in enumerate(image_set):
                        result, matched_polygons, total_polygons, input_polygon_count, error_message = compare_maps(input_image, image2, similarity_threshold=similarity_threshold)
                        if error_message:
                            error_container.error(f"Error in comparing with {uploaded_files2[i].name}: {error_message}")
                        elif result is not None:
                            similarity_ratio = matched_polygons / total_polygons if total_polygons > 0 else 0
                            comparison_results.append((similarity_ratio * 100, uploaded_files2[i].name, result, matched_polygons, total_polygons, input_polygon_count))

                    if comparison_results:
                        comparison_results.sort(key=lambda x: x[0], reverse=True)
                        highest_similarity, highest_similarity_name, highest_similarity_result, highest_matched_polygons, highest_total_polygons, input_polygon_count = comparison_results[0]

                        with result_container.container():
                            with container:
                                st.subheader("Comparison Result")
                                ui.badges([
                                    (f"Highest Map Similarity: {highest_similarity:.2f}%", 'default'),
                                    (f"Input Polygons: {input_polygon_count}", 'secondary'),
                                    (f"Matched Polygons: {highest_matched_polygons}/{highest_total_polygons}", 'outline')
                                ], class_name="flex gap-2")

                                col1, col2 = st.columns(2)
                                with col1:
                                    with st.container(border=True):
                                        st.image(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB), caption="First Map Image", use_column_width=True)
                                with col2:
                                    with st.container(border=True):
                                        st.image(highest_similarity_result, caption=f"Most Similar Map: {highest_similarity_name}", use_column_width=True)
                                with st.expander("View Explanation"):
                                    st.write("Determine the map that best matches the input map, based on a comparison of their polygon characteristic angles. The matching criterion is that the ratio of the number of matching polygons to the total number of polygons in the map.")
                                with st.expander("View Other Results"):
                                    for similarity, filename, result_image, matched_polygons, total_polygons, _ in comparison_results[1:]:
                                        with st.container():
                                            ui.badges([
                                                (f"Similarity: {similarity:.2f}%", 'secondary'),
                                                (f"Matched Polygons: {matched_polygons}/{total_polygons}", 'outline')
                                            ], class_name="flex gap-2", key=f"badge{filename}")
                                            col1, col2 = st.columns(2)

                                            with col1:
                                                with st.container(border=True):
                                                    st.image(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB), caption="First Map Image", use_column_width=True)
                                            with col2:
                                                with st.container(border=True):
                                                    st.image(result_image, caption=f"{filename}", use_column_width=True)
                                            st.write("---")
                    else:
                        error_container.error("No valid comparisons could be made.")

def image(src_as_string, **style):
    return img(src=src_as_string, style=styles(**style))

def link(link, text, **style):
    return a(_href=link, _target="_blank", style=styles(textDecoration="none", **style))(text)

def link2(link, text, **style):
    return a(_href=link, _target="_blank", style=styles(**style))(text)
def layout(*args):

    style = """
    <style>
      # MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
    </style>
    """

    style_div = styles(
        left=0,
        bottom=0,
        margin=px(0, 0, 0, 0),
        width=percent(100),
        text_align="center",
        height="100px",
        # opacity=0.9
    )

    style_hr = styles(
    )

    body = p()
    foot = div(style=style_div)(hr(style=style_hr), body)

    st.markdown(style, unsafe_allow_html=True)

    for arg in args:
        if isinstance(arg, str):
            body(arg)
        elif isinstance(arg, HtmlElement):
            body(arg)

    st.markdown(str(foot), unsafe_allow_html=True)

def footer():
    myargs = [
        " Copyright © 2024, Created byㅤ",
        link("https://truongthanhma.github.io/site/", "Thanh Ma"),
        "ㅤ",
        link("https://fb.com/tuitenphuan", "An Thai"),
        "ㅤ",
        link("https://scholar.google.fr/citations?user=-3kO5x0AAAAJ&hl=fr", "Salem Benferhat"),
        br(),
        "STARWARS Website:ㅤ",
        link2("https://sites.google.com/view/horizoneurope2020-starwars/", "Horizon Europe 2020"),
    ]
    layout(*myargs)
if __name__ == "__main__":
    main()
