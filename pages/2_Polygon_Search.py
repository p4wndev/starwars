import streamlit as st
import cv2
import numpy as np
from PIL import Image
from process_image.compare_polygon import *
from process_image.windows_search import *
import io
import os
import matplotlib.pyplot as plt
from streamlit_image_select import image_select
import streamlit_shadcn_ui as ui
from htbuilder import HtmlElement, div, ul, li, br, hr, a, p, img, styles, classes, fonts
from htbuilder.units import percent, px
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
import pickle

st.set_page_config(page_title="Starwar Project", page_icon="🌏")


def listing_polygon_map(image, min_vertices, min_area, max_area):
    # Convert PIL image to NumPy array if necessary
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Debugging: print image shapes to ensure they are loaded correctly
    print(f"Image shape: {image.shape}")

    # Extract polygons from both images
    if image is not None and len(image.shape) == 3:
        list_polygons = extract_polygons_adjustable(image, min_vertices, min_area, max_area)
    else:
        raise ValueError(f"Invalid image1: shape = {image.shape if image is not None else None}")

    return list_polygons

def listing_polygon_input(image):
    # Convert PIL image to NumPy array if necessary
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Debugging: print image shapes to ensure they are loaded correctly
    print(f"Image shape: {image.shape}")

    # Extract polygons from both images
    if image is not None and len(image.shape) == 3:
        list_polygons = extract_polygons(image)
    else:
        raise ValueError(f"Invalid image: shape = {image.shape if image is not None else None}")

    return list_polygons

@st.cache_resource
def load_cached_polygons():
    cache_file_path = os.path.join(current_dir[:-6], "cache", "polygons2.pkl")
    with open(cache_file_path, 'rb') as f:
        return pickle.load(f)

def read_image_2(uploaded_file):
    if uploaded_file is not None:
        uploaded_file.seek(0)
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        return image
    return None

def display_map(image_path):
    if image_path is None:
        st.warning("No map to display.")
        return
    with st.expander("Map:", expanded=True):
        st.image(image_path)

def update_and_display_polygons(buffer_distance, min_area, max_area, image2, upload_info):
    polygons = extract_polygons_adjustable(image2, min_vertices=3, min_area=min_area, max_area=max_area)

    buffered_polygons = [Polygon(poly[0]).buffer(buffer_distance) for poly in polygons]
    merged_polygon = unary_union(buffered_polygons)
    final_polygon = merged_polygon.buffer(-buffer_distance)

    if isinstance(final_polygon, MultiPolygon):
        list_polygons_input = [(list(poly.exterior.coords), None) for poly in final_polygon.geoms]
    else:
        list_polygons_input = [(list(final_polygon.exterior.coords), None)]
    # print(list_polygons_input)
    if upload_info:
        if len(list_polygons_input[0][0]) == 0:
            upload_info.info("Please upload an image with polygons or adjust the buffer distance and area range")
            return None, 0
    merged_image_buffer = plot_polygons(list_polygons_input, '')
    merged_image_pil = Image.open(merged_image_buffer)
    merged_image_array = np.array(merged_image_pil)

    fig_m, ax_m = plt.subplots(figsize=(10, 10))
    ax_m.imshow(merged_image_array)
    num_polygons = len(list_polygons_input)
    ax_m.axis('off')

    # Chuyển fig thành image
    buf = io.BytesIO()
    fig_m.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)

    plt.close(fig_m)

    return img, num_polygons

@st.cache_data
def load_sample_image(image_path):
    return cv2.imread(image_path)

current_dir = os.path.dirname(__file__)
sample_map_path = os.path.join(current_dir[:-6], "images/", "map_blue_low.png")
sample_map_path_2 = os.path.join(current_dir[:-6], "images/", "map_blue_medium.png")
sample_input_1_path = os.path.join(current_dir[:-6], "images/", "sample_polygon_1.png")
sample_input_2_path = os.path.join(current_dir[:-6], "images/", "sample_polygon_2.png")
sample_input_3_path = os.path.join(current_dir[:-6], "images/", "sample_polygon_3.png")
sample_input_4_path = os.path.join(current_dir[:-6], "images/", "sample_polygon_4.png")
sample_input_5_path = os.path.join(current_dir[:-6], "images/", "sample_polygon_5.png")

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
    image_path_logo2 = os.path.join(current_dir[:-6], "images", "LOGO2.png")
    image_path_logo1 = os.path.join(current_dir[:-6], "images", "logo1.jpg")
    image_path_eu = os.path.join(current_dir[:-6], "images", "eu.png")
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
    upload_info = st.empty()
    if 'cache_map' not in st.session_state:
        st.session_state.cache_map = load_sample_image(sample_map_path_2)
    if 'input_map' not in st.session_state:
        st.session_state.input_map = []
    if 'list_polygons_cache' not in st.session_state:
        st.session_state.list_polygons_cache = load_cached_polygons()
    if 'list_polygons_map' not in st.session_state:
        st.session_state.list_polygons_map = []
    sample_input_expander = st.sidebar.expander("Use Sample Input and Map")
    with sample_input_expander:
        sample_input = image_select(
                    label="Select a input image",
                    images=[
                        sample_input_1_path,
                        sample_input_2_path,
                        sample_input_3_path,
                        sample_input_4_path,
                        sample_input_5_path,
                    ],
                    captions=["Sample 1", "Sample 2", "Sample 3", "Sample 4", "Sample 5"],
                )
        sidebar_expander_col = sample_input_expander.columns(2)
        with sidebar_expander_col[0]:
            display_sample = st.button("Display map", use_container_width=True)
        with sidebar_expander_col[1]:
            search_sample = st.button("Search", key='search_sample', use_container_width=True)
    if display_sample:
        display_map(sample_map_path_2)
    if sample_input:
        sample_image = cv2.imread(sample_input)
        if sample_image.shape[2] == 4:
            sample_image = cv2.cvtColor(sample_image, cv2.COLOR_BGRA2BGR)
        image2 = sample_image
    st.sidebar.link_button("Download sample map", "https://drive.google.com/file/d/1x610cvzhqdc9s5MoJKkWmY-ljHKQ5Ud3/view?usp=sharing")
    uploaded_file = st.sidebar.file_uploader("Upload an map image", type=["jpg", "jpeg", "png"], key="map_uploader")
    if uploaded_file is not None:
        # Đọc tấm hình từ file tải lên
        image = read_image_2(uploaded_file)
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        list_polygon_form = st.sidebar.form("map_form")
        with list_polygon_form:
            buffer_distance_ = st.number_input("Select Buffer Distance", min_value=0.0, max_value=5.0, value=1.0, step=0.1, key="buffer_distance_input_")
            min_area_, max_area_ = st.select_slider("Select Area Range", options=[50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 5000, 10000, 25000, 50000, 100000, 1000000], value=(50, 10000))
            create_list_polygon_button = list_polygon_form.form_submit_button("Create List Polygon")
        if create_list_polygon_button:
            # st.session_state.input_map = image
            map_image, num_polygons_map_input = update_and_display_polygons(buffer_distance_, min_area_, max_area_, image, upload_info)
            st.session_state.input_map = map_image
            st.session_state.list_polygons_map = listing_polygon_map(st.session_state.input_map, 5, min_area=min_area_, max_area=1000000)
            ui.badges([(f"Number of polygons in map: {num_polygons_map_input}",'secondary')], key=f"badge_map")
            map_plot = st.empty()
            map_plot.image(map_image)
    search_error = st.empty()
    with st.expander("Upload Images", expanded=True):
        uploaded_image = st.file_uploader("Upload a polygon image", type=["jpg", "jpeg", "png"], key="image_uploader")
        if uploaded_image is not None:
            image2 = read_image_2(uploaded_image)
            if image2.shape[2] == 4:
                image2 = cv2.cvtColor(image2, cv2.COLOR_BGRA2BGR)
        buffer_distance = st.number_input("Select Buffer Distance", min_value=0.0, max_value=5.0, value=1.0, step=0.1, key="buffer_distance_input")
        import sys
        max_int = sys.maxsize
        min_area, max_area = st.select_slider("Select Area Range", options=[50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 5000, 10000, 25000, 50000, 100000, 1000000], value=(50, 10000))
        if min_area and max_area:
            # Sử dụng st.empty() để tạo một vùng có thể cập nhật
            image_polygon, num_polygons_input = update_and_display_polygons(buffer_distance, min_area, max_area, image2, upload_info)
            # Cập nhật hình ảnh đa giác mỗi khi thay đổi giá trị min_area hoặc max_area
            # num_polygons_plot.markdown(f"Number of polygons in input: {num_polygons_input}")
            if image_polygon:
                ui.badges([(f"Number of polygons input: {num_polygons_input}",'secondary')], key=f"badge_input")
                polygon_plot = st.empty()
                polygon_plot.image(image_polygon)
                #Lấy vertices của polygon (ảnh black-white)
                list_polygons_input = listing_polygon_input(image_polygon)
                print(f"list_polygons_input: {list_polygons_input}")
                if (len(list_polygons_input) > 0):
                    print(f"input vertices: {list_polygons_input[0][0]}")
                if len(list_polygons_input) == 0:
                    upload_info.info("Please upload an image with polygons or adjust the buffer distance and area range")
                print(f"Number of polygons input.: {len(list_polygons_input)}")
        with st.form("parameters_form"):
            col = st.columns(4)
            with col[0]:
                top_n = st.number_input("Select Top n", min_value=1, max_value=15, value=5, step=1, key="top_n_input")
            with col[1]:
                k1 = st.number_input("Select K1", min_value=0, max_value=50, value=15, step=1, key="k1_input")
            with col[2]:
                k2 = st.number_input("Select K2", min_value=4, max_value=10, value=5, step=1, key="k2_input")
            with col[3]:
                stop_threshold = st.number_input("Select Stop Threshold", min_value=50, max_value=100, value=95, step=5, key="stop_threshold_input")
            vertex_angle_weight = st.slider("Vertex Angles Weight", min_value=0.0, max_value=1.0, value=0.5, step=0.1, key='pvp_vertex_angle_weight')
            with st.popover("Documents"):
                st.markdown("""
                            - **Top n**: The number of top-scoring polygons to be returned in the results.

                            - **K1**: The tolerance angle in degrees for comparing polygon orientations.

                            - **K2**: The minimum number of consecutive vertices that need to match between two polygons.

                            - **Stop Threshold**: The similarity threshold (in percentage) at which the search will stop if a polygon with similarity greater than or equal to this value is found.

                            - **Vertex Angle Weight**: is a tuning parameter that allows you to adjust the relative importance of vertex angles compared to polygon angles in the similarity calculation.
                            """)
            search_button = st.form_submit_button("Search")
    if (search_button and image2 is not None) or (search_sample and image2 is not None):
        if st.session_state.input_map:
            print(f"Number of polygons input: {len(list_polygons_input)}")
            print(f"Number of polygons in map: {len(st.session_state.list_polygons_map)}")
            result, result_img = find_polygon(st.session_state.input_map.copy(),
                                              list_polygons_input,
                                              st.session_state.list_polygons_map,
                                              top_n=top_n,
                                              stop_threshold=stop_threshold,
                                              k1=k1,
                                              k2=k2,
                                              vertex_angle_weight=vertex_angle_weight)
            centroid_input, sorted_vertices_input, vertex_angles_input, sorted_angles_input, polygon_angles_input = calculate_angles(list_polygons_input[0][0])
            fig_input, polygon_angles_input, vertex_angles_input = plot_polygon_and_image(list_polygons_input[0][0], centroid_input, vertex_angles_input, sorted_angles_input, polygon_angles_input)
            st.image(result_img)

            with st.expander("Detailed Results"):
                for i, (polygon1, polygon2, similarity, vertices, centroid) in enumerate(result[:top_n]):
                    ui.badges([(f"Similarity Result {i+1}: {similarity:.2f}%", 'primary')], key=f"similarity_badge_{i}")
                    # st.write(f"Similarity Result {i+1}: {similarity:.2f}%")
                    col1, col2 = st.columns(2)

                    with col1:
                        st.pyplot(fig_input)
                        st.write("Input Polygon")
                        ui.badges([(f"Vertex angles:", 'secondary')], key=f"vertex_angles_input{i}")
                        st.write(vertex_angles_input)
                        ui.badges([(f"Polygon angles:", 'secondary')], key=f"polygon_angles_input{i}")
                        st.write(polygon_angles_input)

                    with col2:
                        centroid2, sorted_vertices2, vertex_angles2, sorted_angles2, polygon_angles2 = calculate_angles(polygon2[0])
                        fig2, polygon_angles2, vertex_angles2 = plot_polygon_and_image(polygon2[0], centroid2, vertex_angles2, sorted_angles2, polygon_angles2)
                        st.pyplot(fig2)
                        st.write(f"Similar Polygon {i+1}")
                        ui.badges([(f"Vertex angles:", 'secondary')], key=f"vertex_angles_similar{i}")
                        st.write(vertex_angles2)
                        ui.badges([(f"Polygon angles:", 'secondary')], key=f"polygon_angles_similar{i}")
                        st.write(polygon_angles2)

                    st.write("---")
        else:
            print(f"Number of polygons input: {len(list_polygons_input)}")
            print(f"Number of polygons in map: {len(st.session_state.list_polygons_map)}")
            result, result_img = find_polygon(st.session_state.input_map.copy() if st.session_state.input_map else st.session_state.cache_map.copy(),
                                              list_polygons_input,
                                              st.session_state.list_polygons_map if st.session_state.list_polygons_map else st.session_state.list_polygons_cache,
                                              top_n=top_n,
                                              stop_threshold=stop_threshold,
                                              k1=k1,
                                              k2=k2,
                                              vertex_angle_weight=vertex_angle_weight)

            centroid_input, sorted_vertices_input, vertex_angles_input, sorted_angles_input, polygon_angles_input = calculate_angles(list_polygons_input[0][0])
            fig_input, polygon_angles_input, vertex_angles_input = plot_polygon_and_image(list_polygons_input[0][0], centroid_input, vertex_angles_input, sorted_angles_input, polygon_angles_input)
            # st.pyplot(result_img)
            st.image(result_img)
            with st.expander("Detailed Results"):
                for i, (polygon1, polygon2, similarity, vertices, centroid) in enumerate(result[:top_n]):
                    ui.badges([(f"Similarity Result {i+1}: {similarity:.2f}%", 'primary')], key=f"similarity_badge_{i}")
                    # st.write(f"Similarity Result {i+1}: {similarity:.2f}%")
                    col1, col2 = st.columns(2)

                    with col1:
                        st.pyplot(fig_input)
                        st.write("Input Polygon")
                        ui.badges([(f"Vertex angles:", 'secondary')], key=f"vertex_angles_input{i}")
                        st.write(vertex_angles_input)
                        ui.badges([(f"Polygon angles:", 'secondary')], key=f"polygon_angles_input{i}")
                        st.write(polygon_angles_input)

                    with col2:
                        centroid2, sorted_vertices2, vertex_angles2, sorted_angles2, polygon_angles2 = calculate_angles(polygon2[0])
                        fig2, polygon_angles2, vertex_angles2 = plot_polygon_and_image(polygon2[0], centroid2, vertex_angles2, sorted_angles2, polygon_angles2)
                        st.pyplot(fig2)
                        st.write(f"Similar Polygon {i+1}")
                        value_similar = np.concatenate([vertex_angles2, polygon_angles2]).tolist()
                        ui.badges([(f"Vertex angles:", 'secondary')], key=f"vertex_angles_similar{i}")
                        st.write(vertex_angles2)
                        ui.badges([(f"Polygon angles:", 'secondary')], key=f"polygon_angles_similar{i}")
                        st.write(polygon_angles2)


                    st.write("---")
    elif search_button and image is None:
        search_error.error("Please upload an image to search")

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
    footer()
