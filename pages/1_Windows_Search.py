import streamlit as st
import cv2
import numpy as np
from PIL import Image
import heapq
# from process_image.compare_polygon import
# from process_image.compare_cluster import process_image_and_find_similar_cluster, process_cluster, display_results, compare_images_knn, plot_matching_clusters_knn
from process_image.windows_search import *
import io, base64
import os
import matplotlib.pyplot as plt
from streamlit_image_select import image_select
import streamlit_shadcn_ui as ui
from htbuilder import HtmlElement, div, ul, li, br, hr, a, p, img, styles, classes, fonts
from htbuilder.units import percent, px
from htbuilder.funcs import rgba, rgb
from keras.models import load_model
import tensorflow as tf
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
st.set_page_config(page_title="Starwar Project", page_icon="🌏")

def read_image_2(uploaded_file):
    if uploaded_file is not None:
        uploaded_file.seek(0)
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        return image
    return None

@st.cache_data
def load_sample_image(image_path):
    return cv2.imread(image_path)

@st.cache_resource
def load_tflite_model(model_path):
    # Load mô hình TFLite
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# Cache the model loading process
@st.cache_resource
def load_cached_model(model_path):
    model = load_model(model_path)
    print("Model loaded successfully!")
    return model

# Cache the Super Resolution model
@st.cache_resource
def load_super_resolution_model(model_path, scale):
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(model_path)
    sr.setModel("fsrcnn", scale)
    return sr

# C:\Users\VTOS\Desktop\STARWAR\models\polygon_classification\mobilenet\best_mobinet_polygon_classification.tflite
# Paths to the models
current_dir = os.path.dirname(__file__)
polygon_model_path = os.path.join(current_dir[:-6], "models/polygon_classification/mobilenet/", "best_mobinet_polygon_classification.h5")
sr_model_path = os.path.join(current_dir[:-6], "models/upscale/", "FSRCNN-small_x2.pb")
interpreter_path = os.path.join(current_dir[:-6], "models/polygon_classification/mobilenet/", "best_mobinet_polygon_classification.tflite")
sample_map_path = os.path.join(current_dir[:-6], "images/", "map_blue_low.png")
sample_input_1_path = os.path.join(current_dir[:-6], "images/", "sample_input_1.png")
sample_input_2_path = os.path.join(current_dir[:-6], "images/", "sample_input_2.png")
sample_input_3_path = os.path.join(current_dir[:-6], "images/", "sample_input_3.png")
sample_input_4_path = os.path.join(current_dir[:-6], "images/", "sample_input_4.png")
sample_input_5_path = os.path.join(current_dir[:-6], "images/", "sample_input_5.png")
# Load the models
model = load_cached_model(polygon_model_path)
sr = load_super_resolution_model(sr_model_path, 2)
interpreter = load_tflite_model(interpreter_path)


def detect_windows(image, window_size, stride):
    windows = []  # Danh sách để lưu trữ các cửa sổ

    if image.shape[0] <= window_size[1] or image.shape[1] <= window_size[0]:
        st.warning("Size of image < window size")
        return None

    # Duyệt qua tấm hình và dịch chuyển cửa sổ
    for y in range(0, image.shape[0] - window_size[1] + 1, stride):
        row = []  # Dòng của bảng để lưu các cửa sổ
        for x in range(0, image.shape[1] - window_size[0] + 1, stride):
            # Lấy ra cửa sổ từ tấm hình
            window = image[y:y + window_size[1], x:x + window_size[0]]

            # Thêm cửa sổ vào danh sách
            row.append(window)

        # Thêm dòng vào bảng
        windows.append(row)

    return windows

def detect_windows_and_upscale(image, window_size, stride, sr):
    if image is None:
        print(f"Could not read the image")
        return None

    # Chuyển đổi ảnh sang BGR nếu nó là BGRA
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    windows = []

    if image.shape[0] <= window_size[1] or image.shape[1] <= window_size[0]:
        print("Size of image < window size")
        return None

    for y in range(0, image.shape[0] - window_size[1] + 1, stride):
        row = []
        for x in range(0, image.shape[1] - window_size[0] + 1, stride):
            window = image[y:y + window_size[1], x:x + window_size[0]]

            # Đảm bảo window có 3 kênh màu
            if window.shape[2] != 3:
                window = cv2.cvtColor(window, cv2.COLOR_BGRA2BGR)

            # Thử catch lỗi khi upsample
            try:
                upscaled_window = sr.upsample(window)
            except Exception as e:
                print(f"Error upscaling window at ({x}, {y}): {e}")
                upscaled_window = window  # Sử dụng window gốc nếu upscale thất bại

            row.append(upscaled_window)
        windows.append(row)

    return windows

def display_windows(windows):
    if windows is None:
        st.warning("No windows to display.")
        return

    # Số hàng và cột của bảng
    num_rows = len(windows)
    num_cols = len(windows[0])

    # Hiển thị các cửa sổ dưới dạng bảng
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2, num_rows * 2))
    for i in range(num_rows):
        for j in range(num_cols):
            axs[i, j].imshow(cv2.cvtColor(windows[i][j], cv2.COLOR_BGR2RGB))
            axs[i, j].axis('off')
    with st.expander("Windows:", expanded=True):
        st.pyplot(fig)

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

@st.cache_data
def preprocess_sample_windows():
    image_path = sample_map_path
    image = cv2.imread(image_path)
    image_plot = np.array(image)
    if image is None:
        st.error(f"Could not read image from {image_path}")
        return None, None
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    try:
        windows = detect_windows(image, (500, 500), 300)
    except Exception as e:
        st.error(f"Error in detect_windows_and_upscale: {e}")
        windows = detect_windows(image, (500, 500), 300)  # Fallback to non-upscaled version
    return windows, image_plot

SAMPLE_WINDOWS, IMAGE_PLOT = preprocess_sample_windows()

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
    if "windows" not in st.session_state:
        st.session_state.windows = []
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
        display_windows(SAMPLE_WINDOWS)
    # if sample_input:
    #     st.session_state.sample_windows = SAMPLE_WINDOWS
    #     sample_image = cv2.imread(sample_input)
    #     if sample_image.shape[2] == 4:
    #         sample_image = cv2.cvtColor(sample_image, cv2.COLOR_BGRA2BGR)
    #     image_plot = IMAGE_PLOT
    #     image2 = sample_image
    #     window_size = (500,500)
    #     stride = 300
    if sample_input:
        st.session_state.sample_windows = SAMPLE_WINDOWS
        sample_image = load_sample_image(sample_input)
        if sample_image.shape[2] == 4:
            sample_image = cv2.cvtColor(sample_image, cv2.COLOR_BGRA2BGR)
        image_plot = IMAGE_PLOT
        image2 = sample_image
        window_size = (500,500)
        stride = 300
    st.sidebar.link_button("Download sample map", "https://drive.google.com/file/d/1x610cvzhqdc9s5MoJKkWmY-ljHKQ5Ud3/view?usp=sharing")
    uploaded_file = st.sidebar.file_uploader("Upload an map image", type=["jpg", "jpeg", "png"], key="map_uploader")
    if uploaded_file is not None:
        # Đọc tấm hình từ file tải lên
        st.session_state.windows = []
        image_plot = Image.open(uploaded_file)
        image_plot = np.array(image_plot)
        image = read_image_2(uploaded_file)
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        # Thiết lập các thông số cho cửa sổ
        windows_form = st.sidebar.form("windows_form")
        with windows_form:
            window_size = windows_form.number_input("Windows Size (px)", min_value=10, max_value=5000, value=500, step=10)
            window_size = (window_size, window_size)
            stride = windows_form.number_input("Stride (px)", min_value=10, max_value=3000, value=300, step=10)
            upscale = windows_form.checkbox("Upscale")
            create_windows_button = windows_form.form_submit_button("Create Search Windows")
        # Nút bấm để thực hiện cắt ảnh
        if create_windows_button:
            if upscale:
                st.session_state.windows = detect_windows_and_upscale(image, window_size, stride, sr)
                # st.session_state.windows = detect_windows(image, window_size, stride)
            else:
                st.session_state.windows = detect_windows(image, window_size, stride)
    if st.session_state.windows:
        display_windows(st.session_state.windows)
    search_error = st.empty()
    merge_option = st.toggle("Merge", value=False)
    if not merge_option:
        with st.expander("Upload Images", expanded=True):
            with st.form("parameters_form"):
                col = st.columns(3)
                with col[0]:
                    top_n = st.number_input("Select Top n", min_value=1, max_value=20, value=5, step=1, key="top_n_input")
                    min_similarity = st.number_input("Select Min Similarity", min_value=50, max_value=100, value=80, step=5, key="min_similarity_input")
                with col[1]:
                    k1 = st.number_input("Select K1", min_value=0, max_value=50, value=15, step=5, key="k1_input")
                    min_similar_ratio = st.number_input("Select Min Similar Polygon Ratio", min_value=0.1, max_value=1.0, value=0.5, step=0.1, key="min_similar_ratio_input")
                with col[2]:
                    k2 = st.number_input("Select K2", min_value=4, max_value=10, value=5, step=1, key="k2_input")
                    window_step = st.number_input("Select Window Step", min_value=1, max_value=5, value=2, step=1, key="window_step_input")
                with st.popover("Documents"):
                    st.markdown("""
                                - **Top_n**: The number of top scoring windows to be returned. Adjacent windows with similar scores will be merged if they are contiguous. Windows with a score of 0 will not be illustrated.

                                - **K1**: The tolerance angle (degrees).

                                - **K2**: The minimum subsequence length.

                                - **Min Similarity**: The minimum percentage similarity required to determine that a pair of polygons matches.

                                - **Min Similar Polygon Ratio**: The minimum polygon similarity ratio for early stopping.

                                - **Window Step**: The step size for window searching (Smaller steps lead to higher accuracy but take more time to search).
                                """)
                uploaded_image = st.file_uploader("Upload an image to search", type=["jpg", "jpeg", "png"], key="image_uploader")
                if uploaded_image is not None:
                    # image2 = Image.open(uploaded_image)
                    # image2 = np.array(image2)
                    buffer_distance = st.number_input("Select Buffer Distance", min_value=0.0, max_value=5.0, value=1.0, step=0.1, key="buffer_distance_input")
                    min_area, max_area = st.select_slider("Select Area Range", options=[0, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 10000, 25000, 50000, 100000], value=(0, 10000))
                    image2 = read_image_2(uploaded_image)
                    if image2.shape[2] == 4:
                        image2 = cv2.cvtColor(image2, cv2.COLOR_BGRA2BGR)
                    # image2 = sr.upsample(image2)
                search_button = st.form_submit_button("Search")
    if merge_option:
        upload_info = st.empty()
        with st.expander("Upload Images", expanded=True):
            uploaded_image = st.file_uploader("Upload an image to search", type=["jpg", "jpeg", "png"], key="image_uploader")
            if uploaded_image is not None:
                image2 = read_image_2(uploaded_image)
                if image2.shape[2] == 4:
                    image2 = cv2.cvtColor(image2, cv2.COLOR_BGRA2BGR)
            buffer_distance = st.number_input("Select Buffer Distance", min_value=0.0, max_value=5.0, value=1.0, step=0.1, key="buffer_distance_input")
            min_area, max_area = st.select_slider("Select Area Range", options=[0, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 10000, 25000, 50000, 100000], value=(0, 10000))

            if (min_area and max_area) :
            # Sử dụng st.empty() để tạo một vùng có thể cập nhật
                image2, num_polygons_input = update_and_display_polygons(buffer_distance, min_area, max_area, image2, upload_info)
                if image2:
                    ui.badges([(f"Number of polygons input: {num_polygons_input}",'secondary')], key=f"badge_input")
                    polygon_plot = st.empty()
                    polygon_plot.image(image2)
                    #Lấy vertices của polygon (ảnh black-white)
                    list_polygons_input = listing_polygon_input(image2)
                    print(list_polygons_input)
                    print(80 * "_")
                    # print(f"list_polygons_input: {list_polygons_input}")
                    if (len(list_polygons_input) > 0):
                        print(f"input vertices: {list_polygons_input[0][0]}")
                    if len(list_polygons_input) == 0:
                        upload_info.info("Please upload an image with polygons or adjust the buffer distance and area range")
                    print(f"Number of polygons input.: {len(list_polygons_input)}")
            with st.form("parameters_form"):
                col = st.columns(3)
                with col[0]:
                    top_n = st.number_input("Select Top n", min_value=1, max_value=10, value=5, step=1, key="top_n_input")
                    min_similarity = st.number_input("Select Min Similarity", min_value=50, max_value=100, value=80, step=5, key="min_similarity_input")
                with col[1]:
                    k1 = st.number_input("Select K1", min_value=0, max_value=50, value=15, step=5, key="k1_input")
                    min_similar_ratio = st.number_input("Select Min Similar Polygon Ratio", min_value=0.1, max_value=1.0, value=0.5, step=0.1, key="min_similar_ratio_input")
                with col[2]:
                    k2 = st.number_input("Select K2", min_value=4, max_value=10, value=5, step=1, key="k2_input")
                    window_step = st.number_input("Select Window Step", min_value=1, max_value=5, value=2, step=1, key="window_step_input")
                with st.popover("Documents"):
                    st.markdown("""
                                - **Top_n**: The number of top scoring windows to be returned. Adjacent windows with similar scores will be merged if they are contiguous. Windows with a score of 0 will not be illustrated.

                                - **K1**: The tolerance angle (degrees).

                                - **K2**: The minimum subsequence length.

                                - **Min Similarity**: The minimum percentage similarity required to determine that a pair of polygons matches.

                                - **Min Similar Polygon Ratio**: The minimum polygon similarity ratio for early stopping.

                                - **Window Step**: The step size for window searching (Smaller steps lead to higher accuracy but take more time to search).
                                """)
                search_button = st.form_submit_button("Search")
        # search_button = st.button("Search")
    if (search_button and image2 is not None) or (search_sample and image2 is not None):
        # So sánh ảnh tìm kiếm với các cửa sổ trong ảnh tải lên
        if st.session_state.windows:
            results = compare_image(image2, st.session_state.windows,
                                    window_step=window_step,
                                    min_similarity=min_similarity,
                                    min_similar_ratio=min_similar_ratio,
                                    k1=k1,
                                    k2=k2,
                                    interpreter=interpreter
                                    )
        else:
            results = compare_image(image2, st.session_state.sample_windows,
                                    window_step=window_step,
                                    min_similarity=min_similarity,
                                    min_similar_ratio=min_similar_ratio,
                                    k1=k1,
                                    k2=k2,
                                    interpreter=interpreter
                                    )
        filtered_results = [item for item in results if item['score'] > 0]
        top_results = heapq.nlargest(top_n, filtered_results, key=lambda x: x['score'])
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
        # Hiển thị kết quả
        grouped_windows = group_adjacent_windows(top_results, window_size, stride)
        bounding_boxes = [get_bounding_box(group, stride, window_size) for group in grouped_windows]
        image_with_boxes = draw_bounding_boxes(image_plot, bounding_boxes, colors)
        image_with_boxes = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)
        st.image(image_with_boxes, caption="Search Results")
        total_group = len(grouped_windows)
        if total_group != 0:
            for i in range(0, total_group, 3):
                row = st.columns(3)
                for j in range(3):
                    if i + j < total_group:
                        group = grouped_windows[i + j]
                        total_score = sum(w['score'] for w in group)
                        total_matched = sum(w['matched_polygons'] for w in group)
                        polygon_counts = {
                            'special_polygon': sum(w['polygon_type_counts']['special_polygon'] for w in group),
                            'superhero_polygon': sum(w['polygon_type_counts']['superhero_polygon'] for w in group)
                        }
                        with row[j]:
                            ui.metric_card(
                                title=f"Group {i+j+1}",
                                content=f"Score: {total_score}",
                                description=f"Windows in group: {len(group)}  Matched Polygons: {total_matched}  Special Polygons: {polygon_counts['special_polygon']}  Superhero Polygons: {polygon_counts['superhero_polygon']}",
                                key=f"card{i+j}"
                            )
                    else:
                        # Thêm một cột trống nếu không có đủ nhóm để điền vào hàng
                        with row[j]:
                            st.empty()
        else:
            st.info("No similar window found. Please adjust the parameters for better results!")
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
