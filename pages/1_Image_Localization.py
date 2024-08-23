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
import streamlit_shadcn_ui as ui
from htbuilder import HtmlElement, div, ul, li, br, hr, a, p, img, styles, classes, fonts
from htbuilder.units import percent, px
from htbuilder.funcs import rgba, rgb
import pandas as pd
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from keras.models import load_model
import tensorflow as tf
st.set_page_config(page_title="Starwar Project", page_icon="🌏")

def read_image_2(uploaded_file):
    if uploaded_file is not None:
        uploaded_file.seek(0)
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        return image
    return None

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
# @st.cache_resource
# def load_super_resolution_model(model_path, scale):
#     sr = cv2.dnn_superres.DnnSuperResImpl_create()
#     sr.readModel(model_path)
#     sr.setModel("fsrcnn", scale)
#     return sr
# C:\Users\VTOS\Desktop\STARWAR\models\polygon_classification\mobilenet\best_mobinet_polygon_classification.tflite
# Paths to the models
current_dir = os.path.dirname(__file__)
polygon_model_path = os.path.join(current_dir[:-6], "models/polygon_classification/mobilenet/", "best_mobinet_polygon_classification.h5")
sr_model_path = os.path.join(current_dir[:-6], "models/upscale/", "FSRCNN-small_x4.pb")
interpreter_path = os.path.join(current_dir[:-6], "models/polygon_classification/mobilenet/", "best_mobinet_polygon_classification.tflite")

# Load the models
model = load_cached_model(polygon_model_path)
# sr = load_super_resolution_model(sr_model_path, 4)
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

    # Đọc tấm hình
    # image = cv2.imread(image_path)
    # if image.shape[2] == 4:
    #     image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    # Kiểm tra nếu tấm hình không tồn tại
    if image is None:
        print(f"Could not read the image")
        return None

    windows = []  # Danh sách để lưu trữ các cửa sổ

    if image.shape[0] <= window_size[1] or image.shape[1] <= window_size[0]:
        print("Size of image < window size")
        return None

    # Duyệt qua tấm hình và dịch chuyển cửa sổ
    for y in range(0, image.shape[0] - window_size[1] + 1, stride):
        row = []  # Dòng của bảng để lưu các cửa sổ
        for x in range(0, image.shape[1] - window_size[0] + 1, stride):
            # Lấy ra cửa sổ từ tấm hình
            window = image[y:y + window_size[1], x:x + window_size[0]]

            # Upscale the window using the sr.upsample() method
            upscaled_window = sr.upsample(window)

            # Thêm cửa sổ đã upscale vào danh sách
            row.append(upscaled_window)

        # Thêm dòng vào bảng
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
    st.sidebar.link_button("Download sample map", "https://drive.google.com/file/d/1x610cvzhqdc9s5MoJKkWmY-ljHKQ5Ud3/view?usp=sharing")
    uploaded_file = st.sidebar.file_uploader("Upload an map image", type=["jpg", "jpeg", "png"], key="map_uploader")
    if uploaded_file is not None:
        # Đọc tấm hình từ file tải lên
        image_plot = Image.open(uploaded_file)
        image_plot = np.array(image_plot)
        image = read_image_2(uploaded_file)
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        # Thiết lập các thông số cho cửa sổ
        window_size = st.sidebar.number_input("Windows Size (px)", min_value=10, max_value=5000, value=500, step=10)
        window_size = (window_size, window_size)
        stride = st.sidebar.number_input("Stride (px)", min_value=10, max_value=3000, value=300, step=10)
        upscale = st.sidebar.checkbox("Upscale")
        # Nút bấm để thực hiện cắt ảnh
        if st.sidebar.button("Create Search Windows"):
            if upscale:
                # st.session_state.windows = detect_windows_and_upscale(image, window_size, stride, sr)
                st.session_state.windows = detect_windows(image, window_size, stride)
            else:
                st.session_state.windows = detect_windows(image, window_size, stride)
        if st.session_state.windows:
            display_windows(st.session_state.windows)
    search_error = st.empty()
    with st.expander("Upload Images", expanded=True):
        col = st.columns(3)
        with col[0]:
            top_n = st.number_input("Select Top n", min_value=1, max_value=10, value=5, step=1, key="top_n_input")
            min_similarity = st.number_input("Select Min Similarity", min_value=50, max_value=100, value=80, step=5, key="min_similarity_input")
        with col[1]:
            k1 = st.number_input("Select K1", min_value=0, max_value=50, value=10, step=5, key="k1_input")
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
            
            image2 = read_image_2(uploaded_image)
            if image2.shape[2] == 4:
                image2 = cv2.cvtColor(image2, cv2.COLOR_BGRA2BGR)
            # image2 = sr.upsample(image2)
        search_button = st.button("Search")
    if search_button and image2 is not None:
        # So sánh ảnh tìm kiếm với các cửa sổ trong ảnh tải lên
        results = compare_image(image2, st.session_state.windows, 
                                window_step=window_step, 
                                min_similarity=min_similarity, 
                                min_similar_ratio=min_similar_ratio,
                                k1=k1,
                                k2=k2, 
                                interpreter=interpreter)
        filtered_results = [item for item in results if item['score'] > 0]
        top_results = heapq.nlargest(top_n, filtered_results, key=lambda x: x['score'])
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
        # Hiển thị kết quả
        grouped_windows = group_adjacent_windows(top_results, window_size, stride)
        bounding_boxes = [get_bounding_box(group, stride, window_size) for group in grouped_windows]
        image_with_boxes = draw_bounding_boxes(image_plot, bounding_boxes, colors)
        image_with_boxes = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)
        st.image(image_with_boxes, caption="Search Results")
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