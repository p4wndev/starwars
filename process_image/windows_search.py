from process_image.compare_polygon import *
from process_image.process_polygon import *
import tensorflow as tf

HEIGHT = 64
WIDTH = 64
N_CHANNELS = 3
class_names = ['normal_polygon', 'special_polygon', 'superhero_polygon']

def process_image_and_find_similar_polygons_2(polygon1, polygon2, k1=15, k2=3):
    # Tính toán các góc cho polygon1
    centroid1, sorted_vertices1, vertex_angles1, sorted_angles1, polygon_angles1 = calculate_angles(polygon1)
    # Tính toán các góc cho polygon2
    centroid2, sorted_vertices2, vertex_angles2, sorted_angles2, polygon_angles2 = calculate_angles(polygon2)

    # Tính độ tương đồng
    similarity1 = calculate_percentage_match(vertex_angles1, vertex_angles2, k1, k2)
    similarity2 = calculate_percentage_match(polygon_angles1, polygon_angles2, k1, k2)

    similarity = (similarity1 + similarity2) / 2
    return similarity
def classify_image_tflite(image, interpreter):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    image = cv2.resize(image, (WIDTH, HEIGHT))
    image = np.array(image, dtype="float32") / 255.0
    image = np.expand_dims(image, axis=0)

    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    return class_names[int(np.argmax(output_data))]


# def compare_image(input_image, image_set, window_step, min_similarity, min_similar_ratio, k1,k2, interpreter):
#     polygons_input = extract_polygons(input_image)
#     input_polygon_count = len(polygons_input)
#     results = []
#     max_score = 0
#     best_window = None

#     for idx in range(0, len(image_set), window_step):  # Nhảy cóc qua các hàng, mỗi lần nhảy window_step cửa sổ
#         for col in range(0, len(image_set[0]), window_step):  # Nhảy cóc qua các cột, mỗi lần nhảy window_step cửa sổ
#             image = image_set[idx][col]
#             polygons_image = extract_polygons(image)
#             matched_polygons = 0
#             total_polygons = len(polygons_image)
#             matched_polygon_types = []
#             matched_polygon_details = []

#             used_polygons_input = set()
#             used_polygons_image = set()

#             early_stop = False

#             for i, (vertices_input, cropped_polygon_input) in enumerate(polygons_input):
#                 if i in used_polygons_input:
#                     continue

#                 if len(vertices_input) <= 4:
#                     continue
#                 input_classification = classify_image_tflite(cropped_polygon_input, interpreter)

#                 if input_classification == 'normal_polygon':
#                     continue

#                 for j, (vertices_image, cropped_polygon_image) in enumerate(polygons_image):
#                     if j in used_polygons_image:
#                         continue
#                     if len(vertices_image) <= 4:
#                         continue
#                     similarity_scores = process_image_and_find_similar_polygons_2(vertices_input, vertices_image, k1, k2)
#                     if similarity_scores >= min_similarity:
#                         matched_polygons += 1
#                         matched_polygon_types.append(input_classification)
#                         # matched_polygon_types.append('special')
#                         used_polygons_input.add(i)
#                         used_polygons_image.add(j)
#                         matched_polygon_details.append({
#                             'type': input_classification,
#                             # 'type': 'special',
#                             'input_vertices': vertices_input,
#                             'matched_vertices': vertices_image,
#                             'similarity_score': similarity_scores
#                         })

#                         # if similarity_scores == 100 or matched_polygons >= 0.5 * input_polygon_count:
#                         if matched_polygons >= min_similar_ratio * input_polygon_count:
#                             early_stop = True
#                             break

#                 if early_stop:
#                     break

#             polygon_type_counts = {
#                 'special_polygon': matched_polygon_types.count('special_polygon'),
#                 'superhero_polygon': matched_polygon_types.count('superhero_polygon')
#             }

#             score = (polygon_type_counts['special_polygon'] * 1 +
#                      polygon_type_counts['superhero_polygon'] * 3) #

#             if score > max_score:
#                 max_score = score
#                 best_window = {'row': idx, 'col': col, 'score': score, 'matched_polygons': matched_polygons,
#                                'matched_polygon_details': matched_polygon_details, 'polygon_type_counts': polygon_type_counts}

#             results.append({
#                 'row': idx,
#                 'col': col,
#                 'matched_polygons': matched_polygons,
#                 'total_polygons_input': input_polygon_count,
#                 'total_polygons_compared': total_polygons,
#                 'polygon_type_counts': polygon_type_counts,
#                 'score': score,
#                 'matched_polygon_details': matched_polygon_details
#             })

#             if early_stop:
#                 break
#         if early_stop:
#             break

#     if best_window:
#         row, col = best_window['row'], best_window['col']
#         # directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
#         directions = [
#             (-1, -1), (-1, 0), (-1, 1),  # Hàng trên: trái, giữa, phải
#             (0, -1),         (0, 1),      # Hàng giữa: trái, phải (trung tâm là (0, 0))
#             (1, -1), (1, 0), (1, 1)       # Hàng dưới: trái, giữa, phải
#         ]

#         for drow, dcol in directions:
#             new_row, new_col = row + drow, col + dcol
#             if 0 <= new_row < len(image_set) and 0 <= new_col < len(image_set[0]):
#                 image = image_set[new_row][new_col]
#                 polygons_image = extract_polygons(image)
#                 matched_polygons = 0
#                 matched_polygon_types = []
#                 matched_polygon_details = []

#                 used_polygons_input = set()
#                 used_polygons_image = set()

#                 for i, (vertices_input, cropped_polygon_input) in enumerate(polygons_input):
#                     if i in used_polygons_input:
#                         continue

#                     input_classification = classify_image_tflite(cropped_polygon_input, interpreter)

#                     if input_classification == 'normal_polygon':
#                         continue

#                     for j, (vertices_image, cropped_polygon_image) in enumerate(polygons_image):
#                         if j in used_polygons_image:
#                             continue
#                         similarity_scores = process_image_and_find_similar_polygons_2(vertices_input, vertices_image, k1, k2)
#                         if similarity_scores >= min_similarity:
#                             matched_polygons += 1
#                             matched_polygon_types.append(input_classification)
#                             used_polygons_input.add(i)
#                             used_polygons_image.add(j)
#                             matched_polygon_details.append({
#                                 'type': input_classification,
#                                 'input_vertices': vertices_input,
#                                 'matched_vertices': vertices_image,
#                                 'similarity_score': similarity_scores
#                             })

#                             # if similarity_scores == 100 or matched_polygons >= 0.5 * input_polygon_count:
#                             if matched_polygons >= min_similar_ratio * input_polygon_count:

#                                 early_stop = True
#                                 break

#                     if early_stop:
#                         break

#                 polygon_type_counts = {
#                     'special_polygon': matched_polygon_types.count('special_polygon'),
#                     'superhero_polygon': matched_polygon_types.count('superhero_polygon')
#                 }

#                 score = (polygon_type_counts['special_polygon'] * 1 +
#                          polygon_type_counts['superhero_polygon'] * 3)

#                 results.append({
#                     'row': new_row,
#                     'col': new_col,
#                     'matched_polygons': matched_polygons,
#                     'total_polygons_input': input_polygon_count,
#                     'total_polygons_compared': len(polygons_image),
#                     'polygon_type_counts': polygon_type_counts,
#                     'score': score,
#                     'matched_polygon_details': matched_polygon_details
#                 })

#     return results

#Top 3 best windows
def compare_image(input_image, image_set, window_step, min_similarity, min_similar_ratio, k1, k2, interpreter):
    polygons_input = extract_polygons(input_image)
    input_polygon_count = len(polygons_input)
    results = []
    top_windows = []

    for idx in range(0, len(image_set), window_step):
        for col in range(0, len(image_set[0]), window_step):
            image = image_set[idx][col]
            polygons_image = extract_polygons(image)
            matched_polygons = 0
            total_polygons = len(polygons_image)
            matched_polygon_types = []
            matched_polygon_details = []

            used_polygons_input = set()
            used_polygons_image = set()

            early_stop = False

            for i, (vertices_input, cropped_polygon_input) in enumerate(polygons_input):
                if i in used_polygons_input or len(vertices_input) <= 4:
                    continue

                input_classification = classify_image_tflite(cropped_polygon_input, interpreter)

                if input_classification == 'normal_polygon':
                    continue

                for j, (vertices_image, cropped_polygon_image) in enumerate(polygons_image):
                    if j in used_polygons_image or len(vertices_image) <= 4:
                        continue

                    similarity_scores = process_image_and_find_similar_polygons_2(vertices_input, vertices_image, k1, k2)
                    if similarity_scores >= min_similarity:
                        matched_polygons += 1
                        matched_polygon_types.append(input_classification)
                        used_polygons_input.add(i)
                        used_polygons_image.add(j)
                        matched_polygon_details.append({
                            'type': input_classification,
                            'input_vertices': vertices_input,
                            'matched_vertices': vertices_image,
                            'similarity_score': similarity_scores
                        })

                        if matched_polygons >= min_similar_ratio * input_polygon_count:
                            early_stop = True
                            break

                if early_stop:
                    break

            polygon_type_counts = {
                'special_polygon': matched_polygon_types.count('special_polygon'),
                'superhero_polygon': matched_polygon_types.count('superhero_polygon')
            }

            score = (polygon_type_counts['special_polygon'] * 1 +
                     polygon_type_counts['superhero_polygon'] * 3)

            window_result = {
                'row': idx,
                'col': col,
                'matched_polygons': matched_polygons,
                'total_polygons_input': input_polygon_count,
                'total_polygons_compared': total_polygons,
                'polygon_type_counts': polygon_type_counts,
                'score': score,
                'matched_polygon_details': matched_polygon_details
            }

            results.append(window_result)

            # Update top_windows
            if len(top_windows) < 3:
                top_windows.append(window_result)
                top_windows.sort(key=lambda x: x['score'], reverse=True)
            elif score > top_windows[-1]['score']:
                top_windows[-1] = window_result
                top_windows.sort(key=lambda x: x['score'], reverse=True)

    # Process neighboring windows for top 3
    directions = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),         (0, 1),
        (1, -1), (1, 0), (1, 1)
    ]

    for best_window in top_windows:
        row, col = best_window['row'], best_window['col']

        for drow, dcol in directions:
            new_row, new_col = row + drow, col + dcol
            if 0 <= new_row < len(image_set) and 0 <= new_col < len(image_set[0]):
                image = image_set[new_row][new_col]
                polygons_image = extract_polygons(image)
                matched_polygons = 0
                matched_polygon_types = []
                matched_polygon_details = []

                used_polygons_input = set()
                used_polygons_image = set()

                for i, (vertices_input, cropped_polygon_input) in enumerate(polygons_input):
                    if i in used_polygons_input:
                        continue

                    input_classification = classify_image_tflite(cropped_polygon_input, interpreter)

                    if input_classification == 'normal_polygon':
                        continue

                    for j, (vertices_image, cropped_polygon_image) in enumerate(polygons_image):
                        if j in used_polygons_image:
                            continue
                        similarity_scores = process_image_and_find_similar_polygons_2(vertices_input, vertices_image, k1, k2)
                        if similarity_scores >= min_similarity:
                            matched_polygons += 1
                            matched_polygon_types.append(input_classification)
                            used_polygons_input.add(i)
                            used_polygons_image.add(j)
                            matched_polygon_details.append({
                                'type': input_classification,
                                'input_vertices': vertices_input,
                                'matched_vertices': vertices_image,
                                'similarity_score': similarity_scores
                            })

                            if matched_polygons >= min_similar_ratio * input_polygon_count:
                                break

                    if matched_polygons >= min_similar_ratio * input_polygon_count:
                        break

                polygon_type_counts = {
                    'special_polygon': matched_polygon_types.count('special_polygon'),
                    'superhero_polygon': matched_polygon_types.count('superhero_polygon')
                }

                score = (polygon_type_counts['special_polygon'] * 1 +
                         polygon_type_counts['superhero_polygon'] * 3)

                neighbor_result = {
                    'row': new_row,
                    'col': new_col,
                    'matched_polygons': matched_polygons,
                    'total_polygons_input': input_polygon_count,
                    'total_polygons_compared': len(polygons_image),
                    'polygon_type_counts': polygon_type_counts,
                    'score': score,
                    'matched_polygon_details': matched_polygon_details
                }

                results.append(neighbor_result)

    return results


# Các hàm còn lại không thay đổi
def draw_rectangles_on_image(image, positions, window_size, colors, top_n, scores):
    image_with_rectangles = image.copy()
    for i, ((x, y), color, score) in enumerate(zip(positions, colors, scores)):
        overlay = image_with_rectangles.copy()
        cv2.rectangle(overlay, (x, y), (x + window_size[0], y + window_size[1]), color, -1)
        cv2.addWeighted(overlay, 0.3, image_with_rectangles, 0.7, 0, image_with_rectangles)
        cv2.rectangle(image_with_rectangles, (x, y), (x + window_size[0], y + window_size[1]), color, 10)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"{i + 1} ({score})"
        textsize = cv2.getTextSize(text, font, 1, 2)[0]
        text_x = x + (window_size[0] - textsize[0]) // 2
        text_y = y + (window_size[1] + textsize[1]) // 2
        cv2.putText(image_with_rectangles, text, (text_x, text_y), font, 7, (0, 0, 0), 10, cv2.LINE_AA)

    return image_with_rectangles

def are_windows_adjacent(window1, window2, window_size, stride):
    x1, y1 = window1['col'] * stride, window1['row'] * stride
    x2, y2 = window2['col'] * stride, window2['row'] * stride

    return (abs(x1 - x2) <= window_size[0] and abs(y1 - y2) <= window_size[1])

def group_adjacent_windows(windows, window_size, stride):
    groups = []
    used = set()

    for i, window in enumerate(windows):
        if i in used:
            continue

        group = [window]
        used.add(i)

        for j, other_window in enumerate(windows):
            if j in used:
                continue

            if any(are_windows_adjacent(w, other_window, window_size, stride) for w in group):
                group.append(other_window)
                used.add(j)

        groups.append(group)

    return groups

def get_bounding_box(group, stride, window_size):
    min_x = min(w['col'] * stride for w in group)
    min_y = min(w['row'] * stride for w in group)
    max_x = max(w['col'] * stride + window_size[0] for w in group)
    max_y = max(w['row'] * stride + window_size[1] for w in group)
    return (min_x, min_y, max_x - min_x, max_y - min_y)

def draw_bounding_boxes(image, bounding_boxes, colors):
    image_with_boxes = image.copy()
    for i, (x, y, w, h) in enumerate(bounding_boxes):
        color = colors[i % len(colors)]
        cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), color, 10)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"{i + 1}"
        textsize = cv2.getTextSize(text, font, 1, 2)[0]
        text_x = x + (w - textsize[0]) // 2
        text_y = y + (h + textsize[1]) // 2
        cv2.putText(image_with_boxes, text, (text_x, text_y), font, 7, (0, 0, 0), 10, cv2.LINE_AA)

    return image_with_boxes
