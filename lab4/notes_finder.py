import cv2
import copy
import math


def prepare_image_to_processing(img):
    cimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(~cimg, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, -15)


def erode_dilate(img, x, y):
    img = cv2.erode(img, cv2.getStructuringElement(cv2.MORPH_CROSS, (x, y)))
    return cv2.dilate(img, cv2.getStructuringElement(cv2.MORPH_CROSS, (x, y)))


def extract_notes(vertical):
    verticalsize = int(vertical.shape[0] / 30)
    vertical_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (3, verticalsize))
    vertical = cv2.erode(vertical, vertical_structure)
    vertical = cv2.dilate(vertical, vertical_structure)
    vertical_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(verticalsize / 2) if int(verticalsize / 2) > 0 else 1))
    vertical = cv2.erode(vertical, vertical_structure)
    vertical = cv2.dilate(vertical, vertical_structure)
    return cv2.bitwise_not(vertical)


def extract_staff(horizontal):
    horizontalsize = int(horizontal.shape[1] / 30)
    horizontal_sturcture = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalsize, 1))
    horizontal = cv2.erode(horizontal, horizontal_sturcture, (-1, -1))
    return cv2.dilate(horizontal, horizontal_sturcture, (-1, -1))


def remove_key_sign(source, image):
    im2, contours, hierarchy = cv2.findContours(copy.deepcopy(source), cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
    most_left, others = find_most_left_shape(contours, hierarchy)
    most_left = close_contour_in_square(contours[most_left])
    remove_from_image(image, most_left, 255)
    for idx in others:
        remove_from_image(image, close_contour_in_square(contours[idx]), 255)
    removal_square = len(image) - 1, most_left[1], 0, most_left[3]
    remove_from_image(image, removal_square, 255)
    return most_left


def find_most_left_shape(contours, hierarchy):
    possible_contours = find_deepest_contours(hierarchy)

    most_left = close_contour_in_square(contours[0])
    most_left_index = possible_contours[0]
    on_most_left_level_vertically = []
    for i in possible_contours:
        square = close_contour_in_square(contours[i])
        if square[3] < most_left[3]:
            most_left = square
            most_left_index = i
    for i in possible_contours:
        square = close_contour_in_square(contours[i])
        if square[3] >= most_left[3] and square[1] <= most_left[1]:
            on_most_left_level_vertically.append(i)
    return most_left_index, on_most_left_level_vertically


def find_deepest_contours(hierarchy):
    independent_contours = []
    if hierarchy is None:
        return independent_contours

    for index, contour in enumerate(hierarchy[0]):
        if contour[3] == -1:
            independent_contours.append(index)
    return independent_contours


def close_contour_in_square(contour):
    x_max, x_min, y_max, y_min = contour[0][0][0], contour[0][0][0], contour[0][0][1], contour[0][0][1]
    for vertex in contour:
        x_max = vertex[0][0] if vertex[0][0] > x_max else x_max
        y_max = vertex[0][1] if vertex[0][1] > y_max else y_max
        x_min = vertex[0][0] if vertex[0][0] < x_min else x_min
        y_min = vertex[0][1] if vertex[0][1] < y_min else y_min
    return y_max, x_max, y_min, x_min


def remove_from_image(image, square, colour):
    for y in range(square[2], square[0]+1):
        for x in range(square[3], square[1]+1):
            image[y][x] = colour


def count_staff_height(source):
    top = None
    max_size = len(source) - 1
    y = 0
    while top is None:
        for x in source[y]:
            if x > 0:
                top = y
        y += 1
        if y > max_size:
            top = -1

    bottom = None
    y = 0
    while bottom is None:
        for x in range(0, len(source[y])):
            if source[max_size - y][x] > 0:
                bottom = max_size - y + 2
        y += 1
        if y > max_size:
            bottom = -1
    return abs(bottom - top)


def remove_metre_sign(source, img, height, staff_pos, q):
    im2, contours, hierarchy = cv2.findContours(source, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
    if hierarchy is not None and len(hierarchy) > 0:
        metre_sign_idx = find_metre_sign(contours, hierarchy, staff_pos, height, q)
        if metre_sign_idx is not None:
            box = close_contour_in_square(contours[metre_sign_idx])
            remove_from_image(img, (box[0],box[1], box[2], 0), 255)


def find_metre_sign(contours, hierarchy, staff_pos, height, q):
    possibilities = []
    for idx, contour in enumerate(contours):
        box = close_contour_in_square(contour)
        if abs(box[2] - staff_pos) < q and abs(box[0] - (staff_pos + height)) < q:
            possibilities.append(idx)
    metre = None
    if len(possibilities) == 1:
        metre = possibilities[0]

    if len(possibilities) > 1:
        most_left = possibilities[0]
        left_box = close_contour_in_square(contours[possibilities[0]])
        for idx in possibilities[1:]:
            box = close_contour_in_square(contours[idx])

            if box[3] < left_box[3]:
                most_left = idx
                left_box = box
        metre = most_left

    return metre


def count_possible_notes(notes, contours, image, staff_gap, q=0):
    max_square = 0, 0
    possible_notes = []
    count = 0
    boxes = []
    ids = []
    result_conturs = []
    for idx in notes:
        square = close_contour_in_square(contours[idx])
        length = square_sizes(square)
        size_ratio = max(length[0], length[1]) / min(length[0], length[1])
        if (length[0] > max_square[0] or length[1] > max_square[1]) and size_ratio < 2 and abs(length[1] - staff_gap) < q:
            max_square = length
    for idx in notes:
        square = close_contour_in_square(contours[idx])
        boxes.append(square)
        ids.append(idx)
        length = square_sizes(square)

        if abs(length[0] - max_square[0]) > math.sqrt(staff_gap) or abs(length[1] - max_square[1]) > math.sqrt(staff_gap):
            possible_notes.append(len(boxes)-1)
        else:
            count += 1
            result_conturs.append(contours[idx])

    to_remove = []
    new_boxes = []
    for idx in possible_notes:
        for possible_companion in possible_notes:
            if possible_companion == idx:
                continue
            xl = boxes[possible_companion][3] if boxes[possible_companion][3] < boxes[idx][3] else boxes[idx][3]
            xr = boxes[possible_companion][1] if boxes[possible_companion][1] > boxes[idx][1] else boxes[idx][1]
            yu = boxes[possible_companion][0] if boxes[possible_companion][0] > boxes[idx][0] else boxes[idx][0]
            yd = boxes[possible_companion][2] if boxes[possible_companion][2] < boxes[idx][2] else boxes[idx][2]

            if xr - xl <= max_square[0] and yu - yd <= max_square[1]:
                count += 1
                possible_notes.remove(idx)
                possible_notes.remove(possible_companion)
                to_remove.append(boxes[possible_companion])
                to_remove.append(boxes[idx])
                new_boxes.append((yu, xr, yd, xl))
                result_conturs.append(contours[ids[idx]])
    for item in possible_notes:
        to_remove.append(boxes[item])
    for item in to_remove:
        boxes.remove(item)
    for box in new_boxes:
        boxes.append(box)
    return count, boxes, result_conturs


def square_sizes(square):
    return square[1] - square[3], square[0] - square[2]


def localize_staff(image):
    lines = []
    for y in range(0, len(image)):
        should_add = False
        start = -1
        end = -1
        multiple = False
        for x in range(0, len(image[y])):
            if image[y][x] > 0:
                should_add = True
                if start == -1:
                    start = x
                if end != -1:
                    multiple = True
            else:
                if start != -1:
                    if end == -1:
                        end = x
        if should_add and not multiple and (end - start) > len(image[0])/2:
            lines.append(y)

    if len(lines) > 5:
        height = lines[-1] - lines[0]
        approx_gap = height/4
        to_remove = []
        previously_removed = 0
        for i in range(1, len(lines)):
            if abs(lines[i] - lines[i-1 - previously_removed] - approx_gap) < 0.5:
                to_remove.append(lines[i])
                previously_removed += 1
            else:
                previously_removed = 0
    height = lines[-1] - lines[0]
    approx_gap = (height)/4
    return approx_gap, lines[0]


def remove_vertical(img, lines):
    for y in range(0, len(img)):
        for x in range(0, len(img[y])):
            if lines[y][x] and y - 1 >= 0 and (not img[y-1][x] or not img[y+1][x]):
                img[y][x] = False
