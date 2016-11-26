import cv2
import copy


def prepare_image_to_processing(img):
    cimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bw = cv2.adaptiveThreshold(~cimg, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 43, -15)
    bw = cv2.erode(bw, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))
    return cv2.dilate(bw, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))


def extract_notes(vertical):
    verticalsize = int(vertical.shape[0] / 30)
    vertical_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (3, verticalsize))
    vertical = cv2.erode(vertical, vertical_structure)
    vertical = cv2.dilate(vertical, vertical_structure)
    vertical_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(verticalsize / 2)))
    vertical = cv2.erode(vertical, vertical_structure)
    return cv2.dilate(vertical, vertical_structure)


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
            if source[y][x] > 0:
                top = y
        y += 1
        if y > max_size:
            top = -1

    bottom = None
    y = 0
    while bottom is None:
        for x in source[y]:
            if source[max_size - y][x] > 0:
                bottom = max_size - y
        y += 1
        if y > max_size:
            bottom = -1
    return bottom - top


def remove_metre_sign(source, img, height, q):
    im2, contours, hierarchy = cv2.findContours(source, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
    if hierarchy is not None and len(hierarchy) > 0:
        metre_sign_idx = find_metre_sign(contours, hierarchy, height, q)
        if metre_sign_idx is not None:
            remove_from_image(img, close_contour_in_square(contours[metre_sign_idx]), 255)


def find_metre_sign(contours, hierarchy, height, q):
    left_idx, others = find_most_left_shape(contours, hierarchy)
    left = close_contour_in_square(contours[left_idx])
    return left_idx if abs(left[0] - left[2] - height) < q else None


def count_possible_notes(notes, contours):
    max_square = 0,0
    possible_notes = []
    count = 0
    boxes = []
    for idx in notes:
        square = close_contour_in_square(contours[idx])
        boxes.append(square)
        length = square_length(square)
        if length[0] > max_square[0] or length[1] > max_square[1]:
            max_square = length

        if length[0] <= max_square[0]/2 or length[1] <= max_square[1]/2:
            possible_notes.append(len(boxes) - 1)
        else:
            count += 1

    for idx in possible_notes:
        for possible_companion in possible_notes:
            if possible_companion == idx:
                continue
            xl = boxes[possible_companion][3] if boxes[possible_companion][3] < boxes[idx][3] else boxes[idx][3]
            xr = boxes[possible_companion][1] if boxes[possible_companion][1] > boxes[idx][1] else boxes[idx][1]
            yu = boxes[possible_companion][0] if boxes[possible_companion][0] > boxes[idx][0] else boxes[idx][0]
            yd = boxes[possible_companion][2] if boxes[possible_companion][2] < boxes[idx][2] else boxes[idx][2]

            if xr - xl <= max_square[0] and yu -yd <= max_square[1]:
                count += 1
                possible_notes.remove(idx)
                possible_notes.remove(possible_companion)
    return count


def square_length(square):
    return square[1] - square[3], square[0] - square[2]


img = cv2.imread('odetojoy.jpg', cv2.IMREAD_COLOR)
bw = prepare_image_to_processing(img)
horizontal = copy.deepcopy(bw)
vertical = copy.deepcopy(bw)

horizontal = extract_staff(horizontal)
vertical = extract_notes(vertical)
vertical = cv2.bitwise_not(vertical)
cv2.imshow("vertical1", bw)
cv2.imshow("horizontal", horizontal)

key = remove_key_sign(bw, vertical)
no_key_img = cv2.adaptiveThreshold(copy.deepcopy(vertical), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, -4)


remove_metre_sign(no_key_img, vertical, count_staff_height(horizontal), int(len(img)/10))

cv2.imshow("vertical", vertical)

edges = cv2.adaptiveThreshold(vertical, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, -4)
im2, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
cv2.imshow("edges", edges)

possible_notes = find_deepest_contours(hierarchy)
count = count_possible_notes(possible_notes, contours)
print("Found: " + str(count) + " musical objects")

cv2.waitKey(0)
cv2.destroyAllWindows()