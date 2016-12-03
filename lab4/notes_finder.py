import cv2
import copy
import math
import numpy


class NotesFinder:
    def __init__(self, image):
        self.original = copy.deepcopy(image)
        self.image = image
        self.vertical = []
        self.horizontal = []
        self.staff_gap = -1
        self.staff_top_position = -1

    def _prepare_image_to_processing(self):
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.image = cv2.adaptiveThreshold(~self.image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, -15)

    def _erode_dilate(self, x, y):
        self.image = cv2.erode(self.image, cv2.getStructuringElement(cv2.MORPH_CROSS, (x, y)))
        self.image = cv2.dilate(self.image, cv2.getStructuringElement(cv2.MORPH_CROSS, (x, y)))

    def _extract_notes(self):
        verticalsize = int(self.vertical.shape[0] / 30)
        vertical_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (3, verticalsize))
        self.vertical = cv2.erode(self.vertical, vertical_structure)
        self.vertical = cv2.dilate(self.vertical, vertical_structure)
        vertical_structure = cv2.getStructuringElement(cv2.MORPH_RECT,
                                                       (1, int(verticalsize / 2) if int(verticalsize / 2) > 0 else 1))
        self.vertical = cv2.erode(self.vertical, vertical_structure)
        self.vertical = cv2.dilate(self.vertical, vertical_structure)
        self.vertical = cv2.bitwise_not(self.vertical)

    @staticmethod
    def _extract_staff(horizontal):
        horizontalsize = int(horizontal.shape[1] / 30)
        horizontal_sturcture = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalsize, 1))
        horizontal = cv2.erode(horizontal, horizontal_sturcture, (-1, -1))
        return cv2.dilate(horizontal, horizontal_sturcture, (-1, -1))

    def _remove_key_sign(self):
        q = int(len(self.image) / 10)
        im2, contours, hierarchy = cv2.findContours(copy.deepcopy(self.image), cv2.RETR_TREE,
                                                    cv2.CHAIN_APPROX_TC89_KCOS)
        most_left, others = self._find_most_left_shape(contours, hierarchy)
        most_left_box = self._close_contour_in_square(contours[most_left])
        self._remove_from_image(self.vertical, most_left_box, 255)
        for idx in others:
            self._remove_from_image(self.vertical, self._close_contour_in_square(contours[idx]), 255)
        removal_square = len(self.vertical) - 1, most_left_box[1], 0, most_left_box[3]
        self._remove_from_image(self.vertical, removal_square, 255)
        return most_left

    def _find_most_left_shape(self, contours, hierarchy):
        possible_contours = self._find_deepest_contours(hierarchy)

        most_left = self._close_contour_in_square(contours[0])
        most_left_index = possible_contours[0]
        on_most_left_level_vertically = []
        for i in possible_contours:
            square = self._close_contour_in_square(contours[i])
            if square[3] < most_left[3]:
                most_left = square
                most_left_index = i
        for i in possible_contours:
            square = self._close_contour_in_square(contours[i])
            if square[3] >= most_left[3] and square[1] <= most_left[1]:
                on_most_left_level_vertically.append(i)
        return most_left_index, on_most_left_level_vertically

    @staticmethod
    def _find_deepest_contours(hierarchy):
        independent_contours = []
        if hierarchy is None:
            return independent_contours

        for index, contour in enumerate(hierarchy[0]):
            if contour[3] == -1:
                independent_contours.append(index)
        return independent_contours

    @staticmethod
    def _close_contour_in_square(contour):
        x_max, x_min, y_max, y_min = contour[0][0][0], contour[0][0][0], contour[0][0][1], contour[0][0][1]
        for vertex in contour:
            x_max = vertex[0][0] if vertex[0][0] > x_max else x_max
            y_max = vertex[0][1] if vertex[0][1] > y_max else y_max
            x_min = vertex[0][0] if vertex[0][0] < x_min else x_min
            y_min = vertex[0][1] if vertex[0][1] < y_min else y_min
        return y_max, x_max, y_min, x_min

    @staticmethod
    def _remove_from_image(image, square, colour):
        for y in range(square[2], square[0] + 1):
            for x in range(square[3], square[1] + 1):
                image[y][x] = colour

    @staticmethod
    def _count_staff_height(source):
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

    def _remove_metre_sign(self, source, q):
        im2, contours, hierarchy = cv2.findContours(source, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
        metre_sign_idx = self._find_metre_sign(contours, q)
        if metre_sign_idx is not None:
            box = self._close_contour_in_square(contours[metre_sign_idx])
            self._remove_from_image(self.vertical, (box[0], box[1], box[2], 0), 255)

    def _find_metre_sign(self, contours, q):
        possibilities = []
        for idx, contour in enumerate(contours):
            box = self._close_contour_in_square(contour)
            if abs(box[2] - self.staff_top_position) < q and abs(box[0] - (self.staff_top_position + self.staff_gap * 4)) < q:
                possibilities.append(idx)
        metre = None
        if len(possibilities) == 1:
            metre = possibilities[0]

        if len(possibilities) > 1:
            most_left = possibilities[0]
            left_box = self._close_contour_in_square(contours[possibilities[0]])
            for idx in possibilities[1:]:
                box = self._close_contour_in_square(contours[idx])

                if box[3] < left_box[3]:
                    most_left = idx
                    left_box = box
            metre = most_left

        return metre

    def _count_possible_notes(self, notes, contours, q=0):
        max_square = 0, 0
        possible_notes = []
        count = 0
        boxes = []
        ids = []
        result_conturs = []
        for idx in notes:
            square = self._close_contour_in_square(contours[idx])
            length = self._square_sizes(square)
            if self._is_better_box(max_square, length):
                max_square = length

        for idx in notes:
            square = self._close_contour_in_square(contours[idx])
            boxes.append(square)
            ids.append(idx)
            length = self._square_sizes(square)
            is_vertically_different = abs(length[1] - max_square[1]) > math.sqrt(self.staff_gap)
            is_horizontally_different = abs(length[0] - max_square[0]) > math.sqrt(self.staff_gap)
            if is_horizontally_different or is_vertically_different:
                possible_notes.append(len(boxes) - 1)
            else:
                count += 1
                result_conturs.append(contours[idx])

        to_remove = []
        new_boxes = []
        possible_notes_count, boxes, notes_contours = self._verify_notes(possible_notes, boxes, max_square, contours, ids)
        result_conturs += notes_contours
        return count + possible_notes_count, boxes, result_conturs

    def _is_better_box(self, maximum, current):
        size_ratio = max(current[0], current[1]) / min(current[0], current[1])
        size_condition = (current[0] > maximum[0] or current[1] > maximum[1])
        ratio_cond = size_ratio < 2
        height_cond = abs(current[1] - self.staff_gap) < math.sqrt(self.staff_gap)
        return size_condition and ratio_cond and height_cond

    @staticmethod
    def _verify_notes(possible_notes, source_boxes, max_square, contours, ids):
        to_process = possible_notes
        boxes = copy.deepcopy(source_boxes)
        created_boxes = []
        to_remove = []
        result_contours = []
        while len(possible_notes) > 0:
            iteration_boxes = []
            for idx in possible_notes:
                for companion in possible_notes:
                    if companion == idx:
                        continue
                    left = boxes[companion][3] if boxes[companion][3] < boxes[idx][3] else boxes[idx][3]
                    right = boxes[companion][1] if boxes[companion][1] > boxes[idx][1] else boxes[idx][1]
                    up = boxes[companion][0] if boxes[companion][0] > boxes[idx][0] else boxes[idx][0]
                    down = boxes[companion][2] if boxes[companion][2] < boxes[idx][2] else boxes[idx][2]
                    if right - left <= max_square[0] and up - down <= max_square[1]:
                        possible_notes.remove(idx)
                        possible_notes.remove(companion)
                        iteration_boxes.append((up, right, down, left))
                        if boxes[companion] in created_boxes:
                            to_remove.append(boxes[companion])
                            created_boxes.remove(boxes[companion])
                        else:
                            result_contours.append(contours[ids[companion]])
                        if boxes[companion] in created_boxes:
                            to_remove.append(boxes[idx])
                            created_boxes.remove(boxes[idx])
                        else:
                            result_contours.append(contours[ids[idx]])
            for box in iteration_boxes:
                possible_notes.append(len(boxes))
                boxes.append(box)
                created_boxes.append(box)
            if len(iteration_boxes) == 0:
                possible_notes = []
        for box in to_remove:
            boxes.remove(box)

        return len(created_boxes), boxes, result_contours

    @staticmethod
    def _square_sizes(square):
        return square[1] - square[3], square[0] - square[2]

    def _localize_staff(self):
        lines = []
        for y in range(0, len(self.horizontal)):
            should_add = False
            start = -1
            end = -1
            multiple = False
            for x in range(0, len(self.horizontal[y])):
                if self.horizontal[y][x] > 0:
                    should_add = True
                    if start == -1:
                        start = x
                    if end != -1:
                        multiple = True
                else:
                    if start != -1:
                        if end == -1:
                            end = x
            if should_add and not multiple and (end - start) > len(self.horizontal[0]) / 2:
                lines.append(y)

        if len(lines) > 5:
            height = lines[-1] - lines[0]
            approx_gap = height / 4
            to_remove = []
            previously_removed = 0
            for i in range(1, len(lines)):
                if abs(lines[i] - lines[i - 1 - previously_removed] - approx_gap) < 0.5:
                    to_remove.append(lines[i])
                    previously_removed += 1
                else:
                    previously_removed = 0
        height = lines[-1] - lines[0]
        self.staff_gap = height / 4
        self.staff_top_position = lines[0]

    def _remove_vertical(self):
        for y in range(0, len(self.image)):
            for x in range(0, len(self.image[y])):
                if self.horizontal[y][x] and y - 1 >= 0 and (not self.image[y - 1][x] or not self.image[y + 1][x]):
                    self.image[y][x] = False

    def _improve_under_line(self):
        if self.staff_gap <= 7:
            underline = cv2.dilate(self.vertical[self.staff_top_position + int(self.staff_gap * 4.5):],
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 3)))
            for x in range(0, len(underline)):
                self.vertical[self.staff_top_position + int(self.staff_gap * 4.5) + x] = underline[x]

    def _improve_upper_line(self):
        if self.staff_gap <= 7:
            upperline = cv2.dilate(self.vertical[:self.staff_top_position - int(self.staff_gap * 0.5)],
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 3)))
            for x in range(0, len(upperline)):
                self.vertical[x] = upperline[x]

    def _recognize_tones(self, boxes):
        sorted_boxes = sorted(boxes, key=lambda tup: tup[3])
        #notes = ['f2', 'e2', 'd2', 'c2', 'b', 'a', 'g', 'f', 'e', 'd', 'c']
        melody = []
        if len(sorted_boxes) > 0:
            for box in sorted_boxes:
                found = False
                line = self.staff_top_position
                pos = 1
                while not found:
                    if box[2] < line - 1:
                        melody.append(str(pos))
                        found = True
                    elif box[2] - self.staff_gap / 2 < line:
                        melody.append(str(pos) + "-" + str(pos + 1))
                        found = True
                    pos += 1
                    line += int(self.staff_gap)
            #return [notes[2 * (int(x, 10) - 1)] if len(x) == 1 else notes[2 * (int(x[0], 10) - 1) + 1] for x in melody]
            return melody

    def count_notes(self, view=True):
        self._prepare_image_to_processing()
        self.horizontal = self._extract_staff(copy.deepcopy(self.image))
        self._remove_vertical()
        self._erode_dilate(1, 2)
        if view:
            cv2.imshow("BW", self.image)

        self.vertical = copy.deepcopy(self.image)
        self._localize_staff()
        self._extract_notes()

        self._remove_key_sign()
        if view:
            cv2.imshow("WITHOUT KEY", self.vertical)
        no_key_img = cv2.adaptiveThreshold(copy.deepcopy(self.vertical), 255,
                                           cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, -4)
        self._remove_metre_sign(no_key_img, int(len(self.image) / 10))
        self._improve_under_line()
        edges = cv2.adaptiveThreshold(self.vertical, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, -2)
        im2, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
        if view:
            cv2.imshow("ORIGINAL", self.original)
        if view:
            cv2.imshow("im2", im2)

        possible_notes = self._find_deepest_contours(hierarchy)
        count, boxes, result_contours = self._count_possible_notes(possible_notes, contours, int(len(self.image) / 10))
        print(str(count) + " objects have been found.")
        melody = self._recognize_tones(boxes)
        print(melody)

        conimg = cv2.drawContours(numpy.zeros(self.vertical.shape), result_contours, -1, (100, 10, 200), 1)
        if view:
            cv2.imshow("found", conimg)
        return count
