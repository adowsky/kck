import numpy
import sys
from notes_finder import *


def process_image(wait, filename):
    img = []
    if filename is None or filename == "":
        img = cv2.imread('jingle.jpg', cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(filename, cv2.IMREAD_COLOR)

    bw = prepare_image_to_processing(img)
    horizontal = copy.deepcopy(bw)
    horizontal = extract_staff(horizontal)
    remove_vertical(bw, horizontal)
    bw = erode_dilate(bw, 1, 2)

    cv2.imshow("BW", bw)
    vertical = copy.deepcopy(bw)

    line_gap, first = localize_staff(horizontal)
    vertical = extract_notes(vertical)
    key = remove_key_sign(bw, vertical)
    no_key_img = cv2.adaptiveThreshold(copy.deepcopy(vertical), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, -4)
    remove_metre_sign(no_key_img, vertical, line_gap*4, first, int(len(img)/10))
    cv2.imshow("WITHOUT METRE", vertical)
    #vertical = cv2.erode(vertical, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))
    if line_gap <= 7:
        xx = cv2.dilate(vertical[first + int(line_gap*4.5):], cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 3)))
        for x in range(0, len(xx)):
            vertical[first + int(line_gap*4.5) + x] = xx[x]
    cv2.imshow("preEDGES METRE", vertical)
    edges = cv2.adaptiveThreshold(vertical, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, -4)
    cv2.imshow("EDGES METRE", edges)
    im2, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
    cv2.imshow("ORIGINAL", img)

    possible_notes = find_deepest_contours(hierarchy)
    count, boxes, result_contours = count_possible_notes(possible_notes, contours, vertical, line_gap, int(len(img)/10))

    sorted_boxes = sorted(boxes, key=lambda tup: tup[3])
    notes = ['f2', 'e2', 'd2', 'c2', 'b', 'a', 'g', 'f', 'e', 'd', 'c']
    # melody = []
    # if len(sorted_boxes) > 0:
    #     epsilon = (sorted_boxes[0][0] - sorted_boxes[0][2])/2
    #     for box in sorted_boxes:
    #         found = False
    #         line = first
    #         pos = 1
    #         while not found:
    #             if box[2] < line - 1:
    #                 melody.append(str(pos))
    #                 found = True
    #             elif box[2] - line_gap/2 < line:
    #                 melody.append(str(pos) + "-" + str(pos+1))
    #                 found = True
    #             pos += 1
    #             line += int(line_gap)
    #     print(str(count) + " objects have been found.")
    #     #print(melody)
    #     translated = [notes[2*(int(x, 10)-1)] if len(x) == 1 else notes[2*(int(x[0], 10)-1)+1] for x in melody]
    #     print(translated)

    conimg = cv2.drawContours(numpy.zeros(vertical.shape), result_contours, -1, (100, 10, 200), 1)
    cv2.imshow("found", conimg)
    if wait:
        cv2.waitKey(0)
    cv2.destroyAllWindows()
    return count


if __name__ == "__main__":
    filename = None
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    process_image(True, filename)
