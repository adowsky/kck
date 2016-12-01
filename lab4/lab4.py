import sys
from notes_finder import *


def process_image(wait, filename):
    if filename is None or filename == "":
        img = cv2.imread('jingle.jpg', cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(filename, cv2.IMREAD_COLOR)
    notes_finder = NotesFinder(img)
    notes_count = notes_finder.count_notes(wait)

    if wait:
        cv2.waitKey(0)
    cv2.destroyAllWindows()
    return notes_count


if __name__ == "__main__":
    file = None
    if len(sys.argv) > 1:
        file = sys.argv[1]
    process_image(True, file)
