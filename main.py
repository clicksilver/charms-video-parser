from skills import isValidSkill
import argparse
import cv2
import math
import numpy as np
import os
import pytesseract
import sys

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

ONLY_LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz/'
ONLY_NUMBERS = '0123456789'

def getCustomConfig(whitelist):
    """
    Generates a string to configure tesseract OCR:
    - only recognize a set character set
    - treat the input image as a single line of text (--psm 7)
    """
    return '-c tessedit_char_whitelist=' + whitelist + ' --psm 7'


def loadSlotTemplates():
    """
    Loads the template images for each slot type (none, Lv1, Lv2, and Lv3)
    """
    na = cv2.imread('no-slot.jpg')
    l1 = cv2.imread('L1-slot.jpg')
    l2 = cv2.imread('L2-slot.jpg')
    l3 = cv2.imread('L3-slot.jpg')
    return (na, l1, l2, l3)

SLOT_TEMPLATES = loadSlotTemplates()

# video_file="Talisman-Opening-Mystery-10-080521-2021-05-08-14-27-57.mp4"
# video_file = "videos\\250421 Mystery.mp4"
# video_file = "videos\\280421 Mystery.mp4"

# Horizontal
CG_MIN_X = 458
CG_MAX_X = 1075
NUM_HORZ_SQUARES = 10

# Vertical
CG_MIN_Y = 222
CG_MAX_Y = 655
NUM_VERT_SQUARES = 7

# Unit Square Dim (Estimates)
SQUARE_WIDTH = (517-458)
SQUARE_HEIGHT = (281-222)

# Padding
PADDING_PX_1 = ((CG_MAX_X - CG_MIN_X) - (SQUARE_WIDTH * NUM_HORZ_SQUARES)) / (NUM_HORZ_SQUARES - 1)
PADDING_PX_2 = ((CG_MAX_Y - CG_MIN_Y) - (SQUARE_HEIGHT * NUM_VERT_SQUARES)) / (NUM_VERT_SQUARES - 1)
PADDING_PX = math.floor(PADDING_PX_1)

# Slot Info Panel
SL_MIN_X = 1354
SL_MAX_X = 1480
SL_MIN_Y = 320
SL_MAX_Y = 354
NUM_SL_HORZ = 3
SL_WIDTH = math.floor((SL_MAX_X - SL_MIN_X) / float(NUM_SL_HORZ))


def printMousePos(event, x, y, flags, params):
    """
    Helper to print the x,y coordinates of a mouse click on an image frame
    """
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print(f'clicked at {x}, {y}')


def charmGridIndexToRect(index):
    """
    Charms are laid out in a grid (7 rows, 10 cols) and this grid is indexed
    from left-to-right, then top-to-bottom.
    This function converts the Charm Index into the coordinates of the charm
    icon in the grid layout
    """
    col = math.floor(index % NUM_HORZ_SQUARES)
    row = math.floor((index - col) / float(NUM_HORZ_SQUARES))

    # print(f"{row}, {col}")
    xl = CG_MIN_X + col * (SQUARE_WIDTH + PADDING_PX)
    xr = CG_MIN_X + (col + 1) * (SQUARE_WIDTH + PADDING_PX) - PADDING_PX
    yl = CG_MIN_Y + row * (SQUARE_HEIGHT + PADDING_PX)
    yr = CG_MIN_Y + (row + 1) * (SQUARE_HEIGHT + PADDING_PX) - PADDING_PX

    return ( (xl, yl), (xr, yr) )


def slotIndexToRect(index):
    """
    Converts an index (from 0-3) to the coordinates of the bounding rectangle
    of the slot image.
    """
    xl = SL_MIN_X + index * SL_WIDTH
    xr = SL_MIN_X + (index + 1) * SL_WIDTH
    return ( (xl, SL_MIN_Y), (xr, SL_MAX_Y) )

def getSkillName1Rect():
    """The Rectangle Coordinates containing the first skill name"""
    return ((1156, 417), (1475, 450))

def getSkillName2Rect():
    """The Rectangle Coordinates containing the second skill name"""
    return ((1156, 493),(1479, 530))

def getSkillLevel1Rect():
    """The Rectangle Coordinates containing the first skill level value"""
    return ((1463, 455),(1483, 485))

def getSkillLevel2Rect():
    """The Rectangle Coordinates containing the second skill level value"""
    return ((1462, 531),(1483, 560))

def getSubImage(image, rect):
    """Helper function to extract the sub image of a given rectangle coords"""
    x1 = rect[0][0]
    x2 = rect[1][0]
    y1 = rect[0][1]
    y2 = rect[1][1]
    subimg = image[y1:y2, x1:x2]
    return subimg
    

def drawRectImg(image, rects):
    """Draws a thin red rectangle on a given image with rectangle"""
    return cv2.rectangle(image, rects[0], rects[1], (0, 0, 255), 1)


def drawAllDebugRect(frame):
    """
    Helper function to draw rectangles on all 50 charm positions, and all the
    components in the charm info panel (three slots, skill names and skill
    levels)
    """
    for i in range(50):
       rects = charmGridIndexToRect(i)
       frame = drawRectImg(frame, rects)
    for i in range(3):
        rects = slotIndexToRect(i)
        frame = drawRectImg(frame, rects)
    frame = drawRectImg(frame, getSkillName1Rect())
    frame = drawRectImg(frame, getSkillName2Rect())
    frame = drawRectImg(frame, getSkillLevel1Rect())
    frame = drawRectImg(frame, getSkillLevel2Rect())
    return frame


def binarize(img):
    """Converts image to grayscale"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def threshold(img):
    """Applies thresholding to an image"""
    return cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

def imerode(img, ksize=1, iternum=1):
    """Applies image erosion to a binarized image"""
    return cv2.erode(img, np.ones((ksize,ksize), np.uint8), iterations = iternum)

def imdilate(img, ksize=1, iternum=1):
    """Applies image dilation to a binarized image"""
    return cv2.dilate(img, np.ones((ksize, ksize), np.uint8), iterations = iternum)

def parseTextFromImg(image, rect, char_whitelist, debug=False):
    """
    Given an image, rectangle, and character list, this function returns the
    detected text after each successive image manipulation.
    This function applies the following manipulations:
    - binarize
    - thresholding
    - erosion (1 iteration)
    - erosion (2 iteration)
    The returned object is a list of all text detected after each manipulation.
    """
    texts = []
    def parseText(img):
        return pytesseract.image_to_string(img,
                config=getCustomConfig(char_whitelist))[:-2]
    subimg = getSubImage(image, rect)
    if debug:
        cv2.namedWindow("orig")
        cv2.imshow("orig", subimg)

    # binarize
    subimg = binarize(subimg)
    texts.append(parseText(subimg))
    if debug:
        cv2.namedWindow("binarize")
        cv2.imshow("binarize", subimg)

    # thresholding
    subimg = threshold(subimg)
    texts.append(parseText(subimg))
    if debug:
        cv2.namedWindow("thresh")
        cv2.imshow("thresh", subimg)

    # erosion
    thr = subimg.copy()
    # first pass
    ero1 = imerode(thr, 2, 1)
    texts.append(parseText(ero1))
    # second pass
    ero2 = imerode(thr, 2, 2)
    texts.append(parseText(ero2))
    # debug print
    if debug:
        cv2.namedWindow("ero1")
        cv2.imshow("ero1", ero1)
        cv2.namedWindow("ero2")
        cv2.imshow("ero2", ero2)

    return texts


def extractSlotTemplates(cap, display):
    """
    Hacked function to extract the templates given some manual labelling.
    """
    frame_with_NA = None
    frame_with_L1 = None
    frame_with_L2 = None
    frame_with_L3 = None

    while True:
        ret, f = cap.read()
        t = cap.get(cv2.CAP_PROP_POS_MSEC)
        if t > 0 and frame_with_L1 is None:
            frame_with_L1 = getSubImage(f.copy(), slotIndexToRect(0))
        if t > 18 * 1000 and frame_with_L2 is None:
            frame_with_L2 = getSubImage(f.copy(), slotIndexToRect(0))
            print('read 20 seconds')
        if t > 31 * 1000 and frame_with_NA is None:
            frame_with_NA = getSubImage(f.copy(), slotIndexToRect(0))
            print('read 30 seconds')
        if t > 42 * 1000 and frame_with_L3 is None:
            frame_with_L3 = getSubImage(f.copy(), slotIndexToRect(0))
            print('read 40 seconds')
            break

    if display:
        cv2.namedWindow("slot NA")
        cv2.imshow("slot NA", frame_with_NA)
        cv2.namedWindow("slot L1")
        cv2.imshow("slot L1", frame_with_L1)
        cv2.namedWindow("slot L2")
        cv2.imshow("slot L2", frame_with_L2)
        cv2.namedWindow("slot L3")
        cv2.imshow("slot L3", frame_with_L3)
    
    cv2.imwrite('no-slot.jpg', frame_with_NA)
    cv2.imwrite('L1-slot.jpg', frame_with_L1)
    cv2.imwrite('L2-slot.jpg', frame_with_L2)
    cv2.imwrite('L3-slot.jpg', frame_with_L3)


def matchRectToSlot(subimg):
    """
    Classify the subimage containing a charm's slot as containing either:
    - no deco slot
    - Lv1 deco slot
    - Lv2 deco slot
    - Lv3 deco slot
    """
    vals = []
    for template in SLOT_TEMPLATES:
        res = cv2.matchTemplate(subimg, template, cv2.TM_SQDIFF)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        vals.append(min_val)
    return vals.index(min(vals))

def parseCharmFromImage(frame):
    """
    Given a frame, parse as much information as possible from the info panel.
    This includes the following:
    - the three types of decoration slots
    - first skill name and level
    - second skill name and level (Optional)
    
    Returns a tuple of:
    - the charm string summarizing the above information
    - whether any errors or unexpected values are parsed from the info panel
    """
    parse_errors = []

    # Read the first skill name (REQUIRED)
    skill_1_name = ''
    skill_1_names = parseTextFromImg(frame, getSkillName1Rect(), ONLY_LETTERS)
    for name in skill_1_names:
        if isValidSkill(name):
            skill_1_name = name
    if skill_1_name == '':
        parse_errors.append(0)

    # Read the first skill level (REQUIRED)
    skill_1_level = ''
    skill_1_levels = parseTextFromImg(frame, getSkillLevel1Rect(), ONLY_NUMBERS)
    for name in skill_1_levels:
        if name in list(ONLY_NUMBERS):
            skill_1_level = name

    if skill_1_level == '':
        parse_errors.append(1)

    # String to represent no skill or level found.
    EMPTY = '-'

    # Read the second skill name (OPTIONAL)
    skill_2_name = EMPTY
    skill_2_names = parseTextFromImg(frame, getSkillName2Rect(), ONLY_LETTERS)
    for name in skill_2_names:
        if isValidSkill(name):
            skill_2_name = name

    # Read the second skill level (OPTIONAL)
    if skill_2_name != EMPTY:
        skill_2_level = ''
        skill_2_levels = parseTextFromImg(frame, getSkillLevel2Rect(), ONLY_NUMBERS)
        for name in skill_2_levels:
            if name in list(ONLY_NUMBERS):
                skill_2_level = name
        if skill_2_level == '':
            parse_errors.append(3)
    else:
        skill_2_level = EMPTY

    # Parse the decoration slots
    slot_0_img = getSubImage(frame, slotIndexToRect(0))
    slot_1_img = getSubImage(frame, slotIndexToRect(1))
    slot_2_img = getSubImage(frame, slotIndexToRect(2))
    slot_0_level = matchRectToSlot(slot_0_img)
    slot_1_level = matchRectToSlot(slot_1_img)
    slot_2_level = matchRectToSlot(slot_2_img)

    charm_string = f"{skill_1_name},{skill_1_level},{skill_2_name},{skill_2_level},{slot_0_level}-{slot_1_level}-{slot_2_level}"
    return charm_string, parse_errors

def parseAllFrames(cap):
    """
    Function to just parse every single frame in the video. Very expensive, just
    used this to demo stuff and verify my assumptions.
    """
    ret, frame = cap.read()
    charm_string = parseCharmFromImage(frame)
    frame_count = 0
    charms = []
    while True:
        if charm_string is None:
            print(frame_count)
        else:
            charms.append(charm_string)
        frame_count = frame_count + 1
        ret, frame = cap.read()
        if ret:
            charm_string = parseCharmFromImage(frame)
        else:
            break

    # Cull the extracted charms to only the unique ones
    unique_charms = []
    i = 0
    while True:
        charm_i = charms[i]
        charm_j = charms[i+1]
        if charm_i != charm_j:
            unique_charms.append(charm_i)
        i = i + 1
        if (i+1) == len(charms):
            break
    unique_charms.append(charms[-1])
    return unique_charms


def waitOnKey(key):
    """
    Blocking wait for a specific key press.
    """
    while True:
        if cv2.waitKey(1) & 0xFF == ord(key):
            break


def extractFramesInRange(cap, start, num):
    """
    Helper function to extract |num| frames from |start| as images.
    """
    for i in range(start):
        ret, frame = cap.read()
    for i in range(num):
        ret, frame = cap.read()
        cv2.imwrite(f"data/{i}.jpg", frame)


def extractCharmFrames(cap, debug=False):
    """
    Function to parse the entire video and identify only the first frame showing
    a new charm. This reduces the number of frames to look at from N to just 50.

    This function outputs each individual frame as a new image in the frames/
    directory.
    """
    indexes = []

    os.makedirs("frames", exist_ok=True)
    
    if debug:
        cv2.namedWindow('i')
        cv2.namedWindow('j')

    # determine charm highlight selection pixel
    c = 0
    charm_rect = charmGridIndexToRect(c)
    xc = charm_rect[0][0] + 10
    yc = charm_rect[0][1] + 10

    # fixed frame to compare against
    i = 0
    ret, frame = cap.read()
    color_i = frame[yc][xc]
    frame_i = frame.copy()

    if debug:
        debug_frame_i = frame_i.copy()
        cv2.line(debug_frame_i, (xc,yc), (xc,yc), (0,0,255), 2)
        cv2.imshow('i', debug_frame_i)

    j = i
    while True:
        # advancing frame to compare against the fixed frame i
        j = j + 1 
        ret, frame = cap.read()
        if not ret:
            print(f"Wrote frame {c}...")
            cv2.imwrite(f'frames/{c}.jpg', frame_i)
            break
        color_j = frame[yc][xc]
        frame_j = frame.copy()

        # selection has advanced
        if color_j[0] < 50:
            if debug:
                debug_frame_j = frame_j.copy()
                cv2.line(debug_frame_j, (xc,yc), (xc,yc), (0,0,255), 2)

            print(f"Wrote frame {c}...")
            cv2.imwrite(f'frames/{c}.jpg', frame_i)

            # determine new charm selection pixel position
            c = c + 1
            charm_rect = charmGridIndexToRect(c)
            xc = charm_rect[0][0] + 10
            yc = charm_rect[0][1] + 10

            # update the fixed frame to the current advancing frame
            color_i = frame[yc][xc]
            frame_i = frame_j.copy()
            i = j

            if c == 50:
                break

            if debug:
                cv2.line(debug_frame_j, (xc,yc), (xc,yc), (0,0,255), 2)
                cv2.imshow('j', debug_frame_j)
                waitOnKey('q')


def parseCharmFrames():
    """
    Function reads all the individual charm frames and parses them for the info.
    
    Outputs which frame index it failed to parse (helpful for debugging).

    Returns a list of charm strings.
    """
    charms = []
    for i in range(50):
        print(f"Reading frame {i}...")
        img = cv2.imread(f"frames/{i}.jpg")
        charm_string, parse_errors = parseCharmFromImage(img)
        if len(parse_errors) > 0:
            print(f"failed to parse {i}")
        else:
            charms.append(charm_string)
    return charms


def debugParseCharmFrame(index):
    """
    Helper function to analyze one particular frame and prototype charm parsing.
    """
    img = cv2.imread(f"frames/{index}.jpg")
    subimg = getSubImage(img, getSkillLevel2Rect())
    
    cv2.namedWindow('s1')
    cv2.imshow('s1', subimg)

    # Image preprocessing
    s1_bin = binarize(subimg)
    s1_thr = threshold(s1_bin)
    s1_ero = imerode(s1_thr, 2, 1)
    s1_ero2 = imerode(s1_thr, 2, 2)
    
    def get_text(img):
        return pytesseract.image_to_string(img,
                config=getCustomConfig(ONLY_NUMBERS))[:-2]

    def showImg(name, img):
        cv2.namedWindow(name)
        cv2.imshow(name, img)
        text = get_text(img)
        print(f"{name} text = {text}")

    showImg('bin', s1_bin)
    showImg('thr', s1_thr)
    showImg('ero', s1_ero)
    showImg('ero2', s1_ero2)

    waitOnKey('q')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_file")
    args = parser.parse_args()
    video_file = args.video_file

    print(video_file)

    print("Extracting relevant frames from video")
    cap = cv2.VideoCapture(video_file)
    extractCharmFrames(cap)
    cap.release()

    print("Parse Charms from extracted frames")
    charms = parseCharmFrames()

    # Write the charms out to file.
    f = open(f"{video_file[:-4]}.txt", "w")
    for c in charms:
        f.write(f"{c}\n")
    f.close()

    return 0

if __name__ == "__main__":
    sys.exit(main())
