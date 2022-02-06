import os
import cv2
import numpy as np

# Sources:
# https://docs.opencv.org/master/d4/dc6/tutorial_py_template_matching.html
# https://pythonprogramming.net/template-matching-python-opencv-tutorial/
THRESH_HOLD = .8
NUM_ITEMS = 3
methods = ['TM_CCOEFF', 'TM_CCOEFF_NORMED', 'TM_CCORR', 'TM_CCORR_NORMED', 'TM_SQDIFF', 'TM_SQDIFF_NORMED']
colors = [(230,100,255),(0,0,200),(170,100,130),(150,40,20),(130,100,100),(100,150,150)]

def getFromList(l):
    valid = False
    value = -1
    while not valid:
        print(f"Please select:")
        for x in range(len(l)):
            print(f"{x}:{l[x]}", end="    ")
        print()
        value = int(input())
        valid = 0 <= value < len(l)
    return value


def get_templates_pretrained():
    templates = {}
    for root, dirs, files in os.walk("./Images"):
        for dir in dirs:
            images = []
            for _, _, files in os.walk("./Images/" + dir):
                for file in files:
                    images.append(cv2.imread("./Images/" + dir + '/' + file))
            templates[dir] = images
    return templates


def get_templates_non_pretrained(num_items):
    templates = {}
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    while num_items > 0:
        ret, frame = cap.read()
        cv2.imshow('Learning', frame)
        c = cv2.waitKey(1)
        if c == 27:
            break
        elif c != -1:
            box = cv2.selectROI("Select Item", frame, False)
            print("Please label this image")
            label = str(input())
            imCrop = frame[int(box[1]):int(box[1] + box[3]), int(box[0]):int(box[0] + box[2])]
            imCrop.astype(np.uint8)
            templates[label] = [imCrop]
            num_items -= 1
            cv2.destroyWindow("Select Item")
    cap.release()
    cv2.destroyAllWindows()
    return templates


def get_max_res(frame, method, templates):
    max_res = {}
    for key in templates.keys():
        max_res[key] = None
        for template in templates[key]:
            res = cv2.matchTemplate(frame, template, method)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            #print(max_val)
            #print(max_res[key])
            if max_res[key] is None or max_res[key][1] < max_val:
                max_res[key] = (res,max_val, template)
    return max_res


def get_dim(frame, scale_factor):
    multiplier = (scale_factor * 1080) / frame.shape[0]
    return int(frame.shape[1] * multiplier), int(frame.shape[0] * multiplier)


def template_match(cap, method, templates, scale_factor):
    ret, frame = cap.read()
    frame = cv2.resize(frame, get_dim(frame, scale_factor), interpolation=cv2.INTER_AREA)
    max_res = get_max_res(frame, method, templates)

    for key in max_res.keys():
        res = max_res[key][0]
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        #print(f"Min Value: {min_val}, Max Value: {max_val}")
        if max_val < THRESH_HOLD or (method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED] and min_val > ((1 - THRESH_HOLD)-.12)):
            continue
        _, w, h = max_res[key][2].shape[::-1]
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        cv2.rectangle(frame, top_left, bottom_right, 255, 2)
        cv2.rectangle(frame, (top_left[0], bottom_right[1]), (bottom_right[0], bottom_right[1] + 30), 255, -1)
        cv2.putText(frame, key, ((top_left[0]) + 10, bottom_right[1] + 24), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

    cv2.imshow('Detector', frame)
    for key in max_res.keys():
        cv2.imshow(f'res: {key}',max_res[key][0])


def main():
    print("Do you want pretrained?")
    userInput = input()
    pretrained = userInput == 'Y' or userInput == "y" or userInput == "1"
    print(f"How large do you want the image? (1 is full 1080p)")
    scale_factor = float(input())
    method = eval("cv2." + methods[getFromList(methods)])
    if pretrained:
        templates = get_templates_pretrained()
    else:
        templates = get_templates_non_pretrained(NUM_ITEMS)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    while True:
        # tracker_comparison(cap,templates,scale_factor)
        template_match(cap, method, templates, scale_factor)
        c = cv2.waitKey(1)
        if c == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

def tracker_comparison(cap, templates, scale_factor):
    ret, frame = cap.read()
    frame = cv2.resize(frame, get_dim(frame, scale_factor), interpolation=cv2.INTER_AREA)
    for method, col in zip(methods,colors):
        meth = eval("cv2." + method)
        for key in templates.keys():
            for template in templates[key]:
                res = cv2.matchTemplate(frame, template, meth)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

                if max_val < THRESH_HOLD or (method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED] and min_val < (1 - THRESH_HOLD)):
                    continue
                _, w, h = template.shape[::-1]
                if meth in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                    top_left = min_loc
                else:
                    top_left = max_loc
                bottom_right = (top_left[0] + w, top_left[1] + h)
                cv2.rectangle(frame, top_left, bottom_right, col, 2)
                cv2.rectangle(frame, (top_left[0], bottom_right[1]), (bottom_right[0], bottom_right[1] + 30), col, -1)
                cv2.putText(frame, method, ((top_left[0]) + 10, bottom_right[1] + 24), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 255), 1)
        cv2.imshow('Detector',frame)

    pass
if __name__ == '__main__':
    main()
