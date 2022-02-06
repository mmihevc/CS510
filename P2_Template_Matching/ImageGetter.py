import cv2


def get_templates(num_items):
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
            imCrop = frame[int(box[1]):int(box[1] + box[3]), int(box[0]):int(box[0] + box[2])]
            cv2.imwrite(f"./Images/BabyYoda/BabyYoda{num_items}.png", imCrop)
            num_items -= 1
            cv2.destroyWindow("Select Item")
    cap.release()
    cv2.destroyAllWindows()


def main():
    get_templates(z)


if __name__ == '__main__':
    main()
