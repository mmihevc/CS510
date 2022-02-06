import cv2
import sys
import os
from sys import platform

(major_ver, minor_ver, subminor_ver) = cv2.__version__.split('.')

EXTENSIONS = ".mp4"

TRACKER_TYPES = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']

BOXES = {"pedestrian.mp4": (415, 166, 121, 266),
         "plane_takeoff.mp4": (340, 291, 66, 36),
         "Merlin_run.mp4": (418, 241, 19, 30),
         "frigatebird.mp4": (415, 315, 133, 159),
         "Dusty_snow.mp4": (218, 72, 107, 164),
         "aerobatic.mp4": (423, 163, 64, 69)}

ESCAPE = 27
if platform == "linux" or platform == "linux2":
    RIGHT_ARROW = 83
    LEFT_ARROW = 81
elif platform == "darwin":
    RIGHT_ARROW = 3
    LEFT_ARROW = 2


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


def get_videos():
    videos = []
    for _, _, files in os.walk("./Videos"):
        for file in files:
            if file.endswith(EXTENSIONS):
                videos.append(file)
    return videos


def get_dim(frame):
    print(f"How large do you want the image? (1 is full 1080p)")
    scale_factor = float(input())
    multiplier = (scale_factor * 1080) / frame.shape[0]
    return int(frame.shape[1] * multiplier), int(frame.shape[0] * multiplier)


def scrub(frames, hasFrame, frame_number):
    while True:
        k = cv2.waitKey(0) & 0xff
        if k & 0xff == ESCAPE:
            exit()
        if k == RIGHT_ARROW:
            frame_number += 1
            if not hasFrame[frame_number]:
                break
            else:
                cv2.imshow("Tracking", frames[frame_number])
            # go forward until we cannot
        if k == LEFT_ARROW:
            if frame_number != 0:
                frame_number -= 1
            cv2.imshow("Tracking", frames[frame_number])
            # go backward until we cannot


def main():
    # Inspired by https://learnopencv.com/object-tracking-using-opencv-cpp-python/
    tracker_type = TRACKER_TYPES[getFromList(TRACKER_TYPES)]
    count = 0
    videos = get_videos()

    if tracker_type == 'BOOSTING':
        tracker = cv2.TrackerBoosting_create()
    if tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create()
    if tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    if tracker_type == 'TLD':
        tracker = cv2.TrackerTLD_create()
    if tracker_type == 'MEDIANFLOW':
        tracker = cv2.TrackerMedianFlow_create()
    if tracker_type == 'GOTURN':
        tracker = cv2.TrackerGOTURN_create()
    if tracker_type == 'MOSSE':
        tracker = cv2.TrackerMOSSE_create()
    if tracker_type == "CSRT":
        tracker = cv2.TrackerCSRT_create()

    # Read video
    videos_name = videos[getFromList(videos)]
    new_video_name = "./Videos/" + videos_name
    video = cv2.VideoCapture(new_video_name)
    total_number_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total number of Frames: {total_number_frames}")
    frames = [False] * total_number_frames
    hasFrame = [False] * total_number_frames
    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')
        sys.exit()

    dim = get_dim(frame)
    frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    print(f"Do you wish to scrub? [y/n]")
    user_input = input()
    should_scrub = (str(user_input).lower() == "y" or str(user_input).lower() == "yes" or user_input == str(1))

    bbox = BOXES[videos_name]
    # bbox = cv2.selectROI(frame, False)
    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)
    frame_number = 0
    while True:
        # Read a new frame
        ok, frame = video.read()

        if not ok:
            break
        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        # Start timer
        timer = cv2.getTickCount()

        # Update tracker
        ok, bbox = tracker.update(frame)

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        # Draw bounding box
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
            count += 1
        else:
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # Display tracker type on frame
        cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        frames[frame_number] = frame
        hasFrame[frame_number] = True
        cv2.imshow("Tracking", frame)

        if should_scrub:
            scrub(frames, hasFrame, frame_number)
        else:
            k = cv2.waitKey(3) & 0xff
            if k == ESCAPE:
                break
        frame_number += 1


if __name__ == '__main__':
    main()
