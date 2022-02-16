import cv2
import sys
import numpy as np
import os
from sys import platform

(major_ver, minor_ver, subminor_ver) = cv2.__version__.split('.')


def select_tracker(tracker_list, selected, i):
    if selected:
        return tracker_list[i]
    else:
        print("Potential trackers: ")
        for index in range(len(tracker_list)):
            print(index, tracker_list[index])
        value = int(input(f"Which tracker would you like to use: "))
        print()
        return select_tracker(tracker_list, True, value)


def select_video(video_list, selected, i):
    if selected:
        return video_list[i]
    else:
        print("Potential videos: ")
        for index in range(len(video_list)):
            print(index, video_list[index])
        value = int(input(f"Which video would you like to use: "))
        return select_video(video_list, True, value)


def move_frame_by_frame(f, has_next, index):
    while video.isOpened():
        key = cv2.waitKey(0) & 0xff
        # Quit when 'esc' is pressed
        if key == ord('q'):
            exit()
        if key == 3:
            index += 1
            if not has_next[index]:
                break
            else:
                cv2.imshow("video", f[index])
        elif key == 2:
            if index != 0:
                index -= 1
            cv2.imshow("video", f[index])


if __name__ == '__main__':

    # Set up tracker.
    # Instead of MIL, you can also use

    potential_tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    tracker_type = select_tracker(potential_tracker_types, False, -1)

    if int(minor_ver) < 3:
        tracker = cv2.Tracker_create(tracker_type)
    else:
        if tracker_type == 'BOOSTING':
            tracker = cv2.legacy.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv2.legacy.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.legacy.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()
        if tracker_type == 'MOSSE':
            tracker = cv2.legacy.TrackerMOSSE_create()
        if tracker_type == "CSRT":
            tracker = cv2.TrackerCSRT_create()

    # Read video
    videos = ['videos/arloPuppy.mp4', 'videos/bike.mp4', 'videos/Dusty_snow.mp4', 'videos/frigatebird.mp4',
              'videos/Merlin_run.mp4']
    video_location = select_video(videos, False, -1)
    video = cv2.VideoCapture(video_location)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = [False] * total_frames
    hasNextFrame = [False] * total_frames
    frame_index = 0

    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')
        sys.exit()

    # Define an initial bounding box
    bbox = (287, 23, 86, 320)

    frame_input = input(f"Do you wish to move frame-by-frame [Y/N]: ")
    move_frame = str(frame_input).lower() == "y"

    # Uncomment the line below to select a different bounding box
    bbox = cv2.selectROI(frame, False)

    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)

    while True:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break

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
        else:
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # Display tracker type on frame
        cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

        # Display result
        frames[frame_index] = frame
        hasNextFrame[frame_index] = True
        cv2.imshow("Tracking", frame)

        # Exit if ESC pressed
        if move_frame:
            move_frame_by_frame(frames, hasNextFrame, frame_index)
        else:
            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break
        frame_index += 1
