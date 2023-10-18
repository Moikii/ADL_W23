import cv2 as cv
from pathlib import Path
import os
import numpy as np

def crop_image(img):
    grayscale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, thresholded = cv.threshold(grayscale,0,255,cv.THRESH_OTSU)
    kernel = np.ones((3,3), np.uint8)
    img_eroded = cv.erode(thresholded, kernel)
    x, y, w, h = cv.boundingRect(img_eroded)
    cropped_img = img[y:y+h, x:x+w]
    return cropped_img


def save_image(input_video_path, output_folder, card, counter):
    video_name_without_extension = input_video_path.split('/')[-1].split('.')[0]
    cv.imwrite(output_folder + '/' +  video_name_without_extension + f'_{counter}.jpg', card)


def create_playing_card_dir(VIDEO_DIR):
    PLAYING_CARDS_DIR = VIDEO_DIR + '_processed'
    if not os.path.exists(PLAYING_CARDS_DIR):
        os.makedirs(PLAYING_CARDS_DIR)
    return PLAYING_CARDS_DIR


def get_brightness_score(frame):
    grayscale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    score = np.mean(grayscale)
    #todo: richtige schranke
    #print(np.max(grayscale))
    #print(score)
    return score


def brightness_change(current_frame, old_frame):
    brightness_current_frame = get_brightness_score(current_frame)
    brightness_old_frame = get_brightness_score(old_frame)
    if abs(brightness_current_frame - brightness_old_frame) > 30:
        has_changed = True
    else:
        has_changed = False
    return has_changed


def get_frame(video, frame_count):
    video.set(cv.CAP_PROP_POS_FRAMES, frame_count)
    _, frame = video.read()
    return frame


def get_relevant_frames_from_video(video_path):
    frames = list()
    video = cv.VideoCapture(video_path)
    fps = video.get(cv.CAP_PROP_FPS)
    total_frame_count = video.get(cv.CAP_PROP_FRAME_COUNT)

    old_frame = get_frame(video, 0)
    current_frame_count = fps
    while current_frame_count < total_frame_count:
        current_frame = get_frame(video, current_frame_count)
        if brightness_change(current_frame, old_frame):
            frames.append(current_frame)
        old_frame = current_frame
        current_frame_count += fps
    return frames


def process_videos(VIDEOS_DIR):
    PLAYING_CARDS_DIR = create_playing_card_dir(VIDEOS_DIR)
    video_paths = [str(path) for path in Path(VIDEOS_DIR).glob('*')]

    for video_path in video_paths:
        frames = get_relevant_frames_from_video(video_path)
        for i, frame in enumerate(frames):
            card = crop_image(frame)
            save_image(video_path, PLAYING_CARDS_DIR, card, i)

    print(f'Processed cards saved at: ' + PLAYING_CARDS_DIR)
    return PLAYING_CARDS_DIR



if __name__ == '__main__':
    # input parameters
    VIDEOS_DIR = '/home/moiki/Documents/Files/studies/4_Semester/ADL/ADL_W23/data/sg_cards'

    # do processing
    PLAYING_CARD_DIR = process_videos(VIDEOS_DIR)
    



    