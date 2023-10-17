import cv2 as cv
import imutils
from pathlib import Path
import os
import numpy as np


def read_image(path: str):
    img = cv.imread(path)
    img = imutils.resize(img, width=1000, height=1800)
    # cv.imshow('original image',img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    return img

def find_edges(img):
    blurred_img = cv.GaussianBlur(img.copy(),(5,5),0)
    edged_img = cv.Canny(blurred_img,50,100)
    # cv.imshow('edged image',edged_img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    return edged_img

def create_mask(img, edged_img):
    contours,_ = cv.findContours(edged_img.copy(),cv.RETR_LIST,cv.CHAIN_APPROX_NONE)
    filled_card = cv.fillPoly(img.copy(), [max(contours, key = cv.contourArea)], color = (255, 255, 255))
    _, mask = cv.threshold(filled_card, thresh= 180, maxval = 255, type = cv.THRESH_BINARY)
    # cv.imshow("mask", mask)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    return mask

def crop_image(img, mask):
    masked_black_img = cv.bitwise_and(img, mask)
    tmp = cv.cvtColor(masked_black_img, cv.COLOR_BGR2GRAY)
    _,alpha = cv.threshold(tmp,0,255,cv.THRESH_BINARY)
    b, g, r = cv.split(masked_black_img)
    rgba = [b,g,r, alpha]
    masked_transparent = cv.merge(rgba,4)
    x, y, w, h = cv.boundingRect(masked_transparent[..., 3])
    cropped_img = masked_transparent[y:y+h, x:x+w, :]
    # cv.imshow("cropped", cropped_img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    return cropped_img


def process_photos(PLAYING_CARDS_DIR):
    PROCESSED_PLAYING_CARDS_DIR = PLAYING_CARDS_DIR + '_processed'
    if not os.path.exists(PROCESSED_PLAYING_CARDS_DIR):
        os.makedirs(PROCESSED_PLAYING_CARDS_DIR)

    cards= [str(path).split('/')[-1] for path in Path(PLAYING_CARDS_DIR).glob('*')]

    for card in cards:
        img = read_image(PLAYING_CARDS_DIR + '/' + card)
        edged_img = find_edges(img)
        mask = create_mask(img, edged_img)
        cropped_img = crop_image(img, mask)

        cv.imwrite(PROCESSED_PLAYING_CARDS_DIR + '/' + card, cropped_img)

    return PROCESSED_PLAYING_CARDS_DIR



def process_videos(PLAYING_CARDS_DIR, n_frames, sensitivity, testing):
    PLAYING_CARDS_FRAMES_DIR = PLAYING_CARDS_DIR + '_frames'
    if not os.path.exists(PLAYING_CARDS_FRAMES_DIR):
        os.makedirs(PLAYING_CARDS_FRAMES_DIR)

    video_names = [str(path).split('/')[-1] for path in Path(PLAYING_CARDS_DIR).glob('*')]

    for video_name in video_names:
        video = cv.VideoCapture(PLAYING_CARDS_DIR + '/' + video_name)
        fps = video.get(cv.CAP_PROP_FPS)
        frame_count = video.get(cv.CAP_PROP_FRAME_COUNT)
        frame_delta = frame_count/n_frames

        brightness_last_frame = np.NaN
        for i in range(n_frames):
            time_ms = int(1000*frame_delta*i/fps)
            video.set(cv.CAP_PROP_POS_MSEC, time_ms)
            ret, frame = video.read()
            brightness = np.average(np.linalg.norm(frame, axis=2)) / np.sqrt(3)
            
            if(abs(brightness_last_frame - brightness) > sensitivity):
                cv.imwrite(PLAYING_CARDS_FRAMES_DIR + '/' + video_name.split('.')[0]  + f'_{time_ms}.png', frame)
                if testing:
                    print(f'time: {time_ms:.2f}, last brightness: {brightness_last_frame:.2f}, current brightness: {brightness:.2f}')
                    cv.imshow(f'frame {time_ms:.2f}ms' , frame)
                    cv.waitKey(0)
                    cv.destroyAllWindows()

            brightness_last_frame = brightness
        
    return PLAYING_CARDS_FRAMES_DIR



if __name__ == '__main__':
    relative_location_videos = '/data/sg_cards'
    n_frames = 10
    sensitivity = 7
    testing = False


    PROJECT_DIR = Path(__file__).parent.parent
    PLAYING_CARDS_DIR = str(PROJECT_DIR) + relative_location_videos

    PLAYING_CARDS_FRAMES_DIR = process_videos(PLAYING_CARDS_DIR, n_frames, sensitivity, testing)
    process_photos(PLAYING_CARDS_FRAMES_DIR)



    