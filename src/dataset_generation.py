import cv2 as cv
import imutils
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
PLAYING_CARDS_DIR = str(PROJECT_DIR) + '/data/sg_cards'
PROCESSED_PLAYING_CARDS_DIR = str(PROJECT_DIR) + '/data/sg_cards_processed'

def read_image(path: str):
    img = cv.imread(path)
    img = imutils.resize(img, width=1000, height=1800)
    # cv.imshow('original image',img)
    # cv.waitKey(0)
    return img

def find_edges(img):
    blurred_img = cv.GaussianBlur(img.copy(),(5,5),0)
    edged_img = cv.Canny(blurred_img,100,250)
    # cv.imshow('edged image',edged_img)
    # cv.waitKey(0)
    return edged_img

def create_mask(edged_img):
    contours,_ = cv.findContours(edged_img.copy(),cv.RETR_LIST,cv.CHAIN_APPROX_NONE)
    filled_card = cv.fillPoly(img.copy(), [max(contours, key = cv.contourArea)], color = (255, 255, 255))
    _, mask = cv.threshold(filled_card, thresh= 180, maxval = 255, type = cv.THRESH_BINARY)
    # cv.imshow("mask", mask)
    # cv.waitKey(0)
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
    return cropped_img




cards= [str(path).split('/')[-1] for path in Path(PLAYING_CARDS_DIR).glob('*')]

for card in cards:
    img = read_image(PLAYING_CARDS_DIR + '/' + card)
    edged_img = find_edges(img)
    mask = create_mask(edged_img)
    cropped_img = crop_image(img, mask)

    # cv.destroyAllWindows()
    cv.imwrite(PROCESSED_PLAYING_CARDS_DIR + '/' + card, cropped_img)

