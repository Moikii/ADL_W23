import imgaug
import cv2 as cv
from pathlib import Path
import os
import random as rand
import imutils
import photo_preparation_functions as ppf


def find_edges(img):
    blurred_img = cv.GaussianBlur(img.copy(),(5,5),0)
    edged_img = cv.Canny(blurred_img,50,100)
    return edged_img

def create_mask(img, edged_img):
    contours,_ = cv.findContours(edged_img.copy(),cv.RETR_LIST,cv.CHAIN_APPROX_NONE)
    filled_card = cv.fillPoly(img.copy(), [max(contours, key = cv.contourArea)], color = (255, 255, 255))
    _, mask = cv.threshold(filled_card, thresh= 180, maxval = 255, type = cv.THRESH_BINARY)
    return mask
















def create_dataset_dir(OUTPUT_DIR):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    DATASET_IMAGES_DIR = OUTPUT_DIR + '/images/'
    if not os.path.exists(DATASET_IMAGES_DIR):
        os.makedirs(DATASET_IMAGES_DIR)
    
    DATASET_LABELS_DIR = OUTPUT_DIR + '/labels/'
    if not os.path.exists(DATASET_LABELS_DIR):
        os.makedirs(DATASET_LABELS_DIR)


def generate_yaml_file(PLAYING_CARDS_DIR, OUTPUT_DIR):
    card_names = [frame_name.split('_')[0] for frame_name in os.listdir(PLAYING_CARDS_DIR)]
    unique_card_names = list(set(card_names))
    unique_card_names.sort()
    classes_dict = dict((name, i) for i, name in enumerate(unique_card_names))
    number_of_classes = len(unique_card_names)

    with open(OUTPUT_DIR + '/classes.yaml', 'w') as file:
        file.write('classes:\n')
        for card_name in unique_card_names:
            file.write('- ' + card_name + '\n')
        file.write(f'nc: {number_of_classes}')
    return classes_dict


def select_background(BACKGROUNDS_DIR): #todo: resolution 640x640
    background_paths = Path(BACKGROUNDS_DIR).glob('**/*') # .jpg?
    selected_background_path = str(rand.sample(background_paths, 1)[0])
    background = cv.imread(selected_background_path)
    return background


def select_cards(PLAYING_CARDS_DIR, number_of_cards):
    card_paths = Path(PLAYING_CARDS_DIR).glob('*')
    selected_cards_paths = [str(card_path for card_path in rand.sample(card_paths, number_of_cards))]
    selected_cards = [cv.imread(path) for path in selected_cards_paths]
    selected_cards_names = [card_path.split('/')[-1].split('_')[0] for card_path in selected_cards_paths]
    return selected_cards, selected_cards_names


def transform_card(image, card, size, rotation, angle):
    image_height, _, _ = image.shape
    ratio = card.shape[1]/card.shape[0]
    card_height = int(size*image_height)
    card_width = int(card_height*ratio)

    card = cv.resize(card, (card_width, card_height))
    card = imutils.rotate_bound(card, rotation)
    #todo: angle
    return card


def place_cards(background, cards, card_classes, max_size, min_size, max_rotation, max_angle, overlapping):
    image = background
    bounding_boxes = list()

    for card in cards:
        size = rand.uniform(min_size, max_size)
        rotation = rand.uniform(0, max_rotation)
        angle = rand.uniform(0, max_angle) #todo
        transformed_card = transform_card(image, card, size, rotation, angle)
        image, bounding_boxes = place_card(image, transformed_card, overlapping, bounding_boxes)

    labels = ''
    for i, bounding_box in enumerate(bounding_boxes):
        labels += card_classes[i] + ' ' + ' '.join(bounding_box) + '\n'
    return image, labels


def get_min_max_borders(bounding_box):
    bb_x_min = bounding_box[0] + bounding_box[2]/2
    bb_x_max = bounding_box[0] - bounding_box[2]/2
    bb_y_min = bounding_box[1] + bounding_box[3]/2
    bb_y_max = bounding_box[1] - bounding_box[3]/2
    return bb_x_min, bb_x_max, bb_y_min, bb_y_max


def no_bounding_boxes_overlap(bounding_boxes, proposed_bounding_box):
    bb_overlap = False
    pbb_x_min, pbb_x_max, pbb_y_min, pbb_y_max = get_min_max_borders(proposed_bounding_box)

    for bounding_box in bounding_boxes:
        bb_x_min, bb_x_max, bb_y_min, bb_y_max = get_min_max_borders(bounding_box)
        if ((bb_x_min < pbb_x_max) or (pbb_x_min < bb_x_max)) and ((bb_y_min < pbb_y_max) or (pbb_y_min < bb_y_max)):
            bb_overlap = True
            break

    return bb_overlap


def overlay_images(image, card, x_pos, y_pos):
    return image #todo


def place_card(image, card, overlapping, bounding_boxes):
    card_placed = False
    max_tries = 10 #?
    image_height, image_width, _ = image.shape
    card_height, card_width, _ = card.shape
    relative_card_width = card_width/image_width
    relative_card_heigth = card_height/image_height

    while (not card_placed) and (tries < max_tries):
        x_pos = rand.uniform(relative_card_width/2, 1 - relative_card_width/2) # todo: half card in picture
        y_pos = rand.uniform(relative_card_heigth/2, 1 - relative_card_heigth/2)
        proposed_bounding_box = [x_pos, y_pos, relative_card_width, relative_card_heigth]

        if ((no_bounding_boxes_overlap(bounding_boxes, proposed_bounding_box)) or overlapping): #todo: adjust bbs for overlaps
            bounding_boxes.append(proposed_bounding_box)
            image = overlay_images(image, card, x_pos, y_pos)
            card_placed = True
        else:
            tries += 1

    # print(bounding_box)
    # print(card.shape)
    # print(image.shape)

    # cv.imshow('image', image)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    
    return image, bounding_box





def generate_dataset(BACKGROUNDS_DIR, PLAYING_CARDS_DIR, OUTPUT_DIR, number_of_images, max_number_of_cards_per_image,
                     min_size, max_size, max_rotation, max_angle, overlapping, seed):
    rand.seed(seed)
    create_dataset_dir(OUTPUT_DIR)
    classes_dict = generate_yaml_file(PLAYING_CARDS_DIR, OUTPUT_DIR)

    for i in range(number_of_images):
        background = select_background(BACKGROUNDS_DIR)
        number_of_cards_per_image = rand.randint(1, max_number_of_cards_per_image)
        cards, names = select_cards(PLAYING_CARDS_DIR, number_of_cards_per_image)
        card_classes = [classes_dict[name] for name in names]
        image, labels = place_cards(background, cards, card_classes, max_size, min_size, max_rotation, max_angle, overlapping)

        cv.imwrite(OUTPUT_DIR + '/images/' + i + '.jpg', image)
        with open(OUTPUT_DIR + '/labels/' + i + '.txt', 'w') as file:
            file.write(labels)

    print('Dataset generated and saved at: ' + OUTPUT_DIR)
    return OUTPUT_DIR



if __name__ == '__main__':
    #input parameters
    BACKGROUNDS_DIR = '/home/moiki/Documents/Files/studies/4_Semester/ADL/ADL_W23/data/dtd'
    PLAYING_CARDS_DIR = '/home/moiki/Documents/Files/studies/4_Semester/ADL/ADL_W23/data/sg_cards_processed'
    OUTPUT_DIR = '/home/moiki/Documents/Files/studies/4_Semester/ADL/ADL_W23/data/sg_cards_dataset'

    number_of_images = 2
    max_number_of_cards_per_image = 4
    min_size = 0.1
    max_size = 0.4
    max_rotation = 90
    max_angle = 45
    overlapping = True
    seed = 8
    
    # generate dataset
    generate_dataset(BACKGROUNDS_DIR, PLAYING_CARDS_DIR, OUTPUT_DIR, number_of_images, max_number_of_cards_per_image,
                     min_size, max_size, max_rotation, max_angle, overlapping, seed)





