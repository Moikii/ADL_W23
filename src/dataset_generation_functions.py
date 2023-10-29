'''
This file contains function definitions used to create the dataset in the YOLO-format.
A image consists of a randomly selected background and a random number of playing cards
placed on it in also randomly selected positions, rotations and brightness.
Additional we have a parameter to decide if the generated images should contain overlapping
cards (making detection harder).
We also create the data.yaml file, which contains information for the YOLO model during training.
The dataset is saved in a directory given by the user.
'''

import cv2 as cv
from pathlib import Path
import os
import random as rand
import imutils
import numpy as np
import photo_preparation_functions as ppf
from tqdm import tqdm



def transform_coordinates_to_relative_values(bounding_box, image):
    image_height, image_width, _ = image.shape
    x_pos = bounding_box[0]/image_width
    y_pos = bounding_box[1]/image_height
    width = bounding_box[2]/image_width
    height = bounding_box[3]/image_height
    return (x_pos, y_pos, width, height)


def overlay_images(image, card, mask, bounding_box):
    card_height, card_width, _ = card.shape
    # resize card to even number of pixels (avoid rounding errors later)
    card = cv.resize(card, (card_width +card_width%2, card_height + card_height%2))
    mask = cv.resize(mask, (card_width +card_width%2, card_height + card_height%2))

    x_min, x_max, y_min, y_max = get_min_max_bounding_box_coordinates(bounding_box)
    card_placing_area = image[y_min:y_max, x_min:x_max]

    # output for debugging single pixel errors while rounding
    # print(f'new bounding box: {bounding_box}')
    # print(f'new edges: {x_min, x_max, y_min, y_max}')
    # print(f'roi: {roi.shape}')
    # print(f'mask: {mask.shape}')

    # dilate mask to reduce black pixels on card edges when placing
    mask = cv.dilate(mask, np.ones((3,3), np.uint8))
    # mask background and foreground accordingly and overlay them bitwise
    image_foreground = cv.bitwise_or(card, card, mask = mask)
    inv_mask = cv.bitwise_not(mask)
    image_background = cv.bitwise_or(card_placing_area, card_placing_area, mask = inv_mask)
    area_with_placed_card = cv.bitwise_or(image_background, image_foreground)
    image[y_min:y_max, x_min:x_max] = area_with_placed_card
    return image


def get_adjusted_bounding_box_and_card_and_mask(proposed_bounding_box, card, mask, image):
    image_height, image_width, _ = image.shape
    card_height, card_width, _ = card.shape    
    pbb_x_min, pbb_x_max, pbb_y_min, pbb_y_max = get_min_max_bounding_box_coordinates(proposed_bounding_box)

    # get number of pixels that stand over the image edges for each side
    x_adjustment_max = max(0, pbb_x_max - image_width)
    x_adjustment_min = min(0, pbb_x_min)
    y_adjustment_max = max(0, pbb_y_max - image_height)
    y_adjustment_min = min(0, pbb_y_min)
    # adjust bounding box accordingly
    adjusted_x_pos = int(proposed_bounding_box[0] - np.ceil(x_adjustment_max/2) - np.floor(x_adjustment_min/2))
    adjusted_y_pos = int(proposed_bounding_box[1] - np.ceil(y_adjustment_max/2) - np.floor(y_adjustment_min/2))
    adjusted_width = proposed_bounding_box[2] - x_adjustment_max + x_adjustment_min
    adjusted_height = proposed_bounding_box[3] - y_adjustment_max + y_adjustment_min
    # crop card and card-mask to fit the new bounding box
    adjusted_card = card[(-y_adjustment_min):(card_height - y_adjustment_max), (-x_adjustment_min):(card_width - x_adjustment_max)]
    adjusted_mask = mask[-y_adjustment_min:(card_height - y_adjustment_max), -x_adjustment_min:(card_width - x_adjustment_max)]

    # output for debugging single pixel errors while rounding
    # print(f'image data: {image_height, image_width}')
    # print(f'card data: {card_height, card_width}')
    # print(f'old bounding box: {proposed_bounding_box}')
    # print(f'old edges: {pbb_x_min, pbb_x_max, pbb_y_min, pbb_y_max}')
    return (adjusted_x_pos, adjusted_y_pos, adjusted_width, adjusted_height), adjusted_card, adjusted_mask



def get_min_max_bounding_box_coordinates(bounding_box):
    # rounding to integers, because we are working with pixels as positions
    bb_x_min = int(np.floor(bounding_box[0] - bounding_box[2]/2))
    bb_x_max = int(np.ceil(bounding_box[0] + bounding_box[2]/2))
    bb_y_min = int(np.floor(bounding_box[1] - bounding_box[3]/2))
    bb_y_max = int(np.ceil(bounding_box[1] + bounding_box[3]/2))
    return bb_x_min, bb_x_max, bb_y_min, bb_y_max


def no_bounding_boxes_overlap(bounding_boxes, proposed_bounding_box, image):
    no_bb_overlap = True
    image_height, image_width, _ = image.shape
    pbb_x_min, pbb_x_max, pbb_y_min, pbb_y_max = get_min_max_bounding_box_coordinates(proposed_bounding_box)

    # check if overlaps image edges
    if (pbb_x_min < 0) or (pbb_x_max > image_width) or (pbb_y_min < 0) or (pbb_y_max > image_height):
        no_bb_overlap = False
    # check if overlaps other bounding boxes
    for bounding_box in bounding_boxes:
        bb_x_min, bb_x_max, bb_y_min, bb_y_max = get_min_max_bounding_box_coordinates(bounding_box)
        if (bb_x_min < pbb_x_max) and (pbb_x_min < bb_x_max) and (bb_y_min < pbb_y_max) and (pbb_y_min < bb_y_max):
            no_bb_overlap = False
            break
    return no_bb_overlap


def place_card(image, card, mask, overlapping, bounding_boxes):
    card_placed = False
    max_tries = 50
    tries = 0
    image_height, image_width, _ = image.shape
    card_height, card_width, _ = card.shape

    # try to place card, if no success (due to overlapping) stop after max_tries
    while (not card_placed) and (tries < max_tries):
        x_pos = rand.randint(0, image_width-1)
        y_pos = rand.randint(0, image_height-1)
        proposed_bounding_box = (x_pos, y_pos, card_width, card_height)

        if ((no_bounding_boxes_overlap(bounding_boxes, proposed_bounding_box, image)) or overlapping):
            # adjust bounding box, if it reaches over the edges of the image
            bounding_box, card, mask = get_adjusted_bounding_box_and_card_and_mask(proposed_bounding_box, card, mask, image)
            bounding_boxes.append(bounding_box)
            image = overlay_images(image, card, mask, bounding_box)
            card_placed = True
        else:
            tries += 1
    return image, bounding_boxes


def transform_card(image, card, size, rotation, alpha, beta):
    # create mask before resizing card, because low quality images do not allow consistent mask-creation
    card_mask = ppf.create_mask(card)

    image_height, _, _ = image.shape
    ratio = card.shape[1]/card.shape[0]
    card_height = int(size*image_height)
    card_width = int(card_height*ratio)

    card = cv.resize(card, (card_width, card_height))
    card = imutils.rotate_bound(card, rotation)
    # randomize brightness of image
    card  = cv.convertScaleAbs(card, alpha=alpha, beta=beta)
    card_mask = cv.resize(card_mask, (card_width, card_height))
    card_mask = imutils.rotate_bound(card_mask, rotation)
    return card, card_mask


def place_cards(background, cards, card_classes, max_size, min_size, overlapping):
    image = background
    bounding_boxes = list()
    for card in cards:
        size = rand.uniform(min_size, max_size)
        rotation = rand.uniform(0, 360)
        # alpha and beta to change image brightness in next step
        alpha = rand.uniform(0.5, 1.5)
        beta = rand.uniform(-50, 50)

        transformed_card, transformed_card_mask = transform_card(image, card, size, rotation, alpha, beta)
        image, bounding_boxes = place_card(image, transformed_card, transformed_card_mask, overlapping, bounding_boxes)

    # create labels in the format needed by YOLO
    labels = ''
    for i, bounding_box in enumerate(bounding_boxes):
        bounding_box = transform_coordinates_to_relative_values(bounding_box, image)
        labels += str(card_classes[i]) + ' ' + ' '.join([str(data) for data in bounding_box]) + '\n'
    return image, labels


def select_cards(PLAYING_CARDS_DIR, number_of_cards):
    card_paths = [str(path) for path in Path(PLAYING_CARDS_DIR).glob('*')]
    selected_cards_paths = rand.sample(card_paths, number_of_cards)
    selected_cards = [cv.imread(path) for path in selected_cards_paths]
    # splits to get name of cards without .jpg from path
    selected_cards_names = [card_path.split('/')[-1].split('.')[0] for card_path in selected_cards_paths]
    return selected_cards, selected_cards_names


def select_background(BACKGROUNDS_DIR):
    background_paths = [str(path) for path in Path(BACKGROUNDS_DIR).glob('**/*.*')]
    selected_background_path = str(rand.sample(background_paths, 1)[0])
    background = cv.imread(selected_background_path)
    #try except because of some additional files in the downloaded dtd, that have to be skipped
    try:
        #resize images to common YOLO input size
        cropped_background = cv.resize(background, (640,640))
    except:
        cropped_background = select_background(BACKGROUNDS_DIR)
    return cropped_background


def generate_yaml_file(PLAYING_CARDS_DIR, OUTPUT_DIR):
    card_names = [frame_name.split('.')[0] for frame_name in os.listdir(PLAYING_CARDS_DIR)]
    name_to_int_dict = dict((name, i) for i, name in enumerate(card_names))

    with open(OUTPUT_DIR + '/data.yaml', 'w') as file:
        file.write(f'path: {OUTPUT_DIR}\n')
        file.write('train: train/images\n')
        file.write('val: val/images\n')
        file.write('test: test/images\n')
        file.write('names:\n')
        for i, card_name in enumerate(card_names):
            file.write(f'  {i}: {card_name}\n')
    return name_to_int_dict


def create_dataset_dir(OUTPUT_DIR):
    subdirs_to_create = ['/train/images/', '/train/labels/', '/val/images/',
                         '/val/labels/', '/test/images/', '/test/labels/']
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    for subdir in subdirs_to_create:
        FULL_SUBDIR = OUTPUT_DIR + subdir
        if not os.path.exists(FULL_SUBDIR):
            os.makedirs(FULL_SUBDIR)
    

def generate_dataset(BACKGROUNDS_DIR, PHOTOS_DIR, OUTPUT_DIR, number_of_images, max_number_of_cards_per_image,
                     min_size, max_size, overlapping, seed):
    rand.seed(seed)
    current_dataset_split = '/train'

    PLAYING_CARDS_DIR = ppf.process_photos(PHOTOS_DIR)
    create_dataset_dir(OUTPUT_DIR)
    # name_to_int_dict matches a card to an integer, because the labels need an integer and not the actual name
    name_to_int_dict = generate_yaml_file(PLAYING_CARDS_DIR, OUTPUT_DIR)

    print(f'Generating {number_of_images} dataset images...')
    for i in tqdm(range(number_of_images)):
        background = select_background(BACKGROUNDS_DIR)
        number_of_cards_per_image = rand.randint(1, max_number_of_cards_per_image)
        cards, names = select_cards(PLAYING_CARDS_DIR, number_of_cards_per_image)
        card_classes = [name_to_int_dict[name] for name in names]
        image, labels = place_cards(background, cards, card_classes, max_size, min_size, overlapping)

        if((i >= 0.8*number_of_images) and (i < 0.95*number_of_images)):
            current_dataset_split = '/val'
        elif(i >= 0.95*number_of_images):
            current_dataset_split = '/test'
        cv.imwrite(OUTPUT_DIR + current_dataset_split + f'/images/{i}.jpg', image)
        with open(OUTPUT_DIR + current_dataset_split + f'/labels/{i}.txt', 'w') as file:
            file.write(labels)
    print(f'Dataset generated and saved at: "{OUTPUT_DIR}"!')
    return OUTPUT_DIR


if __name__ == '__main__':
    #input parameters
    BACKGROUNDS_DIR = '/home/moiki/Documents/Files/studies/4_Semester/ADL/ADL_W23/data/dtd'
    PHOTOS_DIR = '/home/moiki/Documents/Files/studies/4_Semester/ADL/ADL_W23/data/photos'
    OUTPUT_DIR = '/home/moiki/Documents/Files/studies/4_Semester/ADL/ADL_W23/data/dataset_non_overlapping'

    number_of_images = 60000
    max_number_of_cards_per_image = 4
    min_size = 0.2
    max_size = 0.7
    overlapping = False
    seed = 42

    # generate dataset
    OUTPUT_DIR = generate_dataset(BACKGROUNDS_DIR, PHOTOS_DIR, OUTPUT_DIR, number_of_images, max_number_of_cards_per_image, 
                                  min_size, max_size, overlapping, seed)



