import cv2 as cv
from pathlib import Path
import os
import random as rand
import imutils
import numpy as np
import photo_preparation_functions as ppf
from tqdm import tqdm


def overlay_images(image, card, mask, bounding_box):
    card_height, card_width, _ = card.shape
    card = cv.resize(card, (card_width +card_width%2, card_height + card_height%2))
    mask = cv.resize(mask, (card_width +card_width%2, card_height + card_height%2))

    x_min, x_max, y_min, y_max = get_min_max_borders(bounding_box)
    roi = image[y_min:y_max, x_min:x_max]

    # print(f'new bounding box: {bounding_box}')
    # print(f'new edges: {x_min, x_max, y_min, y_max}')
    # print(f'roi: {roi.shape}')
    # print(f'mask: {mask.shape}')
    #cv.rectangle(image,(x_min,y_min),(x_max,y_max),(0,255,0),2)

    mask = cv.dilate(mask, np.ones((3,3), np.uint8))
    image_foreground = cv.bitwise_or(card, card, mask = mask)
    inv_mask = cv.bitwise_not(mask)
    image_background = cv.bitwise_or(roi, roi, mask = inv_mask)
    dst = cv.bitwise_or(image_background, image_foreground)
    #dst = cv.add(image_background,image_foreground)
    image[y_min:y_max, x_min:x_max] = dst

    return image


def create_dataset_dir(OUTPUT_DIR):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    DATASET_TRAIN_DIR = OUTPUT_DIR + '/train/images/'
    if not os.path.exists(DATASET_TRAIN_DIR):
        os.makedirs(DATASET_TRAIN_DIR)

    DATASET_VAL_DIR = OUTPUT_DIR + '/train/labels/'
    if not os.path.exists(DATASET_VAL_DIR):
        os.makedirs(DATASET_VAL_DIR)

    DATASET_IMAGES_DIR = OUTPUT_DIR + '/val/images/'
    if not os.path.exists(DATASET_IMAGES_DIR):
        os.makedirs(DATASET_IMAGES_DIR)
    
    DATASET_LABELS_DIR = OUTPUT_DIR + '/val/labels/'
    if not os.path.exists(DATASET_LABELS_DIR):
        os.makedirs(DATASET_LABELS_DIR)
    


def generate_yaml_file(PLAYING_CARDS_DIR, OUTPUT_DIR):
    card_names = [frame_name.split('.')[0] for frame_name in os.listdir(PLAYING_CARDS_DIR)]
    unique_card_names = list(set(card_names))
    unique_card_names.sort()
    classes_dict = dict((name, i) for i, name in enumerate(unique_card_names))

    with open(OUTPUT_DIR + '/data.yaml', 'w') as file:
        file.write(f'path: {OUTPUT_DIR}\n')
        file.write('train: train/images\n')
        file.write('val: val/images\n')
        file.write('names:\n')
        for i, card_name in enumerate(unique_card_names):
            file.write(f'  {i}: {card_name}\n')
    return classes_dict


def select_background(BACKGROUNDS_DIR):
    background_paths = [str(path) for path in Path(BACKGROUNDS_DIR).glob('**/*.*')]
    selected_background_path = str(rand.sample(background_paths, 1)[0])
    background = cv.imread(selected_background_path)
    try:
        cropped_background = cv.resize(background, (640,640))
    except:
        #print(selected_background_path)
        cropped_background = select_background(BACKGROUNDS_DIR)
    return cropped_background


def select_cards(PLAYING_CARDS_DIR, number_of_cards):
    card_paths = [str(path) for path in Path(PLAYING_CARDS_DIR).glob('*')]
    selected_cards_paths = rand.sample(card_paths, number_of_cards)
    selected_cards = [cv.imread(path) for path in selected_cards_paths]
    selected_cards_names = [card_path.split('/')[-1].split('.')[0] for card_path in selected_cards_paths]
    return selected_cards, selected_cards_names


def transform_card(image, card, size, rotation, alpha, beta):
    card_mask = ppf.create_mask(card)

    image_height, _, _ = image.shape
    ratio = card.shape[1]/card.shape[0]
    card_height = int(size*image_height)
    card_width = int(card_height*ratio)

    card = cv.resize(card, (card_width, card_height))
    card = imutils.rotate_bound(card, rotation)
    card  = cv.convertScaleAbs(card, alpha=alpha, beta=beta)

    card_mask = cv.resize(card_mask, (card_width, card_height))
    card_mask = imutils.rotate_bound(card_mask, rotation)

    return card, card_mask


def transform_coordinates_to_relative_values(bounding_box, image):
    image_height, image_width, _ = image.shape

    x_pos = bounding_box[0]/image_width
    y_pos = bounding_box[1]/image_height
    width = bounding_box[2]/image_width
    height = bounding_box[3]/image_height
    return (x_pos, y_pos, width, height)


def place_cards(background, cards, card_classes, max_size, min_size, max_rotation, overlapping):
    image = background
    bounding_boxes = list()
    for card in cards:
        size = rand.uniform(min_size, max_size)
        rotation = rand.uniform(0, max_rotation)
        alpha = rand.uniform(0.5, 1.5)
        beta = rand.uniform(-50, 50)

        transformed_card, transformed_card_mask = transform_card(image, card, size, rotation, alpha, beta)
        image, bounding_boxes = place_card(image, transformed_card, transformed_card_mask, overlapping, bounding_boxes)

    labels = ''
    for i, bounding_box in enumerate(bounding_boxes):
        bounding_box = transform_coordinates_to_relative_values(bounding_box, image)
        labels += str(card_classes[i]) + ' ' + ' '.join([str(data) for data in bounding_box]) + '\n'
    return image, labels


def get_min_max_borders(bounding_box):
    bb_x_min = int(np.floor(bounding_box[0] - bounding_box[2]/2))
    bb_x_max = int(np.ceil(bounding_box[0] + bounding_box[2]/2))
    bb_y_min = int(np.floor(bounding_box[1] - bounding_box[3]/2))
    bb_y_max = int(np.ceil(bounding_box[1] + bounding_box[3]/2))
    return bb_x_min, bb_x_max, bb_y_min, bb_y_max


def no_bounding_boxes_overlap(bounding_boxes, proposed_bounding_box, image):
    no_bb_overlap = True
    image_height, image_width, _ = image.shape
    pbb_x_min, pbb_x_max, pbb_y_min, pbb_y_max = get_min_max_borders(proposed_bounding_box)

    # overlaps image edges
    if (pbb_x_min < 0) or (pbb_x_max > image_width) or (pbb_y_min < 0) or (pbb_y_max > image_height):
        no_bb_overlap = False

    # overlaps other bounding boxes
    for bounding_box in bounding_boxes:
        bb_x_min, bb_x_max, bb_y_min, bb_y_max = get_min_max_borders(bounding_box)
        if (bb_x_min < pbb_x_max) and (pbb_x_min < bb_x_max) and (bb_y_min < pbb_y_max) and (pbb_y_min < bb_y_max):
            no_bb_overlap = False
            break
    return no_bb_overlap


def get_adjusted_bounding_box_and_card_and_mask(proposed_bounding_box, card, mask, image):
    image_height, image_width, _ = image.shape
    card_height, card_width, _ = card.shape    

    pbb_x_min, pbb_x_max, pbb_y_min, pbb_y_max = get_min_max_borders(proposed_bounding_box)

    x_adjustment_max = max(0, pbb_x_max - image_width)
    x_adjustment_min = min(0, pbb_x_min)
    y_adjustment_max = max(0, pbb_y_max - image_height)
    y_adjustment_min = min(0, pbb_y_min)

    adjusted_x_pos = int(proposed_bounding_box[0] - np.ceil(x_adjustment_max/2) - np.floor(x_adjustment_min/2))
    adjusted_y_pos = int(proposed_bounding_box[1] - np.ceil(y_adjustment_max/2) - np.floor(y_adjustment_min/2))
    adjusted_width = proposed_bounding_box[2] - x_adjustment_max + x_adjustment_min
    adjusted_height = proposed_bounding_box[3] - y_adjustment_max + y_adjustment_min

    adjusted_card = card[(-y_adjustment_min):(card_height - y_adjustment_max), (-x_adjustment_min):(card_width - x_adjustment_max)]
    adjusted_mask = mask[-y_adjustment_min:(card_height - y_adjustment_max), -x_adjustment_min:(card_width - x_adjustment_max)]

    # print(f'image data: {image_height, image_width}')
    # print(f'card data: {card_height, card_width}')
    # print(f'old bounding box: {proposed_bounding_box}')
    # print(f'old edges: {pbb_x_min, pbb_x_max, pbb_y_min, pbb_y_max}')
    return (adjusted_x_pos, adjusted_y_pos, adjusted_width, adjusted_height), adjusted_card, adjusted_mask


def place_card(image, card, mask, overlapping, bounding_boxes):
    card_placed = False
    max_tries = 50
    image_height, image_width, _ = image.shape
    card_height, card_width, _ = card.shape

    tries = 0
    while (not card_placed) and (tries < max_tries):
        x_pos = rand.randint(0, image_width-1)
        y_pos = rand.randint(0, image_height-1)
        proposed_bounding_box = (x_pos, y_pos, card_width, card_height)

        if ((no_bounding_boxes_overlap(bounding_boxes, proposed_bounding_box, image)) or overlapping):
            bounding_box, card, mask = get_adjusted_bounding_box_and_card_and_mask(proposed_bounding_box, card, mask, image)
            bounding_boxes.append(bounding_box)
            image = overlay_images(image, card, mask, bounding_box)
            card_placed = True
        else:
            tries += 1
    return image, bounding_boxes




def generate_dataset(BACKGROUNDS_DIR, PHOTOS_DIR, OUTPUT_DIR, number_of_images, max_number_of_cards_per_image,
                     min_size, max_size, max_rotation, overlapping, seed):
    PLAYING_CARDS_DIR = ppf.process_photos(PHOTOS_DIR)

    rand.seed(seed)
    folder = '/train'
    create_dataset_dir(OUTPUT_DIR)
    classes_dict = generate_yaml_file(PLAYING_CARDS_DIR, OUTPUT_DIR)

    print(f'Generating {number_of_images} dataset images...')
    for i in tqdm(range(number_of_images)):
        background = select_background(BACKGROUNDS_DIR)
        number_of_cards_per_image = rand.randint(1, max_number_of_cards_per_image)
        cards, names = select_cards(PLAYING_CARDS_DIR, number_of_cards_per_image)
        card_classes = [classes_dict[name] for name in names]
        image, labels = place_cards(background, cards, card_classes, max_size, min_size, max_rotation, overlapping)

        if(i >= 0.8*number_of_images):
            folder = '/val'
        cv.imwrite(OUTPUT_DIR + folder + f'/images/{i}.jpg', image)
        with open(OUTPUT_DIR + folder + f'/labels/{i}.txt', 'w') as file:
            file.write(labels)

    print(f'Dataset generated and saved at: {OUTPUT_DIR}!')
    return OUTPUT_DIR





if __name__ == '__main__':
    #input parameters
    BACKGROUNDS_DIR = '/home/moiki/Documents/Files/studies/4_Semester/ADL/ADL_W23/data/dtd'
    PHOTOS_DIR = '/home/moiki/Documents/Files/studies/4_Semester/ADL/ADL_W23/data/photos'
    OUTPUT_DIR = '/home/moiki/Documents/Files/studies/4_Semester/ADL/ADL_W23/data/dataset'

    number_of_images = 100
    max_number_of_cards_per_image = 4
    min_size = 0.2
    max_size = 0.7
    max_rotation = 360
    overlapping = True
    seed = 42
    
    # generate dataset
    generate_dataset(BACKGROUNDS_DIR, PHOTOS_DIR, OUTPUT_DIR, number_of_images, max_number_of_cards_per_image,
                     min_size, max_size, max_rotation, overlapping, seed)





