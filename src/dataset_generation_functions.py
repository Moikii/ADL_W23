"""
This file contains function definitions used to create the dataset in the YOLO-format.
A image consists of a randomly selected background and a random number of playing cards
placed on it in also randomly selected positions, rotations and brightness.
Additional we have a parameter to decide if the generated images should contain overlapping
cards (making detection harder).
We also create the data.yaml file, which contains information for the YOLO model during training.
The dataset is saved in a directory given by the user.
"""

import cv2 as cv
from pathlib import Path
import os
import random as rand
import imutils
import numpy as np
import photo_preparation_functions as ppf
from tqdm import tqdm
from typing import Tuple, List, Dict


def transform_coordinates_to_relative_values(bounding_box: Tuple, image) -> Tuple:
    """
    Returns normalized bounding box center, width and height, calculated from pixel coordinates.
    """
    image_height, image_width, _ = image.shape
    x_pos = bounding_box[0] / image_width
    y_pos = bounding_box[1] / image_height
    width = bounding_box[2] / image_width
    height = bounding_box[3] / image_height
    return (x_pos, y_pos, width, height)


def overlay_images(image, card, mask, bounding_box: Tuple):
    """
    Place a playing card on the background. Returns the image with newly placed card on it.
    """
    card_height, card_width, _ = card.shape
    # resize card to even number of pixels (avoid rounding errors later)
    card = cv.resize(card, (card_width + card_width % 2, card_height + card_height % 2))
    mask = cv.resize(mask, (card_width + card_width % 2, card_height + card_height % 2))

    x_min, x_max, y_min, y_max = get_min_max_bounding_box_coordinates(bounding_box)
    card_placing_area = image[y_min:y_max, x_min:x_max]

    mask = cv.dilate(mask, np.ones((3, 3), np.uint8))
    image_foreground = cv.bitwise_or(card, card, mask=mask)
    inv_mask = cv.bitwise_not(mask)
    image_background = cv.bitwise_or(
        card_placing_area, card_placing_area, mask=inv_mask
    )
    area_with_placed_card = cv.bitwise_or(image_background, image_foreground)
    image[y_min:y_max, x_min:x_max] = area_with_placed_card
    return image


def get_adjusted_bounding_box_and_card_and_mask(
    proposed_bounding_box: Tuple, card, mask, image
) -> Tuple:
    """
    Return adjusted bounding box, card and mask, in case the position is on the edge of the backgound.
    """
    image_height, image_width, _ = image.shape
    card_height, card_width, _ = card.shape
    pbb_x_min, pbb_x_max, pbb_y_min, pbb_y_max = get_min_max_bounding_box_coordinates(
        proposed_bounding_box
    )

    # get number of pixels that stand over the image edges for each side
    x_adjustment_max = max(0, pbb_x_max - image_width)
    x_adjustment_min = min(0, pbb_x_min)
    y_adjustment_max = max(0, pbb_y_max - image_height)
    y_adjustment_min = min(0, pbb_y_min)
    # adjust bounding box accordingly
    adjusted_x_pos = int(
        proposed_bounding_box[0]
        - np.ceil(x_adjustment_max / 2)
        - np.floor(x_adjustment_min / 2)
    )
    adjusted_y_pos = int(
        proposed_bounding_box[1]
        - np.ceil(y_adjustment_max / 2)
        - np.floor(y_adjustment_min / 2)
    )
    adjusted_width = proposed_bounding_box[2] - x_adjustment_max + x_adjustment_min
    adjusted_height = proposed_bounding_box[3] - y_adjustment_max + y_adjustment_min
    # crop card and card-mask to fit the new bounding box
    adjusted_card = card[
        (-y_adjustment_min) : (card_height - y_adjustment_max),
        (-x_adjustment_min) : (card_width - x_adjustment_max),
    ]
    adjusted_mask = mask[
        -y_adjustment_min : (card_height - y_adjustment_max),
        -x_adjustment_min : (card_width - x_adjustment_max),
    ]

    return (
        (adjusted_x_pos, adjusted_y_pos, adjusted_width, adjusted_height),
        adjusted_card,
        adjusted_mask,
    )


def get_min_max_bounding_box_coordinates(bounding_box: Tuple) -> Tuple:
    """
    Return bounding box defined by min and max values instead of center, width and height.
    """
    bb_x_min = int(np.floor(bounding_box[0] - bounding_box[2] / 2))
    bb_x_max = int(np.ceil(bounding_box[0] + bounding_box[2] / 2))
    bb_y_min = int(np.floor(bounding_box[1] - bounding_box[3] / 2))
    bb_y_max = int(np.ceil(bounding_box[1] + bounding_box[3] / 2))
    return bb_x_min, bb_x_max, bb_y_min, bb_y_max


def no_bounding_boxes_overlap(
    bounding_boxes: List, proposed_bounding_box: Tuple, image
) -> bool:
    """
    Return a bool that indicates if the new bounding box overlapps with the existing ones.
    """
    no_bb_overlap = True
    image_height, image_width, _ = image.shape
    pbb_x_min, pbb_x_max, pbb_y_min, pbb_y_max = get_min_max_bounding_box_coordinates(
        proposed_bounding_box
    )

    # check if new bounding box overlaps image edges
    if (
        (pbb_x_min < 0)
        or (pbb_x_max > image_width)
        or (pbb_y_min < 0)
        or (pbb_y_max > image_height)
    ):
        no_bb_overlap = False
    # check if new bounding box overlaps other bounding boxes
    for bounding_box in bounding_boxes:
        bb_x_min, bb_x_max, bb_y_min, bb_y_max = get_min_max_bounding_box_coordinates(
            bounding_box
        )
        if (
            (bb_x_min < pbb_x_max)
            and (pbb_x_min < bb_x_max)
            and (bb_y_min < pbb_y_max)
            and (pbb_y_min < bb_y_max)
        ):
            no_bb_overlap = False
            break
    return no_bb_overlap


def place_card(image, card, mask, overlapping: bool, bounding_boxes: List) -> Tuple:
    """
    Return a tuple containing the image with a newly placed card and the updated list of bounding boxes.
    """
    card_placed = False
    max_tries = 50
    tries = 0
    image_height, image_width, _ = image.shape
    card_height, card_width, _ = card.shape

    # try to place card, if no success (due to overlapping) stop after max_tries
    while (not card_placed) and (tries < max_tries):
        x_pos = rand.randint(0, image_width - 1)
        y_pos = rand.randint(0, image_height - 1)
        proposed_bounding_box = (x_pos, y_pos, card_width, card_height)

        if (
            no_bounding_boxes_overlap(bounding_boxes, proposed_bounding_box, image)
        ) or overlapping:
            # adjust bounding box, if it reaches over the edges of the image
            bounding_box, card, mask = get_adjusted_bounding_box_and_card_and_mask(
                proposed_bounding_box, card, mask, image
            )
            bounding_boxes.append(bounding_box)
            image = overlay_images(image, card, mask, bounding_box)
            card_placed = True
        else:
            tries += 1
    return image, bounding_boxes


def transform_card(
    image, card, size: float, rotation: float, alpha: float, beta: float
) -> Tuple:
    """
    Return a tuple containing a transformed card and the corresponding mask.
    """
    card_mask = ppf.create_mask(card)

    image_height, _, _ = image.shape
    ratio = card.shape[1] / card.shape[0]
    card_height = int(size * image_height)
    card_width = int(card_height * ratio)

    card = cv.resize(card, (card_width, card_height))
    card = imutils.rotate_bound(card, rotation)
    card = cv.convertScaleAbs(card, alpha=alpha, beta=beta)
    card_mask = cv.resize(card_mask, (card_width, card_height))
    card_mask = imutils.rotate_bound(card_mask, rotation)
    return card, card_mask


def place_cards(
    background,
    cards: List,
    card_classes: List,
    max_size: float,
    min_size: float,
    overlapping: bool,
) -> Tuple:
    """
    Place selected cards on the background. Returns a tuple containing this image and a list with the according bounding boxes.
    """
    image = background
    bounding_boxes = list()
    for card in cards:
        size = rand.uniform(min_size, max_size)
        rotation = rand.uniform(0, 360)
        # alpha and beta to change image brightness in next step
        alpha = rand.uniform(0.5, 1.5)
        beta = rand.uniform(-50, 50)

        transformed_card, transformed_card_mask = transform_card(
            image, card, size, rotation, alpha, beta
        )
        image, bounding_boxes = place_card(
            image, transformed_card, transformed_card_mask, overlapping, bounding_boxes
        )

    # create labels in the format needed by YOLO
    labels = ""
    for i, bounding_box in enumerate(bounding_boxes):
        bounding_box = transform_coordinates_to_relative_values(bounding_box, image)
        labels += (
            str(card_classes[i])
            + " "
            + " ".join([str(data) for data in bounding_box])
            + "\n"
        )
    return image, labels


def select_cards(PLAYING_CARDS_DIR: str, number_of_cards: int) -> Tuple[List, List]:
    """
    Returns a tuple containing a list with the selected cards and a list with the according names.
    Images are randomly sampled from the images contained in the directory.
    """
    card_paths = [str(path) for path in Path(PLAYING_CARDS_DIR).glob("*")]
    selected_cards_paths = rand.sample(card_paths, number_of_cards)
    selected_cards = [cv.imread(path) for path in selected_cards_paths]

    selected_cards_names = [
        card_path.split("/")[-1].split(".")[0] for card_path in selected_cards_paths
    ]  # splits to get name of cards without .jpg from path
    return selected_cards, selected_cards_names


def select_background(BACKGROUNDS_DIR: str):
    """
    Returns a randomly sampled, resized image from the inputfolder.
    """
    background_paths = [str(path) for path in Path(BACKGROUNDS_DIR).glob("**/*.*")]
    selected_background_path = str(rand.sample(background_paths, 1)[0])
    background = cv.imread(selected_background_path)

    try:  # try except because of some additional files in the downloaded dtd, that have to be skipped
        cropped_background = cv.resize(
            background, (640, 640)
        )  # resize images to common YOLO input size
    except:
        cropped_background = select_background(BACKGROUNDS_DIR)
    return cropped_background


def generate_yaml_file(PLAYING_CARDS_DIR: str, OUTPUT_DIR: str) -> Dict[str, int]:
    """
    Generate the meta-data for YOLO training. Returns a dictionary with the mapping of card name to an label.
    """
    card_names = [
        frame_name.split(".")[0] for frame_name in os.listdir(PLAYING_CARDS_DIR)
    ]
    name_to_int_dict = dict((name, i) for i, name in enumerate(card_names))

    with open(OUTPUT_DIR + "/data.yaml", "w") as file:
        file.write(f"path: {OUTPUT_DIR}\n")
        file.write("train: train/images\n")
        file.write("val: val/images\n")
        file.write("test: test/images\n")
        file.write("names:\n")
        for i, card_name in enumerate(card_names):
            file.write(f"  {i}: {card_name}\n")
    return name_to_int_dict


def create_dataset_dir(OUTPUT_DIR: str) -> None:
    """
    Creates the folder structure for generated images if it does not exist already.
    """
    subdirs_to_create = [
        "/train/images/",
        "/train/labels/",
        "/val/images/",
        "/val/labels/",
        "/test/images/",
        "/test/labels/",
    ]
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    for subdir in subdirs_to_create:
        FULL_SUBDIR = OUTPUT_DIR + subdir
        if not os.path.exists(FULL_SUBDIR):
            os.makedirs(FULL_SUBDIR)


def generate_dataset(
    BACKGROUNDS_DIR: str,
    PHOTOS_DIR: str,
    OUTPUT_DIR: str,
    number_of_images: int,
    max_number_of_cards_per_image: int,
    min_size: float,
    max_size: float,
    overlapping: bool,
    seed: int,
) -> str:
    """
    Generatete dataset to train, validate and test a YOLO model. Returns the output directory.
    """
    rand.seed(seed)
    current_dataset_split = "/train"

    PLAYING_CARDS_DIR = ppf.process_photos(PHOTOS_DIR)
    create_dataset_dir(OUTPUT_DIR)
    name_to_int_dict = generate_yaml_file(PLAYING_CARDS_DIR, OUTPUT_DIR)

    print(f"Generating {number_of_images} dataset images...")
    for i in tqdm(range(number_of_images)):
        background = select_background(BACKGROUNDS_DIR)
        number_of_cards_per_image = rand.randint(1, max_number_of_cards_per_image)
        cards, names = select_cards(PLAYING_CARDS_DIR, number_of_cards_per_image)
        card_classes = [name_to_int_dict[name] for name in names]
        image, labels = place_cards(
            background, cards, card_classes, max_size, min_size, overlapping
        )

        if (i >= 0.8 * number_of_images) and (i < 0.95 * number_of_images):
            current_dataset_split = "/val"
        elif i >= 0.95 * number_of_images:
            current_dataset_split = "/test"
        cv.imwrite(OUTPUT_DIR + current_dataset_split + f"/images/{i}.jpg", image)
        with open(OUTPUT_DIR + current_dataset_split + f"/labels/{i}.txt", "w") as file:
            file.write(labels)
    print(f'Dataset generated and saved at: "{OUTPUT_DIR}"!')
    return OUTPUT_DIR
