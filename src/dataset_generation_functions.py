import imgaug
import cv2 as cv
from pathlib import Path
import os
import random as rand


def select_background(backgrounds_path):
    background_paths = [str(path) for path in Path(backgrounds_path).glob('**/*.jpg')]
    selected_background_path = rand.sample(background_paths, 1)[0]
    background = cv.imread(selected_background_path)
    # cv.imshow('background', background)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    return background


def select_cards(cards_path, number_of_cards):
    card_paths = [str(path) for path in Path(cards_path).glob('*')]
    selected_cards_paths = rand.sample(card_paths, number_of_cards)
    selected_cards_names = [card_path.split('/')[-1].split('_')[0] for card_path in selected_cards_paths]

    cards = []
    for path in selected_cards_paths:
        card = cv.imread(path)
        cards.append(card)

    # cv.imshow('card 1', cards[0])
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    return cards, selected_cards_names


def generate_yaml_file(PLAYING_CARDS_DIR, GENERATED_DATASET_DIR):
    video_names = [str(path).split('/')[-1].split('.')[0] for path in Path(PLAYING_CARDS_DIR).glob('*')]
    video_list_unique = list(set(video_names))
    video_list_unique.sort()

    classes_dict = dict((name, i) for i, name in enumerate(video_list_unique))
    number_of_classes = len(video_list_unique)

    with open(GENERATED_DATASET_DIR + '/classes.yaml', 'w') as file:
        file.write('classes:\n')
        for name in video_list_unique:
            file.write('- ' + name + '\n')
        file.write(f'nc: {number_of_classes}')
    
    return classes_dict

def place_card(background, card, size, rotation, angle):
    #todo
    boundingbox = 0
    image = 0
    return image, boundingbox


def bb_overlap(bounding_boxes, bounding_box):
    #todo
    pass


def add_new_bb(bounding_boxes, bounding_box):
    #todo
    pass




def place_cards(background, cards, classes, max_size, min_size, max_rotation = 0, max_angle = 0, overlapping = False):
    background_hight, background_width, _ = background.shape
    card_hight, card_width, _ = cards[0].shape
    image = background

    bounding_boxes = list()

    while i in range(len(cards)):
        size = rand.uniform(min_size, max_size)
        rotation = rand.uniform(0, max_rotation)
        angle = rand.uniform(max_angle)

        image_proposal, bounding_box_proposal = place_card(image, size, rotation, angle)
        
        if (bb_overlap(bounding_boxes, bounding_box_proposal)) and (not overlapping):
            i -= 1
        else:
            image = image_proposal
            bounding_boxes = add_new_bb(bounding_boxes, bounding_box_proposal)

    labels = ''
    for i, bounding_box in enumerate(bounding_boxes):
        labels += classes[i] + ' ' + ' '.join(bounding_box) + '\n'
    return image, labels


def generate_dataset(BACKGROUNDS_DIR, PLAYING_CARDS_DIR, PROCESSED_PLAYING_CARDS_DIR, GENERATED_DATASET_DIR, number_of_images, max_number_of_cards, min_size, max_size, max_rotation, max_angle, overlapping, seed):
    rand.seed(seed)

    if not os.path.exists(GENERATED_DATASET_DIR):
        os.makedirs(GENERATED_DATASET_DIR)

    DATASET_IMAGES_DIR = GENERATED_DATASET_DIR + '/images/'
    if not os.path.exists(DATASET_IMAGES_DIR):
        os.makedirs(DATASET_IMAGES_DIR)
    
    DATASET_LABELS_DIR = GENERATED_DATASET_DIR + '/labels/'
    if not os.path.exists(DATASET_LABELS_DIR):
        os.makedirs(DATASET_LABELS_DIR)

    classes_dict = generate_yaml_file(PLAYING_CARDS_DIR, GENERATED_DATASET_DIR)

    for i in range(number_of_images):
        background = select_background(BACKGROUNDS_DIR)
        number_of_cards = rand.randint(1, max_number_of_cards)
        cards, names = select_cards(PROCESSED_PLAYING_CARDS_DIR, number_of_cards)
        classes = [classes_dict[name] for name in names]
        # image, labels = place_cards(background, cards, classes, max_size, min_size, max_rotation, max_angle, overlapping)

        # cv.imwrite(GENERATED_DATASET_DIR + '/images/' + i + '.jpeg', image)
        # with open(GENERATED_DATASET_DIR + '/labels/' + i + '.txt', 'w') as file:
        #     file.write(labels)

    return GENERATED_DATASET_DIR



if __name__ == '__main__':
    relative_path_videos = '/data/sg_cards'
    relative_path_cards = '/data/sg_cards_frames_processed'
    relative_path_backgrounds = '/data/dtd'
    relative_path_dataset = '/data/sg_cards_dataset_1'
    seed = 1
    number_of_images = 2
    max_number_of_cards = 3
    overlapping = True
    max_size = 0.4
    min_size = 0.1
    max_rotation = 90
    max_angle = 45
    #color transformations,...?

    PROJECT_DIR = Path(__file__).parent.parent
    BACKGROUNDS_DIR = str(PROJECT_DIR) + relative_path_backgrounds
    PLAYING_CARDS_DIR = str(PROJECT_DIR) + relative_path_cards
    PROCESSED_PLAYING_CARDS_DIR = str(PROJECT_DIR) + relative_path_cards
    GENERATED_DATASET_DIR = str(PROJECT_DIR) + relative_path_dataset
    
    generate_dataset(BACKGROUNDS_DIR, PLAYING_CARDS_DIR, PROCESSED_PLAYING_CARDS_DIR, GENERATED_DATASET_DIR, number_of_images, max_number_of_cards,
                     min_size, max_size, max_rotation, max_angle, overlapping, seed)





