import photo_preparation_functions as ppf
import dataset_generation_functions as dgf

# input parameters
PHOTOS_DIR = '/home/moiki/Documents/Files/studies/4_Semester/ADL/ADL_W23/data/photos'
BACKGROUNDS_DIR = '/home/moiki/Documents/Files/studies/4_Semester/ADL/ADL_W23/data/dtd'
OUTPUT_DIR = '/home/moiki/Documents/Files/studies/4_Semester/ADL/ADL_W23/data/dataset'

number_of_images = 100000
max_number_of_cards_per_image = 4
min_size = 0.2
max_size = 0.7
max_rotation = 360
overlapping = True
seed = 42

# process photos of cards
PLAYING_CARDS_DIR = ppf.process_photos(PHOTOS_DIR)

# generate dataset from processed images
OUTPUT_DIR = dgf.generate_dataset(BACKGROUNDS_DIR, PLAYING_CARDS_DIR, OUTPUT_DIR, number_of_images, max_number_of_cards_per_image,
                    min_size, max_size, max_rotation, overlapping, seed)