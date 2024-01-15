"""
Some basic functionality tests are conducted when executing this script. The commandline should give back an 'OK',
if everything works correctly. We do some testing of the preprocessing, like structure of the generated dataset
and the contained images, but also testing of the model-download and rule implementation of the game in the application.
"""

import dataset_generation_functions as dgf
import jass_rules as jass
import download_best_model as dbm
import cv2 as cv
from ultralytics import YOLO
import os
import shutil
import unittest
from glob import glob


class TestDatasetGeneration(unittest.TestCase):
    def test_dataset_structure(self) -> None:
        """
        Test if the folder structure is built correctly for further usage to train YOLOv8. Otherwise the test fails.
        """
        DATASET_DIR = str(os.getcwd()) + "/unittest_data/test_dataset"
        PROCESSED_PHOTOS_DIR = str(os.getcwd()) + "/unittest_data/test_photos_processed"
        _ = dgf.generate_dataset(
            BACKGROUNDS_DIR="./unittest_data/test_backgrounds",
            PHOTOS_DIR="./unittest_data/test_photos",
            OUTPUT_DIR=DATASET_DIR,
            number_of_images=5,
            max_number_of_cards_per_image=1,
            min_size=0.2,
            max_size=0.7,
            overlapping=True,
            seed=1,
        )
        subfolders = [
            subfolder.split("/")[-2] for subfolder in glob(DATASET_DIR + "**/*/")
        ]
        subfolders.sort()
        self.assertEqual(subfolders, ["test", "train", "val"])
        shutil.rmtree(DATASET_DIR)
        shutil.rmtree(PROCESSED_PHOTOS_DIR)

    def test_image_structure(self) -> None:
        """
        Test if generated images have the correct structure for the training process. Otherwise the test fails.
        """
        DATASET_DIR = str(os.getcwd()) + "/unittest_data/test_dataset"
        PROCESSED_PHOTOS_DIR = str(os.getcwd()) + "/unittest_data/test_photos_processed"
        _ = dgf.generate_dataset(
            BACKGROUNDS_DIR="./unittest_data/test_backgrounds",
            PHOTOS_DIR="./unittest_data/test_photos",
            OUTPUT_DIR=DATASET_DIR,
            number_of_images=5,
            max_number_of_cards_per_image=1,
            min_size=0.2,
            max_size=0.7,
            overlapping=True,
            seed=1,
        )
        img = cv.imread(DATASET_DIR + "/train/images/1.jpg")
        self.assertTupleEqual(img.shape, (640, 640, 3))
        shutil.rmtree(DATASET_DIR)
        shutil.rmtree(PROCESSED_PHOTOS_DIR)


class TestApplication(unittest.TestCase):
    def test_model_download(self) -> None:
        """
        Test if the trained model is available for download and is usable. Otherwise the test fails.
        """
        DATASET_DIR = str(os.getcwd()) + "/unittest_data/test_dataset"
        PROCESSED_PHOTOS_DIR = str(os.getcwd()) + "/unittest_data/test_photos_processed"
        _ = dgf.generate_dataset(
            BACKGROUNDS_DIR="./unittest_data/test_backgrounds",
            PHOTOS_DIR="./unittest_data/test_photos",
            OUTPUT_DIR=DATASET_DIR,
            number_of_images=5,
            max_number_of_cards_per_image=1,
            min_size=0.2,
            max_size=0.7,
            overlapping=True,
            seed=1,
        )
        try:
            MODEL_PATH = dbm.get_model()
            model = YOLO(MODEL_PATH)
            model(source=DATASET_DIR + "/train/images/1.jpg", conf=0.7, verbose=False)
        except:
            self.assertTrue(False)
        shutil.rmtree(DATASET_DIR)
        shutil.rmtree(PROCESSED_PHOTOS_DIR)

    def test_jass_scoring(self) -> None:
        """
        Test if the score keeping of Jassa works correctly, by playing a game and checking if sum of points is 157. Otherwise the test fails.
        """
        suits = ["h", "s", "e", "l"]
        values = ["6", "7", "8", "9", "x", "u", "o", "k", "a"]
        points = 0
        current_play = list()
        for i, suit in enumerate(suits):
            for j, value in enumerate(values):
                current_play.append(suit + value)
                if len(current_play) % 2 == 0:
                    points += jass.get_points(current_play, "h", i * len(values) + j, 2)
                    current_play = list()
        self.assertEqual(points, 157)


if __name__ == "__main__":
    unittest.main()
