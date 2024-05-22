import os

# Define the root directory
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define paths relative to the root directory
MODEL_DIR = os.path.join(ROOT_DIR, 'model')
RESOURCES_DIR = os.path.join(ROOT_DIR, 'resources')
TRAIN_DIR = os.path.join(RESOURCES_DIR, 'images', 'train', 'train')
VALIDATION_DIR = os.path.join(RESOURCES_DIR, 'images', 'valid', 'test')
