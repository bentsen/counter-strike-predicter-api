import os

import tensorflow as tf
import json

from chatbot.config import TRAIN_DIR, VALIDATION_DIR, RESOURCES_DIR


def create_image_generators(train_dir, validation_dir, target_size=(150, 150), batch_size=32):
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255.0, rotation_range=20,
                                                                    zoom_range=0.2, horizontal_flip=True)
    validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255.0)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    # Save class indices
    with open(os.path.join(RESOURCES_DIR, 'class_indices.json'), 'w') as f:
        json.dump(train_generator.class_indices, f)
        print("Saved class indices to class_indices.json")

    return train_generator, validation_generator


def build_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model


def compile_and_train_model(model, train_generator, validation_generator, epochs=20):
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator
    )

    return model, history


def save_model(model, file_path='./model/weapon_classifier_model.keras'):
    model.save(file_path)


def train_model():
    train_generator, validation_generator = create_image_generators(TRAIN_DIR, VALIDATION_DIR)
    model = build_model((150, 150, 3), train_generator.num_classes)
    model, history = compile_and_train_model(model, train_generator, validation_generator)
    save_model(model)


if __name__ == '__main__':
    train_model()
