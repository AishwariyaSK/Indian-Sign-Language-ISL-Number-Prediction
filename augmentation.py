import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img, save_img

# Paths for original data and where augmented data will be saved
original_data_dir = 'data'
augmented_data_dir = 'augmented_data'

# Create an instance of ImageDataGenerator with desired augmentations
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Loop through each class folder
for class_name in os.listdir(original_data_dir):
    class_path = os.path.join(original_data_dir, class_name)
    if not os.path.isdir(class_path):
        continue  # Skip non-directory files

    # Create the corresponding class folder in augmented_data if it doesn't exist
    augmented_class_path = os.path.join(augmented_data_dir, class_name)
    os.makedirs(augmented_class_path, exist_ok=True)
    
    # Loop through each image in the class folder
    for image_name in os.listdir(class_path):
        image_path = os.path.join(class_path, image_name)

        # Load image and convert it to array
        image = load_img(image_path)
        image_array = img_to_array(image)
        image_array = np.expand_dims(image_array, axis=0)  # Expand dimensions to make it (1, height, width, channels)
        
        # Generate and save augmented images
        save_prefix = os.path.splitext(image_name)[0]  # Use the image name as a prefix for the augmented images
        num_augmented_images = 5  # Number of augmented images to create per original image
        
        for i, batch in enumerate(datagen.flow(image_array, batch_size=1, save_prefix=save_prefix, save_format='jpeg')):
            augmented_image_path = os.path.join(augmented_class_path, f"{save_prefix}_aug_{i}.jpeg")
            augmented_image = array_to_img(batch[0])  # Convert the array back to image
            save_img(augmented_image_path, augmented_image)  # Save the augmented image
            
            if i >= num_augmented_images - 1:
                break  # Stop after generating the desired number of augmented images per original image

print("Augmentation complete!")
