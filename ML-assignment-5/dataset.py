import os

def rename_and_label_images(directory_path, label_dict):
    """
    This function renames and labels images in the given directory
    Args:
        directory_path (str): The path to the directory containing the images
        label_dict (dict): A dictionary mapping class names to integer labels
    """
    # Iterate over the subdirectories of the directory
    for subdirectory in os.listdir(directory_path):
        subdirectory_path = os.path.join(directory_path, subdirectory)
        if not os.path.isdir(subdirectory_path):
            continue
        # Iterate over the images in the subdirectory
        i = 1
        for image_name in os.listdir(subdirectory_path):
            image_path = os.path.join(subdirectory_path, image_name)
            # Get the label of the image from the label_dict
            label = label_dict[subdirectory]
            # Rename the image
            new_image_name = f"{label}_{i}.jpg"
            new_image_path = os.path.join(subdirectory_path, new_image_name)
            os.rename(image_path, new_image_path)
            i += 1

# The path to your train directory
train_directory_path = "C:/Users/dhruv/Documents/ML_assignment_5/dataset/train" 

# A dictionary mapping class names to integer labels
label_dict = {"dolphin": 0, "jellyfish": 1}

# Call the function to rename and label the images
rename_and_label_images(train_directory_path, label_dict)
