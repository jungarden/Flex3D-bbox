import os

# Set the path to the base folder
base_dir = '/path/to/base_dir/'

# Generate a list of files in the trainbox folder
name_list = os.listdir(base_dir)

# List of image file extensions to process
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']

# Perform slicing and renaming of file names
for file_name in name_list:
    # Create the full path for the file
    old_file_path = os.path.join(base_dir, file_name)
    
    # Separate the file extension
    file_base, file_ext = os.path.splitext(file_name)
    
    if file_ext.lower() in image_extensions:
        # Create a new file name (slice the name from index 2 to the end)
        new_file_name = file_name[2:]
        
        # Create the full path for the new file name
        new_file_path = os.path.join(base_dir, new_file_name)
        
        # Rename the file
        os.rename(old_file_path, new_file_path)

print("File renaming complete!")
