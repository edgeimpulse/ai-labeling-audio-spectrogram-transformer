
noise_dir="./sounds/noise-samples"

# Iterate over all files in the output directory
for file in "$noise_dir"/*.wav; do
    # Get the current filename
    filename=$(basename "$file")
    
    # Replace the word in the filename
    new_filename="${filename/metal/damage}"
    
    # Rename the file with the new filename
    mv "$file" "$noise_dir/$new_filename"
done