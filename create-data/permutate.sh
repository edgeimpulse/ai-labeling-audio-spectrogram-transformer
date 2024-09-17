#!/bin/bash

# Path to the large .wav file
large_file="./sounds/ambiance-half.wav"

# Path to the directory containing the short noise files
noise_dir="./sounds/noise-samples"

# Output directory for the generated segments
output_dir="./final-dataset"

# Duration of each segment in seconds
segment_duration=20

# Percentage of segments to attach noise to (between 0 and 100)
noise_percentage=50

# Create the output directory if it doesn't exist
mkdir -p "$output_dir"

# Split the large file into segments
sox "$large_file" "$output_dir/segment_%03d.wav" trim 0 "$segment_duration" : newfile : restart

# Remove the last 15 generated samples
last_samples=("$output_dir"/segment_*.wav)
num_samples=${#last_samples[@]}
num_to_remove=15

if (( num_samples > num_to_remove )); then
    samples_to_remove=("${last_samples[@]: -$num_to_remove}")
    rm "${samples_to_remove[@]}"
fi

# Get a list of all generated segments
segments=("$output_dir"/*.wav)

# Calculate the number of segments to attach noise to
num_segments=${#segments[@]}
echo "Number of segments: $num_segments"
num_noise_segments=$((num_segments * noise_percentage / 100))
echo "Number of noise segments: $num_noise_segments"

# Shuffle the segments randomly
shuf_segments=($(shuf -e "${segments[@]}"))

# Attach noise to the selected segments
for ((i = 0; i < num_noise_segments; i++)); do
    segment="${shuf_segments[i]}"
    noise_file=$(ls "$noise_dir"/*.wav | shuf -n 1)
    noise_name=$(basename "$noise_file")
    # Generate a random start time within the segment duration
    start_time=$(shuf -i 0-$((segment_duration - 10)) -n 1)
    output_file="$output_dir/segment_$((i + 1))_with_$noise_name.$start_time.wav"
    mid_output_file="$output_dir/pre_segment_$((i + 1))_with_$noise_name.$start_time.wav"
    # echo "Attaching noise to $segment at $start_time seconds"
    # Trim the segment from the start time to the end
    trimmed_segment="$output_dir/trimmed_segment_$((i + 1)).wav"
    # echo "Start time: $start_time; End time: $segment_duration"
    sox "$segment" "$trimmed_segment" trim "$start_time"
    
    # Append the noise to the trimmed segment
    # echo "appending noise"
    sox "$noise_file" "$trimmed_segment" "$mid_output_file"
    
    # Concatenate the remaining part of the original segment to the output file
    remaining_segment="$output_dir/remaining_segment_$((i + 1)).wav"
    # echo "Trimming remaining segment from 0 to $start_time"
    sox "$segment" "$remaining_segment" trim 0 "$start_time"
    sox "$remaining_segment" "$mid_output_file" "$output_file"
    
    # Clean up the trimmed segment and remaining segment
    rm "$trimmed_segment" "$remaining_segment" "$mid_output_file"
done

# Iterate over all files in the output directory
for file in "$output_dir"/*.wav; do
    # Get the current filename
    filename=$(basename "$file")
    
    # Replace the word in the filename
    new_filename="${filename/segment_/background_}"
    
    # Rename the file with the new filename
    mv "$file" "$output_dir/$new_filename"
done