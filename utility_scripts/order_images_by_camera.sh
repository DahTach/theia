#!/bin/bash
# This script read each picture meta info and move each picture based on wich camera made it
# Source folder containing JPEG files
source_folder="~/Scaricati/Foto_Pallet"

# Destination folder where files will be moved based on serial number
destination_folder="~/Scaricati/sorted"

# Iterate through JPEG files in the source folder
for file in "$source_folder"/*.jpg; do
    # Check if the file exists and is a regular file
    if [ -f "$file" ]; then
        # Extract serial number using exiftool
        serial_number=$(exiftool -SerialNumber "$file" | awk '{print $NF}')
        
        # Check if serial number is not empty
        if [ -n "$serial_number" ]; then
            # Create destination folder if it doesn't exist
            mkdir -p "$destination_folder/$serial_number"
            
            # Move the file to the destination folder
            mv "$file" "$destination_folder/$serial_number"
            
            echo "Moved $file to $destination_folder/$serial_number"
        else
            echo "Failed to extract serial number for $file"
        fi
    fi
done
