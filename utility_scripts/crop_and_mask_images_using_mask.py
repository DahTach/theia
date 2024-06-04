import cv2, os
import numpy as np
import argparse



parser = argparse.ArgumentParser()
parser.add_argument(
    "-p", "--path", type=str, default=".", help="Path folder"
)

parser.add_argument(
    "-m", "--mask", type=str, default="mask.png", help="The image used as a mask for all the images"
)





def apply_mask_and_crop(image_path, mask_path):
  """
  Applies a mask image to another image and crops based on the white rectangle.

  Args:
      image_path: Path to the image to be masked and cropped (immageB).
      mask_path: Path to the mask image containing the white rectangle.

  Returns:
      A NumPy array representing the cropped image, or None if errors occur.
  """

  try:
    # Read images
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, 0)  # Read mask in grayscale

    # Handle potential errors (e.g., missing files)
    if image is None or mask is None:
      print("Error: Could not read image or mask file.")
      return None

    # Ensure mask is the same size as one of the image channels (grayscale or RGB)
    if mask.shape != image.shape[:2] and mask.shape != image.shape[1:3]:
      print("Error: Mask dimensions do not match image channels.")
      return None

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check if any contours are found (meaning white rectangle exists)
    if len(contours) == 0:
      print("Error: No white rectangle found in the mask.")
      return None

    # Get the largest contour (assuming the white rectangle is the largest)
    largest_contour = max(contours, key=cv2.contourArea)

    # Get bounding rectangle of the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    # Crop the image based on the bounding rectangle
    cropped_image = masked_image[y:y+h, x:x+w]

    return cropped_image

  except Exception as e:
    print(f"Error: {e} no mask image found?")
    return None

 
 



def main():
	args = parser.parse_args()
	image_folder = args.path + "/"
	mask_path = image_folder + args.mask 
	output_path = image_folder + "/cropped_images/"
	try:
		os.mkdir(output_path)
	except:
		pass
	print(F"image_folder{image_folder}")
	print(F"mask_path{mask_path}")
	#exit()
	for filename in os.listdir(image_folder):
	# Check if it's a JPG image (case-insensitive)
		print(F"filename:{filename}")
		if filename.lower().endswith(".jpg") and "mask" not in filename :
			cropped_image = apply_mask_and_crop(image_folder+filename,mask_path)
			if cropped_image is not None:
				cv2.imwrite(output_path+filename, cropped_image,[cv2.IMWRITE_JPEG_QUALITY, 85])
			else:
			  print("Cropping failed.")


if __name__ == "__main__":
    main()


#How to use:
# Modify one image with gimp creating a white rectangle (select fill with white then invert selection and fill in black) for the part of the image you want to preserve and then call it mask.png and put it in the images folder
# execute python3 crop_images_using_mask.py -p foldername -m mask.png
# the modified images will be put in the folder foldername/cropped_images/
