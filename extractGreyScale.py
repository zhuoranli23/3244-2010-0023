import os 
import cv2

root_path = "replace/your/root/path/here"

files = os.listdir(root_path + 'trainB/');
for file in files:
  if file.endswith('.jpg'):	
    image = cv2.imread(root_path + 'trainB/'+ file);
    # this converts to grey scale
    greyScale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # this converts to shaded images, but it doesn't work well so ignore. 
    #dst_gray, dst_color = cv2.pencilSketch(image, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
    
    cv2.imwrite(root_path + "grey/" + file[:-5] + "A.jpg", greyScale);