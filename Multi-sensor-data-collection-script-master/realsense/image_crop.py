import os
from PIL import Image

images_folder = r'F:/realsense/test_2_cameras/run2/2/camera_902512070040/RGB'
save_path = r'F:/realsense/test_2_cameras/run2/2/camera_902512070040/RGB_cropped_for_demo'
os.makedirs(save_path, exist_ok=True)

images_path = [(os.path.join(images_folder, img), os.path.join(save_path, img))  for img in os.listdir(images_folder)]

for each_img, save_path_ in images_path:
    with Image.open(each_img) as img:
        cropped_img = img.crop((0, 180, 640, 480))
        cropped_img.save(save_path_)
    
    print(f'cropped and saved: {save_path_}')