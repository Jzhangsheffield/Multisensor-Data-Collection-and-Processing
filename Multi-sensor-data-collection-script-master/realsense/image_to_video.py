import cv2 as cv
import os

def image_to_video():
    #_path = 'Bothhand_pinch_embed_81_86'
    file_path = r'F:/realsense/test_2_cameras/preprocessing_with_emg/4/camera_141722071078/RGB' # 图片目录
    output = 'F:/realsense/test_2_cameras/preprocessing_with_emg/4/camera_141722071078/RGB/sampler_preparation.mp4'  # 生成视频路径
    img_list = sorted(os.listdir(file_path))[200:2085]
    img_path = [os.path.join(file_path, img) for img in img_list]# 生成图片目录下以图片名字为内容的列表
    # print(img_path)
    height = 480
    weight = 640
    fps = 30
    # fourcc = cv.VideoWriter_fourcc('M', 'J', 'P', 'G') 用于avi格式的生成
    fourcc = cv.VideoWriter_fourcc(*'mp4v')  # 用于mp4格式的生成
    videowriter = cv.VideoWriter(output, fourcc, fps, (weight, height), True)  # 创建一个写入视频对象
    for img in img_path:
        # path = file_path + img
        # print(path)
        frame = cv.imread(img)
        videowriter.write(frame)

    videowriter.release()
    
    
image_to_video()
    
    
    
# image_path = r'F:/realsense/test_2_cameras/run2'
# fourcc = cv.VideoWriter_fourcc(*'mp4v')

# for num_expmt in os.listdir(image_path):
    
#     if num_expmt == '0':
#         sample = 'sample3_1'
#     elif num_expmt == '1':
#         sample = 'sample3_2'
#     elif num_expmt == '2':
#         sample = 'sample4_1'
#     else:
#         sample = 'sample4_2'
        
#     for camera in os.listdir(os.path.join(image_path, num_expmt)):
#         if camera == 'camera_141722071078':
#             camera_position = 'back'
#         else:
#             camera_position = 'top'
#         for modality in os.listdir(os.path.join(image_path, num_expmt, camera)):
#             image_list = [os.path.join(image_path, num_expmt, camera, modality, each_img) for each_img in os.listdir(os.path.join(image_path, num_expmt, camera, modality))]
#             image_list.sort()
#             # print(image_list)
            
#             for fps in [15, 30]:
#                 file_name = f'{sample}_{camera_position}_{fps}.mp4'
#                 save_dir = os.path.join(image_path, num_expmt, camera, modality, file_name)
#                 videowriter = cv.VideoWriter(save_dir, fourcc, fps, (640, 480), True)
                
#                 for img in image_list:
#                     frame = cv.imread(img)
#                     videowriter.write(frame)
                    
#                 videowriter.release()



