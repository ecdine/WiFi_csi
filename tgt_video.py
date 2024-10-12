import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from concurrent.futures import ThreadPoolExecutor
import cv2

def visualize_3d_keypoints(keypoints):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    limbs = [
        (1, 0), (8, 4), (4, 2), (2, 0),
        (10, 6), (6, 3), (3, 0),
        (2, 5), (5, 9), (9, 12),
        (3, 7), (7, 11), (11, 13), (5, 7)
    ]
    for person in keypoints:
        adjusted_keypoints = np.array(person)
        
        ax.scatter(adjusted_keypoints[:, 0], adjusted_keypoints[:, 1], adjusted_keypoints[:, 2], c='r', marker='o')
        
        for limb in limbs:
            start_point = adjusted_keypoints[limb[0]]
            end_point = adjusted_keypoints[limb[1]]
            ax.plot([start_point[0], end_point[0]], 
                    [start_point[1], end_point[1]], 
                    [start_point[2], end_point[2]], 'b')
    
    # Set appropriate limits and labels
    ax.set_xlim([0, 3])
    ax.set_ylim([1, 4])  
    ax.set_zlim([5, 2.5])   
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')  
    ax.set_zlabel('Z Coordinate')  

    plt.draw()
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    
    return image


def load_data_from_npys(npy_directory):
    npy_files = sorted([f for f in os.listdir(npy_directory) if f.endswith('.npy')])
    print(npy_files)
    frames = []

    def process_file(file):
        file_path = os.path.join(npy_directory, file)
        keypoints = np.load(file_path)
        frame = visualize_3d_keypoints(keypoints)
        return frame

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_file, file) for file in npy_files]
        
        for future in futures:
            frame = future.result()
            frames.append(frame)

    height, width, layers = frames[0].shape
    video = cv2.VideoWriter('video/3D_keypoints_video10.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 5, (width, height))
    for frame in frames:
        video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  
    video.release()

if __name__ == "__main__":
    npy_directory = '/megadisk/fanghengyu/3D-Pose/npydata'
    load_data_from_npys(npy_directory)
