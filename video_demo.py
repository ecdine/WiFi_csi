import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from train import TransformerModel
from wipose import CSI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
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
        swapped_and_inverted_keypoints = np.copy(person)
        swapped_and_inverted_keypoints[:, 1], swapped_and_inverted_keypoints[:, 2] = person[:, 2], -person[:, 1]
        ax.scatter(swapped_and_inverted_keypoints[:, 0], swapped_and_inverted_keypoints[:, 1], swapped_and_inverted_keypoints[:, 2], c='r', marker='o')
        for limb in limbs:
            start_point = swapped_and_inverted_keypoints[limb[0]]
            end_point = swapped_and_inverted_keypoints[limb[1]]
            ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], [start_point[2], end_point[2]], 'b')

    ax.set_xlim([0, 3])
    ax.set_ylim([1, 4])  
    ax.set_zlim([-5, -2.5])
    
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Z Coordinate')
    ax.set_zlabel('Y Coordinate')
    plt.draw()  
    fig.canvas.draw()  
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return image

@torch.no_grad()
def load_model_and_run_images():
    csi_path = '/megadisk/fanghengyu/3D-Pose/demodata'
    MODEL = CSI()
    results = MODEL.get_csi(csi_path)
    model = TransformerModel().to(device='cuda:1')
    model.load_state_dict(torch.load('/megadisk/fanghengyu/3D-Pose/path/train__510.pth', map_location='cuda'))
    model.eval()
    frames = []
    batch_size = 64

    def process_keypoints(keypoints):
        keypoints = keypoints.detach().cpu().numpy()
        return visualize_3d_keypoints(keypoints)

    with ThreadPoolExecutor() as executor:
        for i in tqdm(range(0, len(results), batch_size), desc='Processing tensors'):
            batch_tensors = torch.stack(results[i:i+batch_size]).to(device='cuda:1')
            outputs = model(batch_tensors)
            probas = outputs['pred_logits'].softmax(-1)[:, :, :-1]
            keep = probas.max(-1).values > 0.55
            for j, keypoints in enumerate(outputs['pred_keypoints']):
                keypoints = keypoints[keep[j]].reshape(-1, 14, 3)
                frames.append(executor.submit(process_keypoints, keypoints))

        frames = [future.result() for future in frames]
    height, width, layers = frames[0].shape
    video = cv2.VideoWriter('video/3D_keypoints_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 5, (width, height))
    for frame in frames:
        video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    video.release()

if __name__ == "__main__":
    load_model_and_run_images()