import torch
from train import TransformerModel 
from wipose import CSI
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from concurrent.futures import ThreadPoolExecutor

def visualize_3d_keypoints(keypoints, filename):
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
    
    plt.savefig(filename)  
    plt.close(fig) 

@torch.no_grad()
def load_model_and_run_images():
    csi_path = '/megadisk/fanghengyu/3D-Pose/demodata'
    MODEL = CSI()
    results = MODEL.get_csi(csi_path)
    model = TransformerModel().to(device='cuda:1')
    model.load_state_dict(torch.load('/megadisk/fanghengyu/3D-Pose/path/train__510.pth', map_location='cuda'))
    model.eval()
    i = 0
    frames = []
    
    with ThreadPoolExecutor() as executor:
        for tensor in results:
            assert isinstance(tensor, torch.Tensor), "The element is not a PyTorch tensor."
            tensor = tensor.unsqueeze(0)
            tensor = tensor.to(device='cuda:1')
            outputs = model(tensor)
            probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
            keep = probas.max(-1).values > 0.5
            keypoints = outputs['pred_keypoints'][0, keep]
            keypoints = keypoints.reshape(-1, 14, 3)
            keypoints = keypoints.detach().cpu().numpy()
            future = executor.submit(visualize_3d_keypoints, keypoints, f'img/3D_keypoints_pth150_{i+1}.png')
            frames.append(future)
            i += 1
    
    # Wait for all tasks to complete
    for frame in frames:
        frame.result()

if __name__ == "__main__":
    load_model_and_run_images()
