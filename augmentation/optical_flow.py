import os
import cv2
import torch
from torchvision.transforms import ToTensor
from models import FlowNet2  # flownet2-pytorch library

def compute_optical_flow(folder_location, frame_number, save_directory):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load FlowNet2 model
    flownet = FlowNet2().to(device)
    checkpoint = torch.load('FlowNet2_checkpoint.pth.tar')  # Path to FlowNet2 checkpoint
    flownet.load_state_dict(checkpoint['state_dict'])
    flownet.eval()

    # Get frame paths
    frame_paths = sorted([os.path.join(folder_location, f) for f in os.listdir(folder_location)])

    # Load frames
    frame_curr = cv2.imread(frame_paths[frame_number])
    frame_next = cv2.imread(frame_paths[frame_number + 1])

    # Convert frames to tensors
    frame_curr_tensor = ToTensor()(frame_curr).unsqueeze(0).to(device)
    frame_next_tensor = ToTensor()(frame_next).unsqueeze(0).to(device)

    # Compute optical flow
    with torch.no_grad():
        input_var = torch.cat((frame_curr_tensor, frame_next_tensor), dim=1)
        flow = flownet(input_var)

    # Convert flow tensor to numpy array
    flow = flow.squeeze().permute(1, 2, 0).cpu().numpy()

    # Save optical flow image
    save_path = os.path.join(save_directory, f'optical_flow_{frame_number}.png')
    cv2.imwrite(save_path, flow)

# Example usage
folder_location = 'path/to/frames/folder'
save_directory = 'path/to/save/directory'
frame_number = 0
compute_optical_flow(folder_location, frame_number, save_directory)