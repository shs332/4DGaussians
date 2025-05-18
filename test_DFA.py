import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from utils.graphics_utils import focal2fov
from scene.colmap_loader import qvec2rotmat
from scene.dataset_readers import CameraInfo
from scene.neural_3D_dataset_NDC import get_spiral
from torchvision import transforms
import json
from scipy.spatial.transform import Rotation
import torch
import os
import numpy as np

class DFA_dataset(Dataset):
    def __init__(
        self,
        cam_folder,
        split,
        frame_from: str=None,
        frame_to: str=None,
        cam_idx: str=None,
        white_background=True
    ):
        # breakpoint()
        object_name = os.path.split(cam_folder)[-1]
        self.white_background = white_background
        self.dir_from = f"/data2/wlsgur4011/GESI/SC-GS/data/DFA_processed/{object_name}/{frame_from}"
        self.dir_to = f"/data2/wlsgur4011/GESI/SC-GS/data/DFA_processed/{object_name}/{frame_to}"
        cam_idx = int(cam_idx)

        w2c_list1, file_name_list1 = load_extrinsics(self.dir_from)
        intrinsic_list1 = load_intrinsics(self.dir_from)
        
        w2c_list2, file_name_list2 = load_extrinsics(self.dir_to)
        intrinsic_list2 = load_intrinsics(self.dir_to) # fx, fy, cx, cy, all image same

        fx, fy = intrinsic_list1[0][0], intrinsic_list1[0][1]
        image_path = os.path.join(self.dir_from, "images", file_name_list1[0])
        image = Image.open(image_path)
        image_np = np.array(image)
        height, width = image_np.shape[:2]
        
        self.focal = [fy, fx] # focal[0] = fl_y, focal[1] = fl_x
        self.FovY = focal2fov(self.focal[0], height)
        self.FovX = focal2fov(self.focal[1], width)
        self.transform = transforms.ToTensor()
        self.image_paths, self.image_poses, self.image_times, self.cxs, self.cys = [], [], [], [], []
        # breakpoint()
        # 이미지 경로, 포즈, 시간 
        if split == "train":
            for w2c, file_name, intrinsic in zip(w2c_list1, file_name_list1, intrinsic_list1): # from
                R = w2c[:3, :3]
                T = w2c[:3, 3]
                image_dir = os.path.join(self.dir_from, "images", file_name)
                cx, cy = intrinsic[2], intrinsic[3]

                self.image_poses.append((R, T))
                self.image_paths.append(image_dir)
                self.cxs.append(cx)
                self.cys.append(cy)
                self.image_times.append(0.5)

            ### Deform_to
            w2c, file_name, intrinsic = w2c_list2[cam_idx], file_name_list2[cam_idx], intrinsic_list2[cam_idx]
            R = w2c[:3, :3]
            T = w2c[:3, 3]
            image_dir = os.path.join(self.dir_to, "images", file_name)
            cx, cy = intrinsic[2], intrinsic[3]
            self.image_poses.append((R, T))
            self.image_paths.append(image_dir)
            self.cxs.append(cx)
            self.cys.append(cy)
            self.image_times.append(1.0)
            
        elif split == "test":
            idx = 0
            for w2c, file_name, intrinsic in zip(w2c_list2, file_name_list2, intrinsic_list2):
                if idx != cam_idx:
                    R = w2c[:3, :3]
                    T = w2c[:3, 3]
                    image_dir = os.path.join(self.dir_to, "images", file_name)
                    cx, cy = intrinsic[2], intrinsic[3]

                    self.image_poses.append((R, T))
                    self.image_paths.append(image_dir)
                    self.cxs.append(cx)
                    self.cys.append(cy)
                    self.image_times.append(1.0)  
                idx += 1
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        img = Image.open(self.image_paths[index]) #.convert("RGB")
        im_data = np.array(img.convert("RGBA"))
        
        bg = np.array([1,1,1]) if self.white_background else np.array([0, 0, 0])
        norm_data = im_data / 255.0
        arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])

        img = torch.from_numpy(arr).permute(2, 0, 1)  # [3, H, W]
        # image_width = img.shape[2]
        # image_height = img.shape[1]
        
        return img, self.image_poses[index], self.image_times[index], self.cxs[index], \
            self.cys[index], self.focal[1], self.focal[0] ## fL_x, fl_y
    
    def load_pose(self, index):
        return self.image_poses[index]

def dfa_to_colmap(c2w):
    c2w[2, :] *= -1  # flip whole world upside down
    # change deformation
    c2w = c2w[[1, 0, 2, 3], :]
    c2w = c2w[:, [1, 2, 0, 3]]

    w2c = np.linalg.inv(c2w)
    return w2c


def _load_extrinsics(data_dir):
    extrinsics_path = os.path.join(data_dir, "Campose.inf")
    extrinsics_list = []
    with open(extrinsics_path, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip() != ""]
    for line in lines:
        parts = line.split()
        if len(parts) != 12:
            raise Exception(f"Line in CamPose.inf does not contain 12 numbers: {line}")
        nums = [float(x) for x in parts]
        mat_4x3 = np.array(nums).reshape(4, 3)
        mat_4x4 = np.zeros((4, 4))
        mat_4x4[:3, :3] = mat_4x3[:3, :3].T
        mat_4x4[:3, 3] = mat_4x3[3, :]
        mat_4x4[3, :] = np.array([0, 0, 0, 1])
        extrinsics_list.append(mat_4x4)
    
    return extrinsics_list


def load_extrinsics(data_dir):
    extrinsics_list = _load_extrinsics(data_dir)
    n_cameras = len(extrinsics_list)
    
    w2c_list = []
    file_name_list = []
    for view in range(n_cameras):
        file_name = f"img_{view:04d}_rgba.png"
        image_path = os.path.join(data_dir, "images", file_name)
        if not os.path.exists(image_path):
            print(f"Warning: {image_path} does not exist.")
            continue
        transform = extrinsics_list[view]
        w2c = dfa_to_colmap(transform)
        R = w2c[:3, :3].T
        R = R[[1, 0, 2]]
        w2c[:3, :3] = R
        w2c_list.append(w2c)
        file_name_list.append(file_name)
    
    return w2c_list, file_name_list


def load_intrinsics(data_dir):
    intrinsics_path = os.path.join(data_dir, "Intrinsic.inf")
    intrinsic_list = []
    with open(intrinsics_path, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip() != ""]

    i = 0
    while i < len(lines):
        cam_index = int(lines[i])
        row1 = [float(x) for x in lines[i + 1].split()]
        row2 = [float(x) for x in lines[i + 2].split()]
        row3 = [float(x) for x in lines[i + 3].split()]
        
        # Intrinsics matrix:
        # [ fx    0   cx ]
        # [  0   fy   cy ]
        # [  0    0    1 ]
        fx = row1[0]
        cx = row1[2]
        fy = row2[1]
        cy = row2[2]
        intrinsic_list.append((fx, fy, cx, cy))
        i += 4
    
    return intrinsic_list

if __name__ == "__main__":
    object_dir = "/data2/wlsgur4011/GESI/SC-GS/data/DFA_processed/beagle_dog(s1)"
    
    train = DFA_dataset(object_dir,
        "train",
        frame_from = '520',
        frame_to = '525',
        cam_idx = '16',
        white_background=True)
    test = DFA_dataset(object_dir,
        "test",
        frame_from = '520',
        frame_to = '525',
        cam_idx = '16',
        white_background=True)
    

    