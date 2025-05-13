import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from utils.graphics_utils import focal2fov
from scene.colmap_loader import qvec2rotmat
from scene.dataset_readers import CameraInfo
from scene.neural_3D_dataset_NDC import get_spiral
from torchvision import transforms as T
import json
from scipy.spatial.transform import Rotation
import torch
def diva360_to_colmap(c2w):
    c2w[2, :] *= -1  # flip whole world upside down
    c2w = c2w[[1, 0, 2, 3], :]
    c2w[0:3, 1] *= -1  # flip the y and z axis
    c2w[0:3, 2] *= -1

    w2c = np.linalg.inv(c2w)
    rot = w2c[:3, :3]
    tvec = w2c[:3, -1]

    rotation = Rotation.from_matrix(rot)
    qvec = rotation.as_quat()  # Returns [x, y, z, w]
    qvec = np.array(qvec)[[3, 0, 1, 2]]

    return qvec, tvec

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
        2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
        2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
        1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
        2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
        2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
        1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

### TODO: implement based on multipleview_dataset.py ###
class Diva360_dataset(Dataset):
    def __init__(
        self,
        cam_folder,
        split,
        frame_from=None,
        frame_to=None,
        cam_idx=None,
        white_background=True
    ):
        self.white_background = white_background
        with open(os.path.join(cam_folder, f"transforms_merged.json"), "r") as f:
            meta = json.load(f)
        frames = meta["frames"] # transforms_train or transforms_test

        # 카메라 내부 파라미터 설정
        self.focal = [frames[0]["fl_y"], frames[0]["fl_x"]] # focal[0] = fl_y, focal[1] = fl_x
        height = frames[0]["h"]
        width = frames[0]["w"]
        self.FovY = focal2fov(self.focal[0], height)
        self.FovX = focal2fov(self.focal[1], width)
        # breakpoint()
        
        self.transform = T.ToTensor()
        
        # 이미지 경로, 포즈, 시간 로드
        self.image_paths, self.image_poses, self.image_times, self.cxs, self.cys = self.load_images_path(cam_folder, None, None, split,
                                                                                                         frame_from=frame_from, frame_to=frame_to, cam_idx=cam_idx)

        # 비디오 카메라 정보 초기화
        if split == "test":
            self.video_cam_infos = self.get_video_cam_infos(cam_folder)
    
    def load_images_path(self, cam_folder, cam_extrinsics, cam_intrinsics, split, frame_from, frame_to, cam_idx=None):
        # JSON 파일 로드
        from scipy.spatial.transform import Rotation as Rot
        
        with open(os.path.join(cam_folder, f"transforms_merged.json"), "r") as f:
            meta = json.load(f)
        
        image_paths = []
        image_poses = []
        image_times = []
        cx_px = []
        cy_px = []

        # image_length = len(os.listdir(os.path.join(cam_folder, "cam00"))) # change cam__ if dir does not exist
        image_length = 1

        if cam_idx is None: # original logic
            for i, frame in enumerate(meta["frames"]): ## every element in "frames"            
                # transform_matrix에서 카메라 포즈 가져오기, Blender/OpenGL c2w
                c2w = np.array(frame["transform_matrix"])
                cx = frame["cx"]
                cy = frame["cy"]

                # OpenGL에서 COLMAP 좌표계로 변환
                c2w[:3, 1:3] *= -1
                
                # world-to-camera 변환
                w2c = np.linalg.inv(c2w)
                R = np.transpose(w2c[:3, :3])
                T = w2c[:3, 3]
                # breakpoint()

                file_path = frame["file_path"]
                cam_num = os.path.dirname(file_path)
                images_folder = os.path.join(cam_folder, cam_num) # .../cam01

                for frame_idx in range(image_length):
                    # 리스트에 추가
                    image_path = os.path.join(images_folder, f"frame_{str(frame_idx).zfill(5)}.png") # start from frame_00000.png 
                    image_paths.append(image_path)
                    image_poses.append((R, T))
                    image_times.append(float((frame_idx+1)/image_length))
                    cx_px.append(cx)
                    cy_px.append(cy)
                # breakpoint()
            # breakpoint()
            return image_paths, image_poses, image_times, cx_px, cy_px
        elif split == "train": # Gaussian recon stage, all image of frame_from + one image of frame_to in cam_idx

            last_element = {}
            cam_idx = f"cam{cam_idx}" # cam01
            for i, frame in enumerate(meta["frames"]): ## every element in "frames"            
                c2w = np.array(frame["transform_matrix"])
                cx = frame["cx"]
                cy = frame["cy"]

                # OpenGL에서 COLMAP 좌표계로 변환
                c2w[:3, 1:3] *= -1
                
                # world-to-camera 변환
                w2c = np.linalg.inv(c2w)
                R = np.transpose(w2c[:3, :3])
                T = w2c[:3, 3]
                # breakpoint()

                file_path = frame["file_path"]
                cam_num = os.path.dirname(file_path)
                # breakpoint()
                if cam_num == cam_idx:
                    # breakpoint()
                    last_element['image_path'] = os.path.join(cam_folder, cam_num, f"frame_{str(frame_to).zfill(5)}.png")
                    last_element['pose'] = (R, T)
                    last_element['cx'] = cx
                    last_element['cy'] = cy
                    
                images_folder = os.path.join(cam_folder, cam_num) # .../cam01
                image_path = os.path.join(images_folder, f"frame_{str(frame_from).zfill(5)}.png")
                image_paths.append(image_path)
                image_poses.append((R, T))
                image_times.append(0.5)
                cx_px.append(cx)
                cy_px.append(cy)

            image_paths.append(last_element['image_path'])
            image_poses.append(last_element['pose'])
            image_times.append(1)
            cx_px.append(last_element['cx'])
            cy_px.append(last_element['cy'])

            # breakpoint()
            return image_paths, image_poses, image_times, cx_px, cy_px
        
        elif split == "test": # deform stage, N-1 image of frame_to not in cam_idx
            for i, frame in enumerate(meta["frames"]):           
                c2w = np.array(frame["transform_matrix"])
                cx = frame["cx"]
                cy = frame["cy"]

                # OpenGL에서 COLMAP 좌표계로 변환
                c2w[:3, 1:3] *= -1
                
                # world-to-camera 변환
                w2c = np.linalg.inv(c2w)
                R = np.transpose(w2c[:3, :3])
                T = w2c[:3, 3]
                # breakpoint()

                file_path = frame["file_path"]
                cam_num = os.path.dirname(file_path)
                
                if cam_num != cam_idx: # N-1 images
                    images_folder = os.path.join(cam_folder, cam_num) # .../cam01

                    image_path = os.path.join(images_folder, f"frame_{str(frame_to).zfill(5)}.png") # start from frame_00000.png 
                    image_paths.append(image_path)
                    image_poses.append((R, T))
                    image_times.append(0.5)
                    cx_px.append(cx)
                    cy_px.append(cy)
                
            # breakpoint()
            return image_paths, image_poses, image_times, cx_px, cy_px
    
    def get_video_cam_infos(self, datadir):
        with open(os.path.join(datadir, "transforms_test.json"), "r") as f:
            test_meta = json.load(f)
        
        # 유효한 프레임으로부터 카메라 포즈 추출
        poses = []
        for frame in test_meta["frames"]:
            if "transform_matrix" in frame and len(frame["transform_matrix"]) > 0:
                poses.append(np.array(frame["transform_matrix"])[:3,:]) # 4x4 transform matrix에서 3x4로 변환

        poses = np.array(poses)
            
        # breakpoint()
        
        # 나선형 카메라 경로 생성
        N_views = 300  # 비디오용 뷰 개수
        val_poses = get_spiral(poses, np.array([[0.1, 100.0]]), N_views=N_views)
        
        # 카메라 생성
        cameras = []
        len_poses = len(val_poses)
        times = [i/len_poses for i in range(len_poses)]
        
        # 이미지 하나만 로드하여 크기 정보 얻기
        image = Image.open(self.image_paths[0])
        image = self.transform(image)
        
        for idx, p in enumerate(val_poses):
            image_path = None
            image_name = f"{idx}"
            time = times[idx]
            pose = np.eye(4)
            pose[:3,:] = p[:3,:]
            R = pose[:3,:3]
            R = - R
            R[:,0] = -R[:,0]
            T = -pose[:3,3].dot(R)
            FovX = self.FovX
            FovY = self.FovY
            cameras.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                image_path=image_path, image_name=image_name, width=image.shape[2], height=image.shape[1],
                                time = time, mask=None))
        return cameras
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        img = Image.open(self.image_paths[index]) #.convert("RGB")
        im_data = np.array(img.convert("RGBA"))
        
        bg = np.array([1,1,1]) if self.white_background else np.array([0, 0, 0])
        norm_data = im_data / 255.0
        arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])

        img = torch.from_numpy(arr).permute(2, 0, 1)  # [3, H, W]
        image_width = img.shape[2]
        image_height = img.shape[1]
        
        return img, self.image_poses[index], self.image_times[index], self.cxs[index], self.cys[index], image_width, image_height
    
    def load_pose(self, index):
        return self.image_poses[index]