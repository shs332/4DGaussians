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
# from scene.dataset_readers import readCamerasFromDiva360

### TODO: implement based on multipleview_dataset.py ###
class Diva360_dataset(Dataset):
    def __init__(
        self,
        cam_extrinsics,
        cam_intrinsics,
        cam_folder,
        split
    ):
        # 카메라 내부 파라미터 설정
        self.focal = [cam_intrinsics[1]["params"][1], cam_intrinsics[1]["params"][0]] # focal[0] = fl_y, focal[1] = fl_x
        height = cam_intrinsics[1]["height"]
        width = cam_intrinsics[1]["width"]
        self.FovY = focal2fov(self.focal[1], height)
        self.FovX = focal2fov(self.focal[0], width)
        self.transform = T.ToTensor()
        
        # 이미지 경로, 포즈, 시간 로드
        self.image_paths, self.image_poses, self.image_times = self.load_images_path(cam_folder, cam_extrinsics, cam_intrinsics, split)
        self.video_cam_infos = None
        # 비디오 카메라 정보 초기화
        # if split == "test":
        #     self.video_cam_infos = self.get_video_cam_infos(cam_folder)
        # else:
        #     self.video_cam_infos = None
    
    def load_images_path(self, cam_folder, cam_extrinsics, cam_intrinsics, split):
        # JSON 파일 로드
        with open(os.path.join(cam_folder, f"transforms_{split}.json"), "r") as f:
            meta = json.load(f)
        
        image_paths = []
        image_poses = []
        image_times = []
        
        # 유효한 프레임만 필터링
        valid_frames = [f for f in meta["frames"] if "file_path" in f and "transform_matrix" in f and len(f["transform_matrix"]) > 0]
        total_frames = len(valid_frames)
        
        for i, frame in enumerate(valid_frames):
            file_path = frame["file_path"]
            image_path = os.path.join(cam_folder, file_path)
            
            # transform_matrix에서 카메라 포즈 가져오기
            c2w = np.array(frame["transform_matrix"])
            
            # OpenGL에서 COLMAP 좌표계로 변환
            c2w[:3, 1:3] *= -1
            
            # world-to-camera 변환
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])
            T = w2c[:3, 3]
            
            # 카메라 시간 (0~1 사이 정규화)
            if "time" in frame:
                time = frame["time"]
            else:
                time = float(i) / total_frames
            
            # 리스트에 추가
            image_paths.append(image_path)
            image_poses.append((R, T))
            image_times.append(time)
        
        return image_paths, image_poses, image_times
    
    def get_video_cam_infos(self, datadir):
        # 카메라 경로 추출
        try:
            # poses_bounds_multipleview.npy 파일이 있으면 사용
            poses_arr = np.load(os.path.join(datadir, "poses_bounds_multipleview.npy"))
            poses = poses_arr[:, :-2].reshape([-1, 3, 5])
            near_fars = poses_arr[:, -2:]
            poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
        except FileNotFoundError:
            # 없으면 transform_test.json에서 카메라 포즈 추출
            with open(os.path.join(datadir, "transforms_test.json"), "r") as f:
                test_meta = json.load(f)
            
            # 유효한 프레임으로부터 카메라 포즈 추출
            poses = []
            for frame in test_meta["frames"]:
                if "transform_matrix" in frame and len(frame["transform_matrix"]) > 0:
                    poses.append(np.array(frame["transform_matrix"]))
            
            if not poses:
                # 유효한 포즈가 없으면 빈 배열 반환
                return []
        
        # 나선형 카메라 경로 생성
        N_views = 120  # 비디오용 뷰 개수
        val_poses = get_spiral(poses, np.array([[0.1, 100.0]]), N_views=N_views)
        
        # 카메라 생성
        cameras = []
        len_poses = len(val_poses)
        times = [i/len_poses for i in range(len_poses)]
        
        # 이미지 하나만 로드하여 크기 정보 얻기
        image = Image.open(self.image_paths[0])
        image = self.transform(image)
        
        # 각 포즈마다 카메라 정보 생성
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
            
            cameras.append(CameraInfo(
                uid=idx, 
                R=R, 
                T=T, 
                FovY=FovY, 
                FovX=FovX, 
                image=image,
                image_path=image_path, 
                image_name=image_name, 
                width=image.shape[2], 
                height=image.shape[1],
                time=time, 
                mask=None
            ))
        
        return cameras
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        img = Image.open(self.image_paths[index])
        img = self.transform(img)
        return img, self.image_poses[index], self.image_times[index]
    
    def load_pose(self, index):
        return self.image_poses[index]