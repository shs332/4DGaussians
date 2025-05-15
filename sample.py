
if __name__ == "__main__":
    from scene.Diva360 import Diva360_dataset
    from scene.dataset import FourDGSdataset
    train_cam_infos = Diva360_dataset(cam_folder='/root/wlsgur4011/GESI/4DGaussians/data/Diva360/penguin', 
                                      split="train", frame_from='0217', frame_to='0239', cam_idx='00', white_background=True)    
    
    train_cameras = FourDGSdataset(train_cam_infos, None, "Diva360")

    for i in range(10):
        data = train_cameras[i]
        breakpoint()