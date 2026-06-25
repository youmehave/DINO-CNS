import cv2
import numpy as np
import torch
from cns.utils.perception import CameraIntrinsic
from cns.benchmark.stop_policy import PixelStopPolicy
from cns.benchmark.environment import BenchmarkEnvRender
from cns.benchmark.pipeline import CorrespondenceBasedPipeline, VisOpt

def run_demo_video():
    print("[INFO] Initializing pipeline...")
    pipeline = CorrespondenceBasedPipeline(
        detector="AKAZE",
        ckpt_path="checkpoints/cns_state_dict.pth",
        intrinsic=CameraIntrinsic.default(),
        device="cuda:0",
        ransac=True,
        vis=VisOpt.NO
    )
    stop_policy = PixelStopPolicy(waiting_time=0.5, conduct_thresh=5e-3)
    
    print("[INFO] Initializing environment...")
    env = BenchmarkEnvRender(scale=1, section="A")
    env.clear_debug_items()
    
    # Pick the first scene
    tar_img = env.init(0)  
    tPo_norm = np.linalg.norm(env.target_wcT[:3, 3])

    pipeline.frontend.reset_intrinsic(env.camera.intrinsic)
    pipeline.set_target(tar_img, dist_scale=tPo_norm)
    stop_policy.reset()

    # Setup Video Writer
    h, w = tar_img.shape[:2]
    out_file = 'servo_demo.mp4'
    fps = 20
    # Create side-by-side frame: [Target Image | Current Camera View]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(out_file, fourcc, fps, (w * 2, h))

    # Add visual text on the target image
    tar_img_display = tar_img.copy()
    cv2.putText(tar_img_display, "Target (Desired)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    print("[INFO] Starting servoing loop and recording video...")
    while True:
        cur_img = env.observation()
        
        # Draw text on current image
        cur_img_display = cur_img.copy()
        cv2.putText(cur_img_display, "Current Camera View", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Combine side-by-side
        frame = np.concatenate((tar_img_display, cur_img_display), axis=1)
        video_writer.write(frame)
        
        vel, data, timing = pipeline.get_control_rate(cur_img)
        need_stop = (
            stop_policy(data, env.steps*env.dt) or 
            env.exceeds_maximum_steps() or
            env.camera_under_ground() or
            data is None
        )

        if need_stop:
            break
            
        env.action(vel * 2)  # Action step
        
        if env.steps % 20 == 0:
            print(f"[INFO] Step {env.steps}/600 recorded...")

    # Write the final settled frame for an extra 2 seconds so the user can see the final state
    for _ in range(fps * 2):
        video_writer.write(frame)
        
    video_writer.release()
    print(f"[INFO] Success! Video saved to {out_file}")

if __name__ == "__main__":
    run_demo_video()
