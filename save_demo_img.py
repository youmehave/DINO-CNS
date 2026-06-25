import cv2
from cns.benchmark.environment import BenchmarkEnvRender

def main():
    # Initialize the simulation environment
    env = BenchmarkEnvRender(scale=1, section="A")
    
    # 1. Get the target image for the first scene
    tar_img = env.init(0) 
    
    # 2. Get the initial observation (where the camera starts)
    cur_img = env.observation() 
    
    # Save the images
    cv2.imwrite("desired_image.png", tar_img)
    cv2.imwrite("initial_image.png", cur_img)
    print("Images saved successfully.")

if __name__ == "__main__":
    main()
