from vid_img_manager import VideoImgManager
import os
import numpy as np

class Main():

    def __init__(self):
        self.VI_M = VideoImgManager()

    def img_estimation(self, img_path):
        self.VI_M.estimate_img(img_path)

    def live_estimation(self,webcam_id=0):
        self.VI_M.estimate_vid(webcam_id)

if __name__ == "__main__":
    app = Main()
    frame= app.img_estimation(r"1.jpeg")
    # frame= cv.resize(frame, dsize=(500, 300), interpolation=cv.INTER_CUBIC)
    # app.live_estimation("videos/3.mp4")
    