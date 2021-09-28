import cv2 as cv
from pose_estimator import PoseEstimator

class VideoImgManager():

    def __init__(self):
        self.POSE_ESTIMATOR = PoseEstimator()
        self.FIRST = True

    def estimate_vid(self,webcam_id=0):
        """reads webcam, applies pose estimation on webcam"""
        cap = cv.VideoCapture(webcam_id)

        while(True):
            has_frame, frame = cap.read()

            if not has_frame:
                break
            # frame = cv.resize(frame, (780, 540),interpolation = cv.INTER_NEAREST)
            frame = cv.resize(frame, (800, 850),interpolation = cv.INTER_NEAREST)
            # frame=cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)

            if self.FIRST:
                self.WEB_CAM_H,self.WEB_CAM_W = frame.shape[0:2]
                self.FIRST = False

            frame = self.POSE_ESTIMATOR.get_pose_key_angles(frame)
        
            cv.imshow('frame',frame)
            if cv.waitKey(2) & 0xFF == ord('q'):
                cv.destroyAllWindows()
                break
    
    def estimate_img(self,img_path):
        """applies pose estimation on img"""

        img = cv.imread(img_path)
        # img = cv.resize(img, (780, 540),interpolation = cv.INTER_NEAREST)
        img = self.POSE_ESTIMATOR.get_pose_key_angles(img)
        img= cv.resize(img, dsize=(400, 800), interpolation=cv.INTER_CUBIC)

        cv.imshow("Image Pose Estimation",img)
        
        #cv.imwrite("Images/1.jpg", img)

        cv.waitKey(0)
        cv.destroyAllWindows()



