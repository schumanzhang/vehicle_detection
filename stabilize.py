import cv2
import numpy as np


class Stabilizer(object):

    def __init__(self, cap, reset_frequency, path):
        self.cap = cap
        self.frames = [self.cap.read()[1][:, :, 0]]
        self.transforms = [np.identity(3)]
        self.height, self.width = self.frames[0].shape
        self.reset_frequency = reset_frequency
        self.vid_path = path
        
    def get_transform_matrices(self, current):
        # goodFeaturesToTrack - Determines strong corners on an image.
        prev_corner = cv2.goodFeaturesToTrack(self.frames[-1], 200, 0.0001, 10)
        
        cur_corner, status, _ = cv2.calcOpticalFlowPyrLK(self.frames[-1], current, prev_corner, np.array([]))
        prev_corner, cur_corner = map(lambda corners: corners[status.ravel().astype(bool)], [prev_corner, cur_corner])
        transform = cv2.estimateRigidTransform(prev_corner, cur_corner, True)

        if transform is not None:
            transform = np.append(transform, [[0, 0, 1]], axis=0)

        if transform is None:
            transform = self.transforms[-1]

        self.transforms.append(transform)
        self.frames.append(current)

    # height, width = frames[0].shape
    def stabilize_images(self):
        height, width = self.frames[0].shape
        stabilized_frames = []
        last_transform = np.identity(3)

        for frame, transform, index in zip(self.frames, self.transforms, range(len(self.frames))):
            transform = transform.dot(last_transform)
            if index % self.reset_frequency == 0:
                transform = np.identity(3)
            last_transform = transform
            inverse_transform = cv2.invertAffineTransform(transform[:2])
            stabilized_frames.append(cv2.warpAffine(frame, inverse_transform, (width, height)))

        '''
        writer = cv2.VideoWriter(self.vid_path, 
                         cv2.VideoWriter_fourcc('F','M','P','4'), 
                         20.0, (width * 2, height), False)
        '''
        writer = cv2.VideoWriter(self.vid_path, 
                         cv2.VideoWriter_fourcc('F','M','P','4'), 
                         20.0, (width, height), False)
        
        print('Saving frames as video...')
        for frame, stabilized in zip(self.frames, stabilized_frames):
            # writer.write(np.concatenate([stabilized, frame], axis=1))
            writer.write(stabilized)
        writer.release()
        print('output file completed')


    
        


