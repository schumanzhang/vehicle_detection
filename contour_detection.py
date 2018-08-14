from pipeline_processor import PipelineProcessor

import utils
import cv2

class ContourDetection(PipelineProcessor):
    '''
        Detecting moving objects.
        Purpose of this processor is to subtract background, get moving objects
        and detect them with a cv2.findContours method, and then filter off-by
        width and height. 
        bg_subtractor - background subtractor isinstance.
        min_contour_width - min bounding rectangle width.
        min_contour_height - min bounding rectangle height.
        save_image - if True will save detected objects mask to file.
        image_dir - where to save images(must exist).        
    '''

    def __init__(self, bg_subtractor, min_contour_width=35, min_contour_height=35, save_image=False, image_dir="images"):
        super(ContourDetection, self).__init__()

        self.bg_subtractor = bg_subtractor
        self.min_contour_width = min_contour_width
        self.min_contour_height = min_contour_height
        self.save_image = save_image
        self.image_dir = image_dir


    def filter_mask(self, img, a=None):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        # Fill any small holes
        closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        # Remove noise
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
        # Dilate to merge adjacent blobs
        dilation = cv2.dilate(opening, kernel, iterations=2)
        return dilation


    def detect_vehicles(self, fg_mask, context):
        matches = []
        # finding external contours
        im2, contours, hierachy = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1
        )

        for (i, contour) in enumerate(contours):
            (x, y, w, h) = cv2.boundingRect(contour)

            contour_valid = (w >= self.min_contour_width) and (
                h >= self.min_contour_height)

            if not contour_valid:
                continue

            centroid = utils.get_centroid(x, y, w, h)
            matches.append(((x, y, w, h), centroid))

        self.log.debug('#MATCHES FOUND: %s' % len(matches))
        return matches


    def __call__(self, context):
        frame = context['frame'].copy()
        frame_number = context['frame_number']

        fg_mask = self.bg_subtractor.apply(frame, None, 0.001)
        fg_mask[fg_mask < 240] = 0
        fg_mask = self.filter_mask(fg_mask, frame_number)

        if self.save_image:
            utils.save_frame(fg_mask, self.image_dir +
                             "/mask_%04d.png" % frame_number, flip=False)

        context['objects'] = self.detect_vehicles(fg_mask, context)
        context['fg_mask'] = fg_mask

        return context


