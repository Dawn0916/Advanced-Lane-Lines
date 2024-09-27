import cv2
import numpy as np
import matplotlib.image as mpimg

def hist(img):
    bottom_half = img[img.shape[0]//2:,:]
    return np.sum(bottom_half, axis=0)

class LaneLines:
    """ Class containing information about detected lane lines.

    Attributes:
        left_fit (np.array): Coefficients of a polynomial that fit left lane line
        right_fit (np.array): Coefficients of a polynomial that fit right lane line
        parameters (dict): Dictionary containing all parameters needed for the pipeline
        debug (boolean): Flag for debug/normal mode
    """
    def __init__(self):
        """Init Lanelines.

        Parameters:
            left_fit (np.array): Coefficients of polynomial that fit left lane
            right_fit (np.array): Coefficients of polynomial that fit right lane
            binary (np.array): binary image
        """
        self.left_fit = None
        self.right_fit = None
        self.binary = None
        self.nonzero = None
        self.nonzerox = None
        self.nonzeroy = None
        self.clear_visibility = True
        self.dir = []
        self.left_curve_img = mpimg.imread('left_turn.png')
        self.right_curve_img = mpimg.imread('right_turn.png')
        self.keep_straight_img = mpimg.imread('straight.png')
        self.left_curve_img = cv2.normalize(src=self.left_curve_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        self.right_curve_img = cv2.normalize(src=self.right_curve_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        self.keep_straight_img = cv2.normalize(src=self.keep_straight_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # HYPERPARAMETERS
        # Number of sliding windows
        self.nwindows = 9
        # Width of the the windows +/- margin
        self.margin = 100
        # Mininum number of pixels found to recenter window
        self.minpix = 50

    def forward(self, img):
        """Take a image and detect lane lines.

        Parameters:
            img (np.array): An binary image containing relevant pixels

        Returns:
            Image (np.array): An RGB image containing lane lines pixels and other details
        """
        self.extract_features(img)
        return self.fit_poly(img)

    def pixels_in_window(self, center, margin, height):
        """ Return all pixel that in a specific window

        Parameters:
            center (tuple): coordinate of the center of the window
            margin (int): half width of the window
            height (int): height of the window

        Returns:
            pixelx (np.array): x coordinates of pixels that lie inside the window
            pixely (np.array): y coordinates of pixels that lie inside the window
        """
        topleft = (center[0]-margin, center[1]-height//2)
        bottomright = (center[0]+margin, center[1]+height//2)

        condx = (topleft[0] <= self.nonzerox) & (self.nonzerox <= bottomright[0])
        condy = (topleft[1] <= self.nonzeroy) & (self.nonzeroy <= bottomright[1])
        return self.nonzerox[condx&condy], self.nonzeroy[condx&condy]

    def extract_features(self, img):
        """ Extract features from a binary image

        Parameters:
            img (np.array): A binary image
        """
        self.img = img
        # Height of of windows - based on nwindows and image shape
        self.window_height = np.int32(img.shape[0]//self.nwindows)
        #self.window_height = int(img.shape[0] // self.nwindows)


        # Identify the x and y positions of all nonzero pixel in the image
        self.nonzero = img.nonzero()
        self.nonzerox = np.array(self.nonzero[1])
        self.nonzeroy = np.array(self.nonzero[0])

    def find_lane_pixels(self, img):
        """Find lane pixels from a binary warped image.

        Parameters:
            img (np.array): A binary warped image

        Returns:
            leftx (np.array): x coordinates of left lane pixels
            lefty (np.array): y coordinates of left lane pixels
            rightx (np.array): x coordinates of right lane pixels
            righty (np.array): y coordinates of right lane pixels
            out_img (np.array): A RGB image that use to display result later on.
        """
        assert(len(img.shape) == 2)

        # Create an output image to draw on and visualize the result
        out_img = np.dstack((img, img, img))

        histogram = hist(img)
        midpoint = histogram.shape[0]//2
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Current position to be update later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base
        y_current = img.shape[0] + self.window_height//2

        # Create empty lists to reveice left and right lane pixel
        leftx, lefty, rightx, righty = [], [], [], []

        # Step through the windows one by one
        for _ in range(self.nwindows):
            y_current -= self.window_height
            center_left = (leftx_current, y_current)
            center_right = (rightx_current, y_current)

            good_left_x, good_left_y = self.pixels_in_window(center_left, self.margin, self.window_height)
            good_right_x, good_right_y = self.pixels_in_window(center_right, self.margin, self.window_height)

            # Append these indices to the lists
            leftx.extend(good_left_x)
            lefty.extend(good_left_y)
            rightx.extend(good_right_x)
            righty.extend(good_right_y)

            if len(good_left_x) > self.minpix:
                leftx_current = np.int32(np.mean(good_left_x))
            if len(good_right_x) > self.minpix:
                rightx_current = np.int32(np.mean(good_right_x))

        return leftx, lefty, rightx, righty, out_img
    

    def fit_poly(self, img):
        """Find the lane line from an image and draw it."""
        
        leftx, lefty, rightx, righty, out_img = self.find_lane_pixels(img)

        # Fit polynomial if we have enough points
        if len(lefty) > 1500:
            self.left_fit = np.polyfit(lefty, leftx, 2)
        else:
            self.left_fit = None  # No fit found

        if len(righty) > 1500:
            self.right_fit = np.polyfit(righty, rightx, 2)
        else:
            self.right_fit = None  # No fit found

        # Generate y-values for plotting
        ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])

        # Initialize left_fitx and right_fitx as None
        left_fitx, right_fitx = None, None

        # Compute left and right fit x-values if fit was successful
        if self.left_fit is not None:
            left_fitx = self.left_fit[0] * ploty**2 + self.left_fit[1] * ploty + self.left_fit[2]

        if self.right_fit is not None:
            right_fitx = self.right_fit[0] * ploty**2 + self.right_fit[1] * ploty + self.right_fit[2]

        # Visualization
        if left_fitx is not None and right_fitx is not None:
            for i, y in enumerate(ploty):
                l = int(left_fitx[i])
                r = int(right_fitx[i])
                y = int(y)
                cv2.line(out_img, (l, y), (r, y), (0, 255, 0), 5)

        return out_img



    # def fit_poly(self, img):
    #     """Find the lane line from an image and draw it.

    #     Parameters:
    #         img (np.array): a binary warped image

    #     Returns:
    #         out_img (np.array): a RGB image that have lane line drawn on that.
    #     """

    #     leftx, lefty, rightx, righty, out_img = self.find_lane_pixels(img)

    #     if len(lefty) > 1500:
    #         self.left_fit = np.polyfit(lefty, leftx, 2)
    #     if len(righty) > 1500:
    #         self.right_fit = np.polyfit(righty, rightx, 2)

    #     # Generate x and y values for plotting
    #     maxy = img.shape[0] - 1
    #     miny = img.shape[0] // 3
    #     if len(lefty):
    #         maxy = max(maxy, np.max(lefty))
    #         miny = min(miny, np.min(lefty))

    #     if len(righty):
    #         maxy = max(maxy, np.max(righty))
    #         miny = min(miny, np.min(righty))

    #     ploty = np.linspace(miny, maxy, img.shape[0])

    #     left_fitx = self.left_fit[0]*ploty**2 + self.left_fit[1]*ploty + self.left_fit[2]
    #     right_fitx = self.right_fit[0]*ploty**2 + self.right_fit[1]*ploty + self.right_fit[2]

    #     # Visualization
    #     for i, y in enumerate(ploty):
    #         l = int(left_fitx[i])
    #         r = int(right_fitx[i])
    #         y = int(y)
    #         cv2.line(out_img, (l, y), (r, y), (0, 255, 0))

    #     lR, rR, pos = self.measure_curvature()

    #     return out_img

    # def plot(self, out_img):
    #     np.set_printoptions(precision=6, suppress=True)
    #     lR, rR, pos = self.measure_curvature()

    #     value = None
    #     if abs(self.left_fit[0]) > abs(self.right_fit[0]):
    #         value = self.left_fit[0]
    #     else:
    #         value = self.right_fit[0]

    #     if abs(value) <= 0.00015:
    #         self.dir.append('F')
    #     elif value < 0:
    #         self.dir.append('L')
    #     else:
    #         self.dir.append('R')
        
    #     if len(self.dir) > 10:
    #         self.dir.pop(0)

    #     W = 400
    #     H = 500
    #     widget = np.copy(out_img[:H, :W])
    #     widget //= 2
    #     widget[0,:] = [0, 0, 255]
    #     widget[-1,:] = [0, 0, 255]
    #     widget[:,0] = [0, 0, 255]
    #     widget[:,-1] = [0, 0, 255]
    #     out_img[:H, :W] = widget

    #     direction = max(set(self.dir), key = self.dir.count)
    #     msg = "Keep Straight Ahead"
    #     curvature_msg = "Curvature = {:.0f} m".format(min(lR, rR))
    #     if direction == 'L':
    #         y, x = self.left_curve_img[:,:,3].nonzero()
    #         out_img[y, x-100+W//2] = self.left_curve_img[y, x, :3]
    #         msg = "Left Curve Ahead"
    #     if direction == 'R':
    #         y, x = self.right_curve_img[:,:,3].nonzero()
    #         out_img[y, x-100+W//2] = self.right_curve_img[y, x, :3]
    #         msg = "Right Curve Ahead"
    #     if direction == 'F':
    #         y, x = self.keep_straight_img[:,:,3].nonzero()
    #         out_img[y, x-100+W//2] = self.keep_straight_img[y, x, :3]

    #     cv2.putText(out_img, msg, org=(10, 240), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
    #     if direction in 'LR':
    #         cv2.putText(out_img, curvature_msg, org=(10, 280), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

    #     cv2.putText(
    #         out_img,
    #         "Good Lane Keeping",
    #         org=(10, 400),
    #         fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    #         fontScale=1.2,
    #         color=(0, 255, 0),
    #         thickness=2)

    #     cv2.putText(
    #         out_img,
    #         "Vehicle is {:.2f} m away from center".format(pos),
    #         org=(10, 450),
    #         fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    #         fontScale=0.66,
    #         color=(255, 255, 255),
    #         thickness=2)

    #     return out_img

    def plot(self, out_img):
        np.set_printoptions(precision=6, suppress=True)
        lR, rR, pos = self.measure_curvature()

        # Check if self.left_fit and self.right_fit are not None before accessing them
        if self.left_fit is not None and self.right_fit is not None:
            if abs(self.left_fit[0]) > abs(self.right_fit[0]):
                value = self.left_fit[0]
            else:
                value = self.right_fit[0]
        else:
            value = 0  # Default value if fits are not available

        # Determine the direction based on the value
        if abs(value) <= 0.00015:
            self.dir.append('F')
        elif value < 0:
            self.dir.append('L')
        else:
            self.dir.append('R')
        
        # Ensure the direction list doesn't grow indefinitely
        if len(self.dir) > 10:
            self.dir.pop(0)

        # Initialize widget dimensions
        W = 400
        H = 500
        widget = np.copy(out_img[:H, :W])
        widget //= 2
        widget[0,:] = [0, 0, 255]  # Red border
        widget[-1,:] = [0, 0, 255]
        widget[:,0] = [0, 0, 255]
        widget[:,-1] = [0, 0, 255]
        out_img[:H, :W] = widget

        # Determine the direction (most frequent in the last few frames)
        direction = max(set(self.dir), key=self.dir.count)
        msg = "Keep Straight Ahead"
        curvature_msg = "Curvature = {:.0f} m".format(min(lR, rR))

        # Add direction-specific images and messages
        if direction == 'L':
            y, x = self.left_curve_img[:,:,3].nonzero()
            out_img[y, x-100+W//2] = self.left_curve_img[y, x, :3]
            msg = "Left Curve Ahead"
        if direction == 'R':
            y, x = self.right_curve_img[:,:,3].nonzero()
            out_img[y, x-100+W//2] = self.right_curve_img[y, x, :3]
            msg = "Right Curve Ahead"
        if direction == 'F':
            y, x = self.keep_straight_img[:,:,3].nonzero()
            out_img[y, x-100+W//2] = self.keep_straight_img[y, x, :3]

        # Add text messages to the image
        cv2.putText(out_img, msg, org=(10, 240), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
        if direction in 'LR':
            cv2.putText(out_img, curvature_msg, org=(10, 280), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

        # Display good lane keeping message
        cv2.putText(
            out_img,
            "Good Lane Keeping",
            org=(10, 400),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1.2,
            color=(0, 255, 0),
            thickness=2)

        # Display vehicle position message
        cv2.putText(
            out_img,
            "Vehicle is {:.2f} m away from center".format(pos),
            org=(10, 450),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.66,
            color=(255, 255, 255),
            thickness=2)

        return out_img


    # def measure_curvature(self):
    #     ym = 30/720
    #     xm = 3.7/700

    #     left_fit = self.left_fit.copy()
    #     right_fit = self.right_fit.copy()
    #     y_eval = 700 * ym

    #     # Compute R_curve (radius of curvature)
    #     left_curveR =  ((1 + (2*left_fit[0] *y_eval + left_fit[1])**2)**1.5)  / np.absolute(2*left_fit[0])
    #     right_curveR = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])

    #     xl = np.dot(self.left_fit, [700**2, 700, 1])
    #     xr = np.dot(self.right_fit, [700**2, 700, 1])
    #     pos = (1280//2 - (xl+xr)//2)*xm
    #     return left_curveR, right_curveR, pos 
    

    def measure_curvature(self):
        """Calculates the curvature of the lane and vehicle position."""

        # Return 0 curvature and position if no fit is available
        if self.left_fit is None or self.right_fit is None:
            return 0, 0, 0

        # Calculate curvature assuming valid fits
        left_fit = self.left_fit.copy()
        right_fit = self.right_fit.copy()

        ym = 30/720
        xm = 3.7/700
        y_eval = 700 * ym

        # Compute R_curve (radius of curvature)
        left_curveR =  ((1 + (2*left_fit[0] *y_eval + left_fit[1])**2)**1.5)  / np.absolute(2*left_fit[0])
        right_curveR = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])

        xl = np.dot(self.left_fit, [700**2, 700, 1])
        xr = np.dot(self.right_fit, [700**2, 700, 1])
        pos = (1280//2 - (xl+xr)//2)*xm

        return left_curveR, right_curveR, pos

