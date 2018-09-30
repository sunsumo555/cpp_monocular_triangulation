# cpp_monocular_triangulation
Summary: A monocular depth map generation algorithm used for agricultural UAV. It uses triangulation to solve the problem, if the line does not intersect due to measurement noise, the midpoint of the shortest line between the rays will be used as the intersection point

Algorithm Explaination
This algorithm takes in a video, which treats it as a stream of grayscale images. The algorithm compares the current image with the image from kNFrameSkip frames before (for example, setting KNFrameSkip = 10 will make the algorithm compare the current frame with the image from 10 frames prior). This is implemented with c++'s std::queue where new images are added to the queue, and an image is popped to compare with the current image.

With the two images in our hand, each image's features are extracted with OpenCV's ORB feature detector. We then find corresponding points with OpenCV's BFMatcher. If we found less than kMinNMatchesToAccept matches, the loop is continued without further calculation as there are not enough matches for the pair of pictures to provide useful information.

After the matches are acquired, the matches array are sorted with respect to the match distance, then only the kPercentMatchesToUse percent of the matches are used, the other are discarded (for example, kPercentMatchesToUse = 0.7 will use only 70% of the best matches and will discard the rest 30% of the matches)

After the useful matches are found. We then proceed to find the homography matrix using the findHomography function that OpenCV provided. The homography matrix is the matrix that maps the corresponding points of the current uncalibrated image to the prior uncalibrated image. The reason that we find this homography is that the homography image contains the relative camera extrinsics (the relative translation and the relative rotation of the two camera), which will be described next.

After the homography matrix is found, the camera extrinsics are calculated via the decomposeHomographyMat function that OpenCV also provided. However, there will be 4 solutions of extrinsics that are mathematically possible, but there will be only one solution that is physically possible - by physically possible I mean that all of the points are in front of the camera, which we will check by asserting if all of the depths are positive. Note that the decomposeHomographyMat function also requires the camera intrinsics, which can be calibrated with checkerboards. In this project, we used Matlab's camera calibration module to find the camera's intrinsics.

As the translation vector is relative, we have scaled it to the unit vector so that other modules can scale the translation vector by the displacement of the robot to find the absolute translation of the robot. This is done in the FindDepthsFromOrientation by changing the number 1.0 to the preferred magnitude of the translation vector.

With the 4 rotation and the translation matrices available, we proceed to find the correct pair by calculating the depth of each corresponding point by triangulation method. Instead of solving for the depths in runtime, we have solved the solution beforehand with Matlab and computed the solution directly. The closed form of the solution is written in the function FindDepth in the source.cpp. it is solved via Matlab. This is all implemented in the FindDepthsFromOrientation function in the source code.

After FindDepthsFromOrientation is called, you should get the absolute depths of features in front of the robot, resulting in a sparse depth map. However, for visualization purposes, we have scaled the depth values into [0,1] and have overlayed them as circles with color on the current image such that a red circle means that the feature is close to the robot, while a purple circle means that the feature is relatively far from the robot.

Please feel free to tell me how should I improve the code to better collaborate with you guys or anything you want me to explain more in this project. Thank you!
