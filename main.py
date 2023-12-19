import numpy as np
import cv2
import os
import math
import open3d as o3d
from matplotlib import pyplot as plt
from skimage.feature import daisy, match_descriptors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import gtsam
import matplotlib.cm as cm
import gtsam.utils.plot as gtsam_plot

"""
Author: Yash Mewada
Date: 10/26/2023
"""
# Certain bits of this code have been taken from Aniket's Code


class colors:
    """
    Class to define colors for printing
    """

    GREEN = "\033[92m"
    BLUE = "\033[94m"
    YELLOW = "\033[93m"
    END = "\033[0m"


class Point3D(object):
    def __init__(self, pts3d, pts2d):
        self.pts3d = pts3d
        self.pts2d = pts2d


class hw5:
    FM_8POINT = 0
    FM_RANSAC = 1
    FM_LMEDS = 2
    FM_7POINT = 3

    def __init__(self, dataset):
        """
        Constructor to initialize the dataset
        Args: Path to the dataset
        """
        self.dataset = dataset
        self.K = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.struc3Dpts = {}
        self.cloud = []

    def readImages(self):
        """
        Function to read images from the dataset
        """
        images = []
        files = sorted(os.listdir(self.dataset))
        for i in range(len(files)):
            files[i] = self.dataset + files[i]
            img = cv2.imread(files[i])
            # image = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
            images.append(img)
        width = images[0].shape[1]
        height = images[0].shape[0]
        self.K = np.array([[width, 0, width / 2], [0, height, height / 2], [0, 0, 1]])
        return images

    def featureDetector(self, img1, img2, detector="SIFT"):
        """
        Function to detect features in the images
        Args: img1, img2, doORB
        Returns: img3, goodkp1, goodkp2
        """
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        if detector == "ORB":
            featureDetector = cv2.ORB_create()
            kp1, des1 = featureDetector.detectAndCompute(img1, None)
            kp2, des2 = featureDetector.detectAndCompute(img2, None)
            matches = cv2.FlannBasedMatcher(
                dict(algorithm=0, trees=5), dict(checks=32)
            ).knnMatch(des1, des2, k=2)
            goodkp2 = []
            goodkp1 = []
            for m in matches:
                if m[0].distance / m[1].distance < 0.75:  # Using Lowe's ratio test
                    goodkp2.append(kp2[m[0].trainIdx].pt)
                    goodkp1.append(kp1[m[0].queryIdx].pt)
            goodkp1 = np.array(goodkp1)
            goodkp2 = np.array(goodkp2)

        elif detector == "SIFT":
            #
            featureDetector = cv2.SIFT_create(
                nfeatures=5000,
                nOctaveLayers=3,
                contrastThreshold=0.04,
            )
            kp1, des1 = featureDetector.detectAndCompute(img1, None)
            kp2, des2 = featureDetector.detectAndCompute(img2, None)
            matches = cv2.FlannBasedMatcher(
                dict(algorithm=0, trees=5), dict(checks=50)
            ).knnMatch(des1, des2, k=2)
            goodkp2 = []
            goodkp1 = []
            for m in matches:
                if m[0].distance / m[1].distance < 0.75:  # Using Lowe's ratio test
                    goodkp2.append(kp2[m[0].trainIdx].pt)
                    goodkp1.append(kp1[m[0].queryIdx].pt)
            goodkp1 = np.array(goodkp1)
            goodkp2 = np.array(goodkp2)
            # goodkp1 = self.nonMaxSuppression(goodkp1); goodkp2 = self.nonMaxSuppression(goodkp2)

        elif detector == "DAISY":
            des1, _ = daisy(
                img1,
                step=10,
                radius=15,
                rings=2,
                histograms=6,
                orientations=8,
                visualize=True,
            )
            des2, _ = daisy(
                img2,
                step=10,
                radius=15,
                rings=2,
                histograms=6,
                orientations=8,
                visualize=True,
            )
            descsnum1 = des1.shape[0] * des1.shape[1]
            descsnum2 = des2.shape[0] * des2.shape[1]

            des1 = des1.reshape(descsnum1, des1.shape[2])
            des2 = des2.reshape(descsnum2, des2.shape[2])
            matches = match_descriptors(des1, des2)
            goodkp1 = [matches[:, 0]]
            goodkp2 = [matches[:, 1]]
            pxCors = cv2.remap(img1, goodkp1, goodkp2, cv2.INTER_LINEAR)
            print(pxCors)
            goodkp1 = np.array(goodkp1)
            goodkp2 = np.array(goodkp2)

        img3 = img2.copy()

        return img3, goodkp1, goodkp2

    def nonMaxSuppression(self, keypts):
        """
        Non Max Suppression to remove the keypoints which are close to each other
        """
        i = 0
        new_keypts = []
        for i in range(len(keypts) - 1):
            print(keypts[i])
            if np.abs(keypts[i][0] - keypts[i + 1][0]) < 10:
                if np.abs(keypts[i][1] - keypts[i + 1][1]) > 10:
                    new_keypts.append(keypts[i])
                    i -= 1
            i += 1
        return keypts

    def normalizePoints(self, pts):
        """
        Normalize the points
        Args: pts
        Returns: new_ptst,T
        """
        new_ptst = np.empty_like(pts)
        center = (0, 0)

        # Compute centers and average distances for each set of points
        for i in range(len(pts)):
            center += pts[i]
        center /= len(pts)

        distance = 0.0
        for i in range(len(pts)):
            new_ptst[i] = pts[i] - center
            distance += np.linalg.norm(new_ptst[i])
        distance /= len(pts)
        # print("distance: ",distance)
        scale = np.sqrt(2) / distance
        for i in range(len(pts)):
            new_ptst[i] *= scale

        T = np.array(
            [[scale, 0, -center[0] * scale], [0, scale, -center[1] * scale], [0, 0, 1]]
        )

        return new_ptst, T

    def RANSAC(self, pts1, pts2, reProjThreshold=3, maxIters=10000):
        """
        Ransac Implementation (Not Used here!)
        """
        if len(pts1) < 15 or len(pts2) < 15:
            print("Not enough points")
            return None
        else:
            num_iterations = math.inf
            iterations_done = 0
            num_sample = 5

            max_inlier_count = 0
            best_model = None
            bestError = math.inf

            prob_outlier = 0.5
            desired_prob = 0.95

            total_data = np.column_stack((pts1, pts2))  ## [ A | Y]
            data_size = len(total_data)

            # Adaptively determining the number of iterations
            while num_iterations > iterations_done:
                # shuffle the rows and take the first 'num_sample' rows as sample data
                np.random.shuffle(total_data)
                tempInliersPts = np.random.choice(data_size, num_sample, replace=False)
                tempInliers = []
                tempE = self.K.T @ F @ self.K

                for i in tempInliersPts:
                    if i not in tempInliersPts:
                        ditance = np.abs(np.matmul(pts2[i], tempE))
                        if ditance < reProjThreshold:
                            tempInliers.append(i)
                sample_data = total_data[:num_sample, :]

                estimated_model = self.curve_fitting_model.fit(
                    sample_data[:, :-1], sample_data[:, -1:]
                )  ## [a b c]

                # count the inliers within the threshold
                y_cap = pts1.dot(estimated_model)
                err = np.abs(pts2 - y_cap.T)
                inlier_count = np.count_nonzero(err < reProjThreshold)

                # check for the best model
                if inlier_count > max_inlier_count:
                    max_inlier_count = inlier_count
                    best_model = estimated_model

                prob_outlier = 1 - inlier_count / data_size
                print("# inliers:", inlier_count)
                print("# prob_outlier:", prob_outlier)
                num_iterations = math.log(1 - desired_prob) / math.log(
                    1 - (1 - prob_outlier) ** num_sample
                )
                iterations_done = iterations_done + 1

                print("# s:", iterations_done)
                print("# n:", num_iterations)
                print("# max_inlier_count: ", max_inlier_count)

        return best_model

    def findFundametalMat(self, pts1, pts2, method=FM_8POINT):
        """
        Compute the fundamental matrix
        Args: pts1, pts2, method
        Returns: F
        """
        npts1, T1 = self.normalizePoints(pts1)
        npts2, T2 = self.normalizePoints(pts2)
        if method == self.FM_8POINT:
            # Estimate fundamental matrix using 8-point algorithm manually
            A = []
            for i in range(len(npts1)):
                x1 = npts1[i][0]
                y1 = npts1[i][1]
                x2 = npts2[i][0]
                y2 = npts2[i][1]
                A.append([x1 * x2, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1])
                # A.append([x1*x2,x1*y2,x1,y1*x2,y1*y2,y1,x2,y1,1])

            U, S, V = np.linalg.svd(A)
            F = V[-1].reshape(3, 3)
            U, S, V = np.linalg.svd(F)
            minS = np.argmin(S)
            S[minS] = 0
            F = np.dot(U, np.dot(np.diag(S), V))
            F = np.dot(T2.T, np.dot(F, T1))
            F = F / F[2, 2]
            return F

        elif method == self.FM_RANSAC:
            print("RANSAC TBD")

    def estimateEssentialMatrix(self, F):
        """
        Compute Essential Matrix from Fundamental Matrix
        """
        E_est = np.dot(self.K.T, np.dot(F, self.K))
        # reconstructing E by correcting singular values
        U, S, V = np.linalg.svd(E_est, full_matrices=True)
        S = np.diag(S)
        S[0, 0], S[1, 1], S[2, 2] = 1, 1, 0
        E = np.dot(U, np.dot(S, V))
        return E

    def plot_camera(self, R, t, ax, scale=0.5, depth=0.5, faceColor="grey"):
        C = -t  # camera center (in world coordinate system)

        # Generating camera coordinate axes
        axes = np.zeros((3, 6))
        axes[0, 1], axes[1, 3], axes[2, 5] = 1, 1, 1

        # Transforming to world coordinate system
        axes = R.T.dot(axes) + C[:, np.newaxis]

        # Plotting axes
        ax.plot3D(xs=axes[0, :2], ys=axes[1, :2], zs=axes[2, :2], c="r")
        ax.plot3D(xs=axes[0, 2:4], ys=axes[1, 2:4], zs=axes[2, 2:4], c="g")
        ax.plot3D(xs=axes[0, 4:], ys=axes[1, 4:], zs=axes[2, 4:], c="b")

        # generating 5 corners of camera polygon
        pt1 = np.array([[0, 0, 0]]).T  # camera centre
        pt2 = np.array([[scale, -scale, depth]]).T  # upper right
        pt3 = np.array([[scale, scale, depth]]).T  # lower right
        pt4 = np.array([[-scale, -scale, depth]]).T  # upper left
        pt5 = np.array([[-scale, scale, depth]]).T  # lower left
        pts = np.concatenate((pt1, pt2, pt3, pt4, pt5), axis=-1)
        # Transforming to world-coordinate system
        pts = R.T.dot(pts) + C[:, np.newaxis]
        ax.scatter3D(xs=pts[0, :], ys=pts[1, :], zs=pts[2, :], c="k")

        # Generating a list of vertices to be connected in polygon
        verts = [
            [pts[:, 0], pts[:, 1], pts[:, 2]],
            [pts[:, 0], pts[:, 2], pts[:, -1]],
            [pts[:, 0], pts[:, -1], pts[:, -2]],
            [pts[:, 0], pts[:, -2], pts[:, 1]],
        ]

        # Generating a polygon now..
        ax.add_collection3d(
            Poly3DCollection(
                verts, facecolors=faceColor, linewidths=1, edgecolors="k", alpha=0.25
            )
        )
        ax

    def triangulatePoints(self, pts1, pts2, P1, P2):
        """
        Triangulate the points from the projection matrix
        """
        points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)

        # Convert to 3d points
        points_3d = points_4d[:3, :] / points_4d[3, :]

        # Filter out points that are behind the camera
        mask = points_3d[2, :] > 0
        points_3d = points_3d[:, mask]
        pts1 = pts1[mask]
        pts2 = pts2[mask]

        return points_3d.T, pts1, pts2

    def getTmatrix(self, R, t):
        """
        Get the transformation matrix from rotation and translation
        """
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t if t.ndim == 1 else t.ravel()
        return T

    def transformPoints(self, pts, R, T):
        """
        Transform the points from one frame to another
        """
        return np.dot(R, pts).T + T

    def getProjMatrix(self, T):
        """
        Get the projection matrix from transformation matrix
        """
        return self.K @ np.linalg.inv(T)[:3, :]

    def get3Dpts(self, srcPmat, dstPmat, srcpts, dstpts):
        """
        Get the 3D points from the projection matrix
        """
        obj_pts = []
        for pt1, pt2 in zip(srcpts, dstpts):
            obj_pt = cv2.triangulatePoints(srcPmat, dstPmat, pt1, pt2)
            obj_pt /= obj_pt[3]
            obj_pts.append([obj_pt[0], obj_pt[1], obj_pt[2]])
        return obj_pts

    def normaliseScale(self, points, reference_depth=1.0):
        """
        Normalise the scale of the points
        """
        average_depth = np.mean(points[:, 2])
        scale_factor = reference_depth / average_depth
        print("Scale factor: ", scale_factor)
        normalized_points = points * scale_factor

        return normalized_points

    def getpt3dColor(self, pts3d, images):
        """
        Get the color of the 3d points
        """
        colors = []

        for pt in pts3d:
            r, g, b = 0, 0, 0
            for k in pt.pts2d.keys():
                img = images[k]
                pt2d = pt.pts2d[k]
                r += img[int(pt2d[1]), int(pt2d[0]), 0]
                g += img[int(pt2d[1]), int(pt2d[0]), 1]
                b += img[int(pt2d[1]), int(pt2d[0]), 2]
            colors.append(
                [
                    r / len(pt.pts2d.keys()),
                    g / len(pt.pts2d.keys()),
                    b / len(pt.pts2d.keys()),
                ]
            )
        colors = np.array(colors) / 255.0
        return colors

    def findCommonPoints(self, idx, prevkps, q1):
        """
        Find the common points between two sets of points
        and find their respective 3d points and their indices
        """
        common_pts = []
        mIndices = []
        mpts3d = []
        for i, pt1 in enumerate(q1):
            for pt2 in prevkps:
                if np.array_equal(pt1, pt2):
                    common_pts.append(pt1)
                    mIndices.append(i)
                    # mpts3d.append(self.struc3Dpts["l"+str(idx)][i])
        for i, pt in enumerate(common_pts):
            for pt3d in self.cloud:
                try:
                    if np.array_equal(pt, pt3d.pts2d[idx]):
                        print("found", pt3d.pts3d)
                        mpts3d.append(pt3d.pts3d)
                        break
                except KeyError:
                    continue
            for j, pt2 in enumerate(self.struc3Dpts["x" + str(idx) + "2"]):
                try:
                    if np.array_equal(pt, pt2):
                        mpts3d.append(self.struc3Dpts["l" + str(idx)][j])
                        break
                except KeyError:
                    continue
        print("----------------------------")
        print("[+] Number of common points: ", len(common_pts))
        print("[+] Number of 3d points: ", len(mpts3d))
        common_pts = np.array(common_pts, dtype=np.float32)
        mpts3d = np.array(mpts3d, dtype=np.float32)

        assert len(common_pts) == len(mpts3d)

        return common_pts, mpts3d, mIndices

    def find3dpt(self, pt3, idx):
        """
        Find if the 3d point is already present in the point cloud
        """
        for i, pt in enumerate(self.struc3Dpts["l" + str(idx - 1)]):
            try:
                if np.array_equal(pt3, pt):
                    return True, i
            except KeyError:
                continue
        return False, None

    def find3ddemo(self, pt3, idx):
        """
        Find if the 3d point is already present in the point cloud
        """
        for oldpt3d in self.cloud:
            try:
                if np.array_equal(pt3.pts2d[idx], oldpt3d.pts2d[idx]):
                    return True, oldpt3d
            except KeyError:
                continue
        return False, None

    def find2dpt(self, pt2, idx, f):
        """
        Find if the 2d point is already present in the point cloud
        """
        for i, pt in enumerate(self.struc3Dpts["x" + str(idx - 1) + str(f)]):
            try:
                if np.array_equal(pt2, pt):
                    return True, i
            except KeyError:
                continue
        return False, None

    def createPointCloud(self, points3d, index, kp1, kp2):
        """
        Initialize the point cloud
        """
        for i, pt in enumerate(points3d):
            self.cloud.append(Point3D(pt, {index: kp1[i], index + 1: kp2[i]}))

    def update3dpc(self, idx, pts3d, kp1, kp2):
        """
        Update the 3D point cloud by checking if the 3D points are already present
        if not then append the 3D points to the point cloud
        """
        newpts3d = []
        newkp2 = []
        newkp1 = []
        for i, pt3d in enumerate(pts3d):
            point = Point3D(pt3d, {idx: kp1[i], idx + 1: kp2[i]})

            f3d, oldpt3d = self.find3ddemo(point, idx)
            # print("f3d: ", f3d, " oldpt3d: ", oldpt3d)

            found3d, ind = self.find3dpt(pt3d, idx)
            found2d, ind2 = self.find2dpt(kp2[i], idx, 2)
            found2d1, ind21 = self.find2dpt(kp1[i], idx, 1)
            if f3d:
                oldpt3d.pts2d[idx + 1] = kp2[i]
            else:
                self.cloud.append(point)
            if not found3d:
                newpts3d.append(pt3d)
            elif found3d:
                self.struc3Dpts["l" + str(idx)][ind].append(pt3d)

            if not found2d:
                newkp2.append(kp2[i])
            elif found2d:
                self.struc3Dpts["x" + str(idx) + "2"][ind2].append(kp2[i])

            if not found2d1:
                newkp1.append(kp1[i])
            elif found2d1:
                self.struc3Dpts["x" + str(idx) + "1"][ind21].append(kp1[i])

        self.struc3Dpts["l" + str(idx)] = pts3d
        self.struc3Dpts["x" + str(idx) + "1"] = newkp1
        self.struc3Dpts["x" + str(idx) + "2"] = newkp2

    def get3dptsforinit(self, images):
        final3D_pts = {}
        print("[+] Processing initial set of images")
        r1 = np.eye(3)
        t1 = np.zeros((3, 1))
        totalR = []
        totalT = []
        totalR.append(r1)
        totalT.append(t1)
        keypoints = {}

        # Perform feature matching between the first two images
        _, kp1, kp2 = self.featureDetector(images[0], images[1], detector="SIFT")

        # get essential matrix and refine the keypointst
        E, inliers = cv2.findEssentialMat(kp1, kp2, self.K, cv2.RANSAC, 0.999, 1.0)
        kp1 = kp1[inliers.ravel() == 1]
        kp2 = kp2[inliers.ravel() == 1]

        # get the rotation and translation matrix and set the camera pose in the dictionary
        _, rot, tmat, __ = cv2.recoverPose(E, kp1, kp2, hw5.K)
        T21 = self.getTmatrix(rot, tmat)
        Tw0 = np.eye(4)
        self.struc3Dpts["tw0"] = Tw0
        self.struc3Dpts["tw1"] = np.linalg.inv(T21)

        # Get the projection matrix for the first two images
        P1 = self.getProjMatrix(Tw0)
        P2 = self.getProjMatrix(self.struc3Dpts["tw1"])

        # triangulate the 3D points
        pts3d, KP1, KP2 = self.triangulatePoints(kp1, kp2, P1, P2)
        final3D_pts["l0"] = pts3d
        self.struc3Dpts["l0"] = pts3d
        self.createPointCloud(pts3d, 0, KP1, KP2)
        self.struc3Dpts["x01"] = KP1.tolist()
        self.struc3Dpts["x02"] = KP2.tolist()

        print("[+] Initial Pointcloud done")

    def get3Dpts(self, images):
        # self.get3dptsforinit(images)
        print("[+] Processing remaining images")

        prevkeypts = self.struc3Dpts["x02"]
        j = 0
        for i in range(1, len(images) - 1):
            _, kp1, kp2 = self.featureDetector(
                images[i], images[i + 1], detector="SIFT"
            )

            # get refined keypoints and essential matrix
            _, inliers = cv2.findEssentialMat(kp1, kp2, self.K, cv2.RANSAC, 0.999, 1.0)
            kp1 = kp1[inliers.ravel() == 1]
            kp2 = kp2[inliers.ravel() == 1]
            m2dpts1, m3dpts, mIndices = self.findCommonPoints(j, prevkeypts, kp1)

            # find the matched2d points in second image and use those to calculate the pose.
            m2dpts2 = kp2[mIndices]
            print(f"{colors.GREEN}[+] Processing image: {colors.END}", i)
            print("[+] Number of matched points: ", len(m2dpts1))
            print("[+] Number of initial points: ", len(m3dpts))
            T20 = self.getRnT(m3dpts, m2dpts2)
            self.struc3Dpts["tw" + str(i + 1)] = np.linalg.inv(T20)

            # get the projection matrix for the current image
            P1 = self.getProjMatrix(self.struc3Dpts["tw" + str(i)])
            P2 = self.getProjMatrix(self.struc3Dpts["tw" + str(i + 1)])

            # triangulate the 3D points
            pts3d, kp1, kp2 = self.triangulatePoints(kp1, kp2, P1, P2)
            self.update3dpc(i, pts3d, kp1, kp2)
            # self.struc3Dpts["l"+str(i)] = pts3d
            # self.struc3Dpts["x"+str(i)+"1"] = kp1.tolist()
            # self.struc3Dpts["x"+str(i)+"2"] = kp2.tolist()
            prevkeypts = kp2
            print("----------------------------")
            j += 1

        print("[+] Pointcloud done")
        print("[+] keys: ", self.struc3Dpts.keys())

    def getRnT(self, pts3d, pts2d):
        """
        Obtain transformation matrix from 3D points and 2D points using PnP
        """
        ret, rvec, tvec, _ = cv2.solvePnPRansac(pts3d, pts2d, self.K, None)
        print("Rvec: ", ret)
        if ret:
            R = cv2.Rodrigues(rvec)[0]
            T = self.getTmatrix(R, tvec)
        else:
            print(f"{colors.BLUE}PnP failed{colors.END}")
            R = np.eye(4)
            t = np.zeros((4, 1))
            T = self.getTmatrix(R, t)
        return T

    def printFactorGraph(
        self,
        graph,
        est,
        X,
    ):
        """
        Print the factor graph based ion given marginal and values
        """
        numofImages = 24

        # print(est.size())
        totalR = []
        totalT = []
        # for i in range(numofImages):
        #     gtsam_plot.plot_pose3(X(i), est.atPose3(X(i)))
        # # Plot the edges
        for edge_index in range(graph.size())[1:]:
            key1 = graph.at(edge_index).keys()
            print("key1: ", key1)
            pose1 = est.atPose2(key1)
            pose2 = est.atPose2(key1)
            plt.plot([pose1.x(), pose2.x()], [pose1.y(), pose2.y()], color="blue")

        plt.title("Factor Graph Before Optimization")
        plt.show()

    def optimize(self):
        L = gtsam.symbol_shorthand.L
        X = gtsam.symbol_shorthand.X

        # Set the Cam matrix in factor graph
        K = gtsam.Cal3_S2(self.K[0, 0], self.K[1, 1], 0, self.K[0, 2], self.K[1, 2])

        # Create the factor graph
        graph = gtsam.NonlinearFactorGraph()

        # set the noise model on the pose and landmark factor
        measurement_noise = gtsam.noiseModel.Isotropic.Sigma(2, 20.0)
        poseNoise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([0.3, 0.3, 0.3, 0.1, 0.1, 0.1])
        )
        factor = gtsam.PriorFactorPose3(
            X(0), gtsam.Pose3(self.struc3Dpts["tw0"]), poseNoise
        )
        graph.push_back(factor)
        # Create the initial estimate to the graph
        # Add the landmark noise
        pts3dNoise = gtsam.noiseModel.Isotropic.Sigma(3, 1.0)
        factor = gtsam.PriorFactorPoint3(
            L(0), gtsam.Point3(self.struc3Dpts["l0"][0]), pts3dNoise
        )
        graph.push_back(factor)

        initialEstimate = gtsam.Values()

        # Insert camera poses and 3d points
        k = 0
        p = 0
        for i, pt in enumerate(self.cloud):
            origins = list(pt.pts2d.keys())
            initialEstimate.insert(L(i), gtsam.Point3(pt.pts3d))
            for origin in origins:
                measurement = pt.pts2d[origin]
                factor = gtsam.GenericProjectionFactorCal3_S2(
                    measurement, measurement_noise, X(int(origin)), L(i), K
                )
                graph.push_back(factor)

        for i, (key, value) in enumerate(self.struc3Dpts.items()):
            if key[0] == "t":
                j = int(key[2:])
                initialEstimate.insert(X(j), gtsam.Pose3(value))

        params = gtsam.LevenbergMarquardtParams()
        params.setVerbosity("TERMINATION")
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initialEstimate, params)
        res = optimizer.optimize()

        print("Optimization complete")
        print('initial error = {}'.format(graph.error(initialEstimate)))
        print(f"{colors.BLUE}Converged Error: {colors.END}", graph.error(res))

        # Update the camera poses and 3d points
        # pts3dColors = self.getpt3dColor(self.cloud, images)
        for i, (key, value) in enumerate(self.struc3Dpts.items()):
            if key[0] == "t":
                j = int(key[2:])
                self.struc3Dpts[key] = res.atPose3(X(j)).matrix()

        newpts3d = []
        for i, pt in enumerate(self.cloud):
            pt3d = res.atPoint3(L(i))
            if pt3d[2] <= 300:
                newpts3d.append(pt3d)

        print("len of newpts3d: ", len(newpts3d))

        fig = plt.figure(figsize=(13.0, 11.0))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_aspect("auto")
        ax.scatter(
            np.array(newpts3d)[:, 0],
            np.array(newpts3d)[:, 1],
            np.array(newpts3d)[:, 2],
            c="b",
            marker="o",
            s=3,
            label="3D points after Bundle Adjustment",
        )
        # filteredpts = []
        newR = []
        newT = []
        for i, (key, value) in enumerate(self.struc3Dpts.items()):
            if key[0] == "t":
                R = np.array(value)[:3, :3]
                t = np.array(value)[:3, 3]
                newR.append(R)
                newT.append(t)
                self.plot_camera(R, t, ax, faceColor="g")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("depth")
        ax.legend()
        plt.show()

        return newpts3d, newR, newT

    def sfminitial(self, images, plot=False):
        """
        Initial Structure from Motion without Bundle Adjustment
        """

        self.get3dptsforinit(images)
        self.get3Dpts(images)

        fig = plt.figure(figsize=(13.0, 11.0))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_aspect("auto")
        filteredpts = []
        nonfilteredpts = []
        pts3dColors = self.getpt3dColor(self.cloud, images)
        filteredColors = []
        for i, pts3d in enumerate(self.cloud):
            pt = pts3d.pts3d
            nonfilteredpts.append([pt[0], pt[1], pt[2]])
            if pt[2] <= 300:
                filteredColors.append(pts3dColors[i])
                filteredpts.append([pt[0], pt[1], pt[2]])
        print("Colors in 3d points: ", pts3dColors)
        oldR = []
        oldT = []
        for i, (key, value) in enumerate(self.struc3Dpts.items()):
            if key[0] == "t":
                R = np.array(value)[:3, :3]
                t = np.array(value)[:3, 3]
                oldR.append(R)
                oldT.append(t)
                self.plot_camera(R, t, ax, faceColor="r", depth=5, scale=5)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("depth")
        ax.legend()
        if plot:
            plt.show()
        return nonfilteredpts, filteredpts, oldR, oldT

    def compareResults(self, noisypts, nonnoisypts, newR, newT, oldR, oldT):
        """
        Compare the results of the bundle adjustment
        """

        fig = plt.figure(figsize=(13.0, 11.0))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_aspect("auto")

        for oR, oT in zip(oldR, oldT):
            self.plot_camera(oR, oT, ax, faceColor="r", depth=5, scale=5)
        for nR, nT in zip(newR, newT):
            self.plot_camera(nR, nT, ax, faceColor="g", depth=5, scale=5)

        ax.scatter(
            np.array(noisypts)[:, 0],
            np.array(noisypts)[:, 1],
            np.array(noisypts)[:, 2],
            c="r",
            marker="o",
            s=3,
            label="3D points before Bundle Adjustment",
        )
        ax.scatter(
            np.array(nonnoisypts)[:, 0],
            np.array(nonnoisypts)[:, 1],
            np.array(nonnoisypts)[:, 2],
            c="b",
            marker="o",
            s=3,
            label="3D points after Bundle Adjustment",
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("depth")
        ax.legend()
        plt.title("Comparison of Bundle Adjustment")
        plt.show()


if __name__ == "__main__":
    dataset = "/home/yash/Documents/eece7150/HW5/Dataset/buddha_images/"

    hw5 = hw5(dataset)
    images = hw5.readImages()
    print("[+] Number of images: ", len(images))
    print(
        f"{colors.BLUE}[+] Creating 3D point cloud without bundle Adjustment{colors.END}"
    )
    noisyPoints, pltpoints, oldR, oldT = hw5.sfminitial(images, plot=False)

    print(f"{colors.BLUE}[+] Starting Factor graph Optimization{colors.END}")
    nonnoisyPoints, newR, newt = hw5.optimize()

    print(f"{colors.BLUE}[+] Total Noisy points: {colors.END}", len(pltpoints))
    print(f"{colors.BLUE}[+] Total Non-Noisy points: {colors.END}", len(nonnoisyPoints))

    hw5.compareResults(pltpoints, nonnoisyPoints, newR, newt, oldR, oldT)
