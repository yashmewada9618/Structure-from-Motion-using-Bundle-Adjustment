# %% [markdown]
# <h1><center>Northeastern University</center></h1>
# <h1><center>EECE 7150 Autonomous Field Robotics</center></h1>
# <h1><center>HW5 Submission</center></h1>
# <h3><center>Yash Mewada</center></h3>
# <h3><center>Date: 26th Oct, 2023</center></h3>
# 
# <h3>Structure from Motion</h3>
# Steps for projecting 3D points from serires of 2D images:
# <ol>
#     <li>Read two consecutive images and perform keypoint detector and matcher through these images.</li>
#     <li>Estimate the Fundamemtal matrix using these matched keypoints.</li>
#     <li>Estimate the Essential matrix from the Fundamental matrix.</li>
#     <li>Estimate the camera pose from the Essential matrix.</li>
#     <li>Triangulate the 3D points from the camera pose and matched keypoints.</li>
#     <li>Repeat the above steps for all the images in the sequence.</li>
# 
# 
# 
# 
# 
# 

# %%
import numpy as np
import cv2
import os
import math
import open3d as o3d
from matplotlib import pyplot as plt
from skimage.feature import daisy, match_descriptors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import gtsam
from gtsam import symbol_shorthand
from gtsam import (Cal3_S2, DoglegOptimizer,
                    GenericProjectionFactorCal3_S2, Marginals,
                    NonlinearFactorGraph, PinholeCameraCal3_S2, Point3, Point2,
                    Pose3, PriorFactorPoint3, PriorFactorPose3, Rot3, Values) 
"""
Author: Yash Mewada
Date: 10/26/2023
"""

# %%
class colors:
    """
    Class to define colors for printing
    """
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    YELLOW = '\033[93m'
    END = '\033[0m'


# %%
class hw5:
    
    FM_8POINT = 0
    FM_RANSAC = 1
    FM_LMEDS = 2
    FM_7POINT = 3
    
    def __init__(self,dataset):
        """
        Constructor to initialize the dataset
        Args: Path to the dataset
        """
        self.dataset = dataset
        self.K = np.array([[1,0,0],[0,1,0],[0,0,1]])
        
    def readImages(self):
        """
        Function to read images from the dataset
        """
        images = []
        files = sorted(os.listdir(self.dataset))
        for i in range(len(files)):
            files[i] = self.dataset + files[i]
            img = cv2.imread(files[i])
            image = cv2.resize(img, (0,0), fx=0.5, fy=0.5) 
            images.append(image)
        width = images[0].shape[1]
        height = images[0].shape[0]
        self.K = np.array([[width,0,width/2],
                           [0,height,height/2],
                           [0,0,1]])
        return images
    
    def featureDetector(self,img1,img2,detector="SIFT"):
        """
        Function to detect features in the images
        Args: img1, img2, doORB
        Returns: img3, goodkp1, goodkp2
        """
        img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
        
        if detector == "ORB":
            featureDetector = cv2.ORB_create()
            kp1,des1 = featureDetector.detectAndCompute(img1,None)
            kp2,des2 = featureDetector.detectAndCompute(img2,None)
            matches = cv2.FlannBasedMatcher(dict(algorithm=0, trees=5), dict(checks=32)).knnMatch(des1, des2, k=2)
            goodkp2 = []; goodkp1 = []
            for m in matches:
                if m[0].distance/m[1].distance < 0.75: # Using Lowe's ratio test
                    goodkp2.append(kp2[m[0].trainIdx].pt)
                    goodkp1.append(kp1[m[0].queryIdx].pt)
            goodkp1 = np.array(goodkp1); goodkp2 = np.array(goodkp2)
        
        elif detector == "SIFT":
            featureDetector = cv2.SIFT_create(nfeatures=4000, nOctaveLayers=12, contrastThreshold=0.025, sigma=1.5)
            kp1,des1 = featureDetector.detectAndCompute(img1,None)
            kp2,des2 = featureDetector.detectAndCompute(img2,None)
            matches = cv2.FlannBasedMatcher(dict(algorithm=0, trees=5), dict(checks=32)).knnMatch(des1, des2, k=2)
            goodkp2 = []; goodkp1 = []
            for m in matches:
                if m[0].distance/m[1].distance < 0.75: # Using Lowe's ratio test
                    goodkp2.append(kp2[m[0].trainIdx].pt)
                    goodkp1.append(kp1[m[0].queryIdx].pt)
            goodkp1 = np.array(goodkp1); goodkp2 = np.array(goodkp2)
            
        elif detector == "DAISY":
            des1,_ = daisy(img1,step=10,radius=15,rings=2,histograms=6,orientations=8,visualize=True)
            des2,_ = daisy(img2,step=10,radius=15,rings=2,histograms=6,orientations=8,visualize=True)
            descsnum1 = des1.shape[0] * des1.shape[1]
            descsnum2 = des2.shape[0] * des2.shape[1]
            
            des1 = des1.reshape(descsnum1,des1.shape[2])
            des2 = des2.reshape(descsnum2,des2.shape[2])
            matches = match_descriptors(des1,des2)
            goodkp1 = [matches[:,0]]; goodkp2 = [matches[:,1]]
            pxCors = cv2.remap(img1,goodkp1,goodkp2,cv2.INTER_LINEAR)
            print(pxCors)
            goodkp1 = np.array(goodkp1); goodkp2 = np.array(goodkp2)
        
        img3 = img2.copy()
        
        return img3,goodkp1,goodkp2

    def normalizePoints(self,pts):
        """
        Normalize the points
        Args: pts
        Returns: new_ptst,T
        """
        new_ptst = np.empty_like(pts)
        center = (0,0)
        
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
        scale = np.sqrt(2)/distance
        for i in range(len(pts)):
            new_ptst[i] *= scale
        
        T = np.array([[scale,0,-center[0]*scale],
                      [0,scale,-center[1]*scale],
                      [0,0,1]])

        return new_ptst,T
    
    def RANSAC(self,pts1,pts2,reProjThreshold=3,maxIters=10000):
        """
        Ransac Implementation
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

        #     # Adaptively determining the number of iterations
        #     while num_iterations > iterations_done:

        #         # shuffle the rows and take the first 'num_sample' rows as sample data
        #         np.random.shuffle(total_data)
        #         tempInliersPts = np.random.choice(data_size, num_sample, replace=False)
        #         tempInliers = []
        #         tempE = self.K.T @ F @ self.K
                
        #         for i in tempInliersPts:
        #             if i not in tempInliersPts:
        #                 ditance = np.abs(np.matmul(pts2[i],tempE))
        #                 if ditance < reProjThreshold:
        #                     tempInliers.append(i)
        #         sample_data = total_data[:num_sample, :]

        #         estimated_model = self.curve_fitting_model.fit(sample_data[:,:-1], sample_data[:, -1:]) ## [a b c]

        #         # count the inliers within the threshold
        #         y_cap = pts1.dot(estimated_model)
        #         err = np.abs(pts2 - y_cap.T)
        #         inlier_count = np.count_nonzero(err < reProjThreshold)

        #         # check for the best model 
        #         if inlier_count > max_inlier_count:
        #             max_inlier_count = inlier_count
        #             best_model = estimated_model


        #         prob_outlier = 1 - inlier_count/data_size
        #         print('# inliers:', inlier_count)
        #         print('# prob_outlier:', prob_outlier)
        #         num_iterations = math.log(1 - desired_prob)/math.log(1 - (1 - prob_outlier)**num_sample)
        #         iterations_done = iterations_done + 1

        #         print('# s:', iterations_done)
        #         print('# n:', num_iterations)
        #         print('# max_inlier_count: ', max_inlier_count)
            
        # return best_model
            
        
    def findFundametalMat(self,pts1,pts2,method=FM_8POINT):
        """
        Compute the fundamental matrix
        Args: pts1, pts2, method
        Returns: F
        """
        npts1,T1 = self.normalizePoints(pts1)
        npts2,T2 = self.normalizePoints(pts2)
        if method == self.FM_8POINT:
            # Estimate fundamental matrix using 8-point algorithm manually
            A = []
            for i in range(len(npts1)):
                x1 = npts1[i][0]
                y1 = npts1[i][1]
                x2 = npts2[i][0]
                y2 = npts2[i][1]
                A.append([x1*x2,x2*y1,x2,y2*x1,y2*y1,y2,x1,y1,1])
                # A.append([x1*x2,x1*y2,x1,y1*x2,y1*y2,y1,x2,y1,1])
            
            U,S,V = np.linalg.svd(A)
            F = V[-1].reshape(3,3)
            U,S,V = np.linalg.svd(F)
            minS = np.argmin(S)
            S[minS] = 0
            F = np.dot(U,np.dot(np.diag(S),V))
            F = np.dot(T2.T,np.dot(F,T1))  
            F = F/F[2,2] 
            return F
        
        elif method == self.FM_RANSAC:
            print("RANSAC TBD")
        
    def estimateEssentialMatrix(self,F):
        """
        Compute Essential Matrix from Fundamental Matrix
        """
        # E = np.matmul(F.T,np.matmul(np.diag([1,1,0]),F))
        # U,S,Vt = np.linalg.svd(E)
        # S[0] = S[1] = (S[0] + S[1]) / 2
        # S[2] = 0
        # E = U @ np.diag(S) @ Vt
        
        E = np.matmul(self.K.T,np.matmul(F,self.K))
        # E = cv2.findEssentialMat(pts1,pts2,self.K,cv2.RANSAC,0.999,1.0)[0]
        # U,S,Vt = np.linalg.svd(E)
        # s = np.array([1,1,0])
        # E = U @ np.diag(s) @ Vt
        return E
    
    def plot_camera(self,R,t,ax,scale=.5,depth=.5,faceColor='grey'):

        C = -t #camera center (in world coordinate system)
    
        #Generating camera coordinate axes
        axes = np.zeros((3,6))
        axes[0,1], axes[1,3],axes[2,5] = 1,1,1
        
        #Transforming to world coordinate system 
        axes = R.T.dot(axes)+C[:,np.newaxis]
        print("Axes: ",axes)
    
        #Plotting axes
        ax.plot3D(xs=axes[0,:2],ys=axes[1,:2],zs=axes[2,:2],c='r')
        ax.plot3D(xs=axes[0,2:4],ys=axes[1,2:4],zs=axes[2,2:4],c='g')
        ax.plot3D(xs=axes[0,4:],ys=axes[1,4:],zs=axes[2,4:],c='b')
    
        #generating 5 corners of camera polygon 
        pt1 = np.array([[0,0,0]]).T #camera centre
        pt2 = np.array([[scale,-scale,depth]]).T #upper right 
        pt3 = np.array([[scale,scale,depth]]).T #lower right 
        pt4 = np.array([[-scale,-scale,depth]]).T #upper left
        pt5 = np.array([[-scale,scale,depth]]).T #lower left
        pts = np.concatenate((pt1,pt2,pt3,pt4,pt5),axis=-1)
        #Transforming to world-coordinate system
        pts = R.T.dot(pts)+C[:,np.newaxis]
        ax.scatter3D(xs=pts[0,:],ys=pts[1,:],zs=pts[2,:],c='k')
        
        #Generating a list of vertices to be connected in polygon
        verts = [[pts[:,0],pts[:,1],pts[:,2]], [pts[:,0],pts[:,2],pts[:,-1]],
                [pts[:,0],pts[:,-1],pts[:,-2]],[pts[:,0],pts[:,-2],pts[:,1]]]
        
        #Generating a polygon now..
        ax.add_collection3d(Poly3DCollection(verts, facecolors=faceColor,
                                             linewidths=1, edgecolors='k', alpha=.25))
    
    def write_ascii_Ply(self,points,filename):
        """
        Write an ASCII output PLY file with the 3D x, y, z coordinates of the points separated by commas.
        """
        # print(points)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud(filename, pcd,write_ascii=True)
    
    def visualisePly(self,filename):
        pcd = o3d.io.read_point_cloud(filename)
        o3d.visualization.draw_geometries([pcd])
    
    def getTmatrix(self,R,t):
        """
        Get the transformation matrix from rotation and translation
        """
        T_mat = np.eye(4)
        T_mat[:3,:] = np.hstack((R,t))
        return T_mat

    def getProjMatrix(self,T):
        """
        Get the projection matrix from transformation matrix
        """
        return self.K @ np.linalg.inv(T)[:3]

    def get3Dpts(self,srcPmat,dstPmat,srcpts,dstpts):
        """
        Get the 3D points from the projection matrix
        """
        obj_pts = []
        for pt1,pt2 in zip(srcpts,dstpts):
            obj_pt = cv2.triangulatePoints(srcPmat,dstPmat,pt1,pt2) 
            obj_pt /= obj_pt[3]
            obj_pts.append([obj_pt[0],obj_pt[1],obj_pt[2]])    
        return obj_pts
        
             
        

# %% [markdown]
# Here the algorithm used for estimating Fundamental Matrix is : **Normalized 8-point algorithm**. The reason for that is orders of magnitude difference between coloumn of data matrix; hence the Least Square yields to poor and time consuming results. 

# %%
if __name__ == "__main__":
    # dataset = "/home/yash/Documents/eece7150/HW5/Dataset/buddha_images/"
    dataset = "/home/mewada/Documents/eece7150/HW5/Dataset/buddha_images/"

    hw5 = hw5(dataset)
    images = hw5.readImages()
    final3D_pts = []
    avg_depth = []
    totalRots = []
    totalTrans = []
    
    _,kp11,kp12 = hw5.featureDetector(images[0],images[1],detector="SIFT")
    # F = hw5.findFundametalMat(kp1,kp2,method=hw5.FM_8POINT)
    # E = hw5.estimateEssentialMatrix(F)
    E1,_ = cv2.findEssentialMat(kp11,kp12,hw5.K,cv2.RANSAC)
    _, rotation_matrix, translation, pose_mask = cv2.recoverPose(E1, kp11, kp12, hw5.K)
    T_init = hw5.getTmatrix(rotation_matrix,translation)
    
    # Settin up the factor graph
    L = symbol_shorthand.L
    X = symbol_shorthand.X
    K = Cal3_S2(fx=hw5.K[0,0], fy=hw5.K[1,1], s=0, u0=hw5.K[0,2], v0=hw5.K[1,2])
    measurement_noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)
    pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.3, 0.3, 0.3, 0.1, 0.1, 0.1]))
    graph = NonlinearFactorGraph()
    initial = Values()
    
    pose_factor = PriorFactorPose3(X(0), Pose3(), pose_noise)
    initial.insert(X(0), Pose3())
    initial.insert(X(1), Pose3(Rot3(rotation_matrix), Point3(translation.flatten())))
    print("T_mat",T_init)
    proj_init = hw5.getProjMatrix(T_init)
    init3Dpts = hw5.get3Dpts(hw5.K @ np.hstack((np.eye(3),np.zeros((3,1)))),proj_init,kp11,kp12)
    point_noise = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)
    point_factor = PriorFactorPoint3(L(0), init3Dpts[0], point_noise)
    initial.insert(L(0), init3Dpts[0])
    initial.insert(L(1), init3Dpts[1]) 
    graph.push_back(pose_factor)
    
    for r,(k1,k2) in enumerate(zip(kp11,kp12)):
        graph.push_back(GenericProjectionFactorCal3_S2(
                        Point2(k1[0],k1[1]), measurement_noise, X(0), L(r), K))
        graph.push_back(GenericProjectionFactorCal3_S2(
                        Point2(k2[0],k2[1]), measurement_noise, X(1), L(r), K))
        
    # initial.print('Initial Estimates:\n')
    r1 = np.eye(3)
    t1 = np.zeros((3,1))
    fig = plt.figure(figsize=(13.0, 11.0))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('auto')
    
    for i in range(1,len(images)-1):
        macthes,kp1,kp2 = hw5.featureDetector(images[i],images[i+1],detector="SIFT")
        E,inliers = cv2.findEssentialMat(kp1,kp2,hw5.K,cv2.RANSAC)

        R1,R2,t = cv2.decomposeEssentialMat(E)
        _, rot,tmat,mask = cv2.recoverPose(E,kp1,kp2,hw5.K)
        kp1 = kp1[mask.ravel()>0]
        kp2 = kp2[mask.ravel()>0]
        r1 = np.matmul(R1,r1)
        t1 = t1 + np.array(t)
        hw5.plot_camera(r1,t1[:,0],ax,faceColor='r')
        initial.insert(X(i+1), Pose3(Rot3(R1), Point3(t.flatten())))
        pts3D = hw5.get3Dpts(hw5.K @ np.hstack((np.eye(3),np.zeros((3,1)))),hw5.getProjMatrix(T_init),kp1,kp2)
        # initial.insert(L(i+1), pts3D[0])

        factor_new = GenericProjectionFactorCal3_S2(
                        Point2(kp1[0][0],kp1[0][1]), measurement_noise, X(i), L(i), K)
        graph.push_back(factor_new)
        factor_new1 = GenericProjectionFactorCal3_S2(
                        Point2(kp2[0][0],kp2[0][1]), measurement_noise, X(i+1), L(i), K)
        graph.push_back(factor_new1)
        
        
            
        mostpts = 0
        bestR = np.eye(3)
        bestt = np.zeros((3,1))
        P1 = np.dot(hw5.K,np.concatenate((bestR,bestt),axis=1)),
        P2s = [ np.dot(hw5.K,np.concatenate((R1,t),axis=1)),
                np.dot(hw5.K,np.concatenate((R1,-t),axis=1)),
                np.dot(hw5.K,np.concatenate((R2,t),axis=1)),
                np.dot(hw5.K,np.concatenate((R2,-t),axis=1))]

        obj_pts_cam = []
        for cmIdx,p2 in enumerate(P2s):
            obj_pts = []
            R = p2[:,:3]
            t = p2[:,3]
            for j,(pt1,pt2) in enumerate(zip(kp1,kp2)):
                # if not inliers[j]:
                #     continue
                obj_pt = cv2.triangulatePoints(P1,p2,pt1,pt2) 
                obj_pt /= obj_pt[3]
                if obj_pt[2] > 0:
                    avg_depth.append(obj_pt[2])
                    obj_pts.append([obj_pt[0],obj_pt[1],obj_pt[2]])
            obj_pts_cam.append(obj_pts)
            # obj_pts = np.array(obj_pts).reshape(len(obj_pts), 3)
            # final3D_pts.append(obj_pts)

        best_cam_idx = np.array([len(obj_pts_cam[0]),len(obj_pts_cam[1]),
                                 len(obj_pts_cam[2]),len(obj_pts_cam[3])]).argmax()
        print("Best camera index: ",best_cam_idx)
        max_pts = len(obj_pts_cam[best_cam_idx])
        print('Number of object points', max_pts)
        print("Number of 3D points: ",len(obj_pts_cam))

        MAX_DEPTH = 60.
        obj_pts = []
        for pt in obj_pts_cam[best_cam_idx]:
            if pt[2] < MAX_DEPTH:
                obj_pts.append(pt)
        obj_pts = np.array(obj_pts).reshape(len(obj_pts), 3)
        final3D_pts.append(obj_pts)
        
        # for k, (pt1, pt2) in enumerate(zip(kp1, kp2)):
        #     if inliers[k]:
        #         cv2.circle(images[i], (int(pt1[0]), int(pt1[1])), 7, (255,0,0), -1)
        #         cv2.circle(images[i+1], (int(pt2[0]), int(pt2[1])), 7, (255,0,0), -1)

        # img1 = np.concatenate((images[i],images[i+1]),axis=1)
        # new_sz = 1200.*800.
        # img1 = cv2.resize(img1, (int(np.sqrt(img1.shape[1] * new_sz / img1.shape[0])),
        #                              int(np.sqrt (img1.shape[0] * new_sz / img1.shape[1]))))
        # # cv2.imshow("matches", img1)
        # cv2.imwrite(f'matches_E{i}.png', img1)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    # graph.print('Factor Graph:\n')
    print("Number of 3D points: ",np.shape(final3D_pts))
    for i,objpt in enumerate(final3D_pts):
        # print("Object points: ",objpt)
        # if i == 0 or i == 1:
        #     continue
        # initial.insert(L(i), obj_pts)
        ax.scatter(objpt[:,0], objpt[:,1], objpt[:,2], c='r', marker='o', s=3)
    
    # Optimize the graph and print results
    # params = gtsam.LevenbergMarquardtParams()
    # optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
    # # graph.print('Factor Graph:\n')
    # result = optimizer.optimize()
    # result.print('Final results:\n')
    
    # print('initial error = {}'.format(graph.error(initial)))
    # print('final error = {}'.format(graph.error(result)))
    
    # marginals = Marginals(graph, result)
    # for objpts in final3D_pts:
    #     ax.scatter(objpts[:,0], objpts[:,1], objpts[:,2], c='r', marker='o', s=3)
    avgZ = np.mean(avg_depth)
    print("Average depth: ",avgZ)
    maxZ = np.max(avg_depth)
    print("Max depth: ",maxZ)
    second_maxZ = np.sort(avg_depth)[-2]
    print("Second Max depth: ",second_maxZ)
    minZ = np.min(avg_depth)
    print("Min depth: ",minZ)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('depth')
    ax.view_init(azim=-80, elev=110)
    # save figures
    # cv2.imwrite('matches_E.png', img1)
    # plt.savefig('reconstruction_3D.png')
    plt.show()
    # for R,t in zip(totalRots,totalTrans):
    #     hw5.plot_camera(R,t,ax)
    # plt.show()


