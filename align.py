# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 11:06:28 2021

@author: eee
"""

import cv2, dlib, os
import numpy as np
from skimage import transform
import torch

faceDetector = dlib.get_frontal_face_detector()
PREDICTOR_PATH = "E:/Full_Model/Fine_Tune_data/shape_predictor_68_face_landmarks.dat"  # Landmark model location
landmarkDetector = dlib.shape_predictor(PREDICTOR_PATH)  

def getFaceRect(im, faceDetector):
    faceRects = faceDetector(im, 0)
    if len(faceRects)>0:
        faceRect = faceRects[0]
        newRect = dlib.rectangle(int(faceRect.left()),int(faceRect.top()), 
                             int(faceRect.right()),int(faceRect.bottom()))
    else:
        newRect = dlib.rectangle(0,0,im.shape[1],im.shape[0])
    return newRect

def landmarks2numpy(landmarks_init):
    landmarks = landmarks_init.parts()
    points = []
    for ii in range(0, len(landmarks)):
        points.append([landmarks[ii].x, landmarks[ii].y])
    return np.array(points)

# Estimate the similarity transformation of landmarks_frame to landmarks_im
# Uses 3 points: a tip of the nose and corners of the eyes
def getRigidAlignment(landmarks_frame, landmarks_im):
    # video frame
    video_lmk = [[np.int(landmarks_frame[30][0]), np.int(landmarks_frame[30][1])], 
                 [np.int(landmarks_frame[36][0]), np.int(landmarks_frame[36][1])],
                 [np.int(landmarks_frame[45][0]), np.int(landmarks_frame[45][1])] ]

    # Corners of the eye in normalized image
    img_lmk = [ [np.int(landmarks_im[30][0]), np.int(landmarks_im[30][1])], 
                [np.int(landmarks_im[36][0]), np.int(landmarks_im[36][1])],
                [np.int(landmarks_im[45][0]), np.int(landmarks_im[45][1])] ]

    # Calculate similarity transform
    #tform = cv2.estimateRigidTransform(np.array([video_lmk]), np.array([img_lmk]), False)
    tform = transform.estimate_transform('similarity', np.array(video_lmk), np.array(img_lmk)).params[:2,:]
    return tform  

def getTriMaskAndMatrix(im,srcPoints,dstPoints,dt_im):
    maskIdc = np.zeros((im.shape[0],im.shape[1],3), np.int32)
    matrixA = np.zeros((2,3, len(dt_im)), np.float32)
    for i in range(0, len(dt_im)):
        t_src = []
        t_dst = []
        for j in range(0, 3):
            t_src.append((srcPoints[dt_im[i][j]][0], srcPoints[dt_im[i][j]][1]))
            t_dst.append((dstPoints[dt_im[i][j]][0], dstPoints[dt_im[i][j]][1]))        
        # get an inverse transformatio: from t_dst to t_src
        Ai_temp = cv2.getAffineTransform(np.array(t_dst, np.float32), np.array(t_src, np.float32))
        matrixA[:,:,i] = Ai_temp - [[1,0,0],[0,1,0]]
        # fill in a mask with triangle number
        cv2.fillConvexPoly(img=maskIdc, points=np.int32(t_dst), color=(i,i,i), lineType=8, shift=0) 
    return (maskIdc,matrixA)

# Smoothes the warp field (offsets) depending oon the distance from a face
def smoothWarpField(dstPoints, warpField):
    warpField_blured = warpField.copy()

    # calculate facial mask
    faceHullIndex = cv2.convexHull(np.array(dstPoints[:68,:]), returnPoints=False)
    faceHull = []
    for i in range(0, len(faceHullIndex)):
        faceHull.append((dstPoints[faceHullIndex[i],0], dstPoints[faceHullIndex[i],1]))
    maskFace = np.zeros((warpField.shape[0],warpField.shape[1],3), dtype=np.uint8)  
    cv2.fillConvexPoly(maskFace, np.int32(faceHull), (255, 255, 255))     

    # get distance transform
    dist_transform = cv2.distanceTransform(~maskFace[:,:,0], cv2.DIST_L2,5)
    max_dist = dist_transform.max()

    # initialize a matrix with distance ranges ang sigmas for smoothing
    maxRadius = 0.05*np.linalg.norm([warpField.shape[0], warpField.shape[1]]) # 40 pixels for 640x480 image
    thrMatrix = [[0, 0.1*max_dist, 0.1*maxRadius],
                 [0.1*max_dist, 0.2*max_dist, 0.2*maxRadius],
                 [0.2*max_dist, 0.3*max_dist, 0.3*maxRadius],
                 [0.3*max_dist, 0.4*max_dist, 0.4*maxRadius],
                 [0.4*max_dist, 0.5*max_dist, 0.5*maxRadius],
                 [0.5*max_dist, 0.6*max_dist, 0.6*maxRadius],
                 [0.6*max_dist, 0.7*max_dist, 0.7*maxRadius],
                 [0.7*max_dist, 0.8*max_dist, 0.8*maxRadius],
                 [0.8*max_dist, 0.9*max_dist, 0.9*maxRadius],
                 [0.9*max_dist, max_dist + 1, maxRadius]]
    for entry in thrMatrix:
        # select values in the range (entry[0], entry[1]]
        mask_range = np.all(np.stack((dist_transform>entry[0], dist_transform<=entry[1]), axis=2), axis=2)
        mask_range = np.stack((mask_range,mask_range), axis=2)

        warpField_temp = cv2.GaussianBlur(warpField, (0, 0), entry[2])        
        warpField_blured[mask_range] = warpField_blured[mask_range]       
    return warpField_blured

def mainWarpField(im,srcPoints,dstPoints,dt_im):
    yy,xx = np.mgrid[0:im.shape[0], 0:im.shape[1]]
    numPixels = im.shape[0] * im.shape[1]
    xxyy = np.reshape(np.stack((xx.reshape(numPixels),yy.reshape(numPixels)), axis = 1), (numPixels, 1, 2))           

    # get a mask with triangle indices
    (maskIdc, matrixA) = getTriMaskAndMatrix(im,srcPoints,dstPoints,dt_im)
      
    # compute the initial warp field (offsets) and smooth it
    warpField = getWarpInit(matrixA, maskIdc,numPixels,xxyy)  #size: im.shape[0]*im.shape[1]*2
    warpField_blured = smoothWarpField(dstPoints, warpField)
    
    # get the corresponding indices instead of offsets and make sure theu are in the image range
    warpField_idc = warpField_blured + np.stack((xx,yy), axis = 2)
    warpField_idc[:,:,0] = np.clip(warpField_idc[:,:,0],0,im.shape[1]-1)  #x
    warpField_idc[:,:,1] = np.clip(warpField_idc[:,:,1],0,im.shape[0]-1)  #y
    
    # fill in the image with corresponding indices
    im_new2 = im.copy()
    im_new2[yy,xx,:] = im[np.intc(warpField_idc[:,:,1]), np.intc(warpField_idc[:,:,0]),:]

    return im_new2

def getWarpInit(matrixA, maskIdc,numPixels,xxyy):
    maskIdc_resh = maskIdc[:,:,0].reshape(numPixels)
    warpField = np.zeros((maskIdc.shape[0],maskIdc.shape[1],2), np.float32)
    warpField = warpField.reshape((numPixels,2))
    
    # wrap triangle by triangle
    for i in range(0, matrixA.shape[2]):
        xxyy_masked = []
        xxyy_masked = xxyy[maskIdc_resh==i,:]
        # don't process empty array
        if xxyy_masked.size == 0:
            continue
        warpField_temp = cv2.transform(xxyy_masked, np.squeeze(matrixA[:,:,i]))
        warpField[maskIdc_resh==i,:] =  np.reshape(warpField_temp, (warpField_temp.shape[0], 2))  

        # reshape to original image shape 
    warpField = warpField.reshape((maskIdc.shape[0],maskIdc.shape[1],2))
    return warpField

# Omit the out-of-face control points in the template which fall out of image range after alignment
def omitOOR(landmarks_template, shape):
    outX = np.logical_or((landmarks_template[:,0] < 0),  (landmarks_template[:,0] >= shape[1]))
    outY = np.logical_or((landmarks_template[:,1] < 0), (landmarks_template[:,1] >= shape[0]))
    outXY = np.logical_or(outX,outY)
    landmarks_templateConstrained = landmarks_template.copy()
    landmarks_templateConstrained = landmarks_templateConstrained[~outXY]
    return landmarks_templateConstrained

def insertBoundaryPoints(width, height, np_array):
    ## Takes as input non-empty numpy array
    np_array = np.append(np_array,[[0, 0]],axis=0)
    np_array = np.append(np_array,[[0, height//2]],axis=0)
    np_array = np.append(np_array,[[0, height-1]],axis=0)    
    np_array = np.append(np_array,[[width//2, 0]],axis=0)
    np_array = np.append(np_array,[[width//2, height-1]],axis=0)        
    np_array = np.append(np_array,[[width-1, 0]],axis=0)
    np_array = np.append(np_array,[[width-1, height//2]],axis=0)   
    np_array = np.append(np_array,[[width-1, height-1]],axis=0)
    return np_array

# Calculate Delaunay triangles for set of points.
# Returns the vector of indices of 3 points for each triangle
def calculateDelaunayTriangles(subdiv, points):
    # Get Delaunay triangulation
    triangleList = subdiv.getTriangleList()

    # Find the indices of triangles in the points array
    delaunayTri = []

    for t in triangleList:
        # The triangle returned by getTriangleList is
        # a list of 6 coordinates of the 3 points in
        # x1, y1, x2, y2, x3, y3 format.
        # Store triangle as a list of three points
        pt = []
        pt.append((t[0], t[1]))
        pt.append((t[2], t[3]))
        pt.append((t[4], t[5]))

        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        #if rectContains(rect, pt1) and rectContains(rect, pt2) and rectContains(rect, pt3):
        # Variable to store a triangle as indices from list of points
        ind = []
        # Find the index of each vertex in the points list
        for j in range(0, 3):
            for k in range(0, len(points)):
                if(abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0):
                    ind.append(k)
                    # Store triangulation as a list of indices
        if len(ind) == 3:
            delaunayTri.append((ind[0], ind[1], ind[2]))
    return delaunayTri

def hallucinateControlPoints(landmarks_init, im_shape, INPUT_DIR='E:/Full_Model/Fine_Tune_data/features', performTriangulation = False):

    # load template control points
    templateCP_fn = os.path.join(INPUT_DIR, 'brene_controlPoints.txt')
    templateCP_init = np.int32(np.loadtxt(templateCP_fn, delimiter=' '))    
    
    # align the template to the frame
    tform = getRigidAlignment(templateCP_init, landmarks_init)
    
    # Apply similarity transform to the frame
    templateCP = np.reshape(templateCP_init, (templateCP_init.shape[0], 1, templateCP_init.shape[1]))
    templateCP = cv2.transform(templateCP, tform)
    templateCP = np.reshape(templateCP, (templateCP_init.shape[0], templateCP_init.shape[1]))
    
    # omit the points outside the image range
    templateCP_Constrained = omitOOR(templateCP, im_shape)

    # hallucinate additional keypoint on a new image 
    landmarks_list = landmarks_init.tolist()
    for p in templateCP_Constrained[68:]:
        landmarks_list.append([p[0], p[1]])
    landmarks_out = np.array(landmarks_list)
        
    subdiv_temp = None
    dt_temp = None
    if performTriangulation:
        srcTemplatePoints = templateCP_Constrained.copy()
        srcTemplatePoints = insertBoundaryPoints(im_shape[1], im_shape[0], srcTemplatePoints)
        subdiv_temp = createSubdiv2D(im_shape, srcTemplatePoints)  
        dt_temp = calculateDelaunayTriangles(subdiv_temp, srcTemplatePoints) 
    
    return (subdiv_temp, dt_temp, landmarks_out)

def createSubdiv2D(size, landmarks):
    '''
        Input
    size[0] is height
    size[1] is width    
    landmarks as in dlib-detector output
        Output
    subdiv -- Delaunay Triangulation        
    '''   
    # Rectangle to be used with Subdiv2D
    rect = (0, 0, size[1], size[0])
    
    # Create an instance of Subdiv2D
    subdiv = cv2.Subdiv2D(rect)
    
    # Insert points into subdiv
    for p in landmarks :
        subdiv.insert((p[0], p[1])) 
      
    return subdiv




im =cv2.imread( 'E:/Full_Model/Fine_Tune_data/Frames/Fake_Frames/110130.jpg')
im_height, im_width, im_channels = im.shape
newRect = getFaceRect(im, faceDetector)
landmarks_im = landmarks2numpy(landmarkDetector(im, newRect))

frame=cv2.imread( 'E:/Full_Model/Fine_Tune_data/Frames/Real_Frames/110130.jpg')
newRect = getFaceRect(frame, faceDetector)
landmarks_frame_init = landmarks2numpy(landmarkDetector(frame, newRect))

tform = getRigidAlignment(landmarks_frame_init, landmarks_im)

frame_aligned = cv2.warpAffine(frame, tform, (im_width, im_height))

# image_aligned = cv2.warpAffine(im, tform, (im_width, im_height))

import matplotlib.pyplot as plt

image = cv2.cvtColor(frame_aligned, cv2.COLOR_BGR2RGB)
imgplot = plt.imshow(image)
plt.show()

# image = cv2.cvtColor(image_aligned, cv2.COLOR_BGR2RGB)
# imgplot = plt.imshow(image)
# plt.show()


landmarks_frame = np.reshape(landmarks_frame_init, (landmarks_frame_init.shape[0], 1, landmarks_frame_init.shape[1]))
landmarks_frame = cv2.transform(landmarks_frame, tform)
landmarks_frame = np.reshape(landmarks_frame, (landmarks_frame_init.shape[0], landmarks_frame_init.shape[1]))

(subdiv_temp, dt_im, landmarks_frame) = hallucinateControlPoints(landmarks_init = landmarks_frame, 
                                                                im_shape = frame_aligned.shape, 
                                                                INPUT_DIR='E:/Full_Model/Fine_Tune_data/features', 
                                                                performTriangulation = True)
landmarks_list = landmarks_im.copy().tolist()
for p in landmarks_frame[68:]:
    landmarks_list.append([p[0], p[1]])
srcPoints = np.array(landmarks_list)
srcPoints = insertBoundaryPoints(im_width, im_height, srcPoints) 


dstPoints_frame = landmarks_frame
dstPoints_frame = insertBoundaryPoints(im_width, im_height, dstPoints_frame)

dstPoints = dstPoints_frame - srcPoints + srcPoints 

im_new = mainWarpField(im,srcPoints,dstPoints,dt_im) 

image = cv2.cvtColor(im_new, cv2.COLOR_BGR2RGB)
imgplot = plt.imshow(image)
plt.show()


