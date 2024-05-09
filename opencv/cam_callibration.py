import numpy as np
import cv2 as cv
import glob
import os
import matplotlib.pyplot as plt

def calibrate(showPics=True):
    #global imgGray
    # Read Image
    root = os.getcwd()
    calibrationDir = os.path.join(root,'/home/dolce/Pictures/')  #reads the directory
    imgPathList = glob.glob(os.path.join(calibrationDir,"/home/dolce/Pictures")) # images in the directory

    # Initialize
    nRows = 9  #chess board row
    nCols = 6  #chess board column
    termCriteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,30,0.001)  #termination criteria
    worldPtsCur = np.zeros((nRows*nCols,3), np.float32)       #columns being the holder of the worldpoints
    worldPtsCur[:,:2] = np.mgrid[0:nRows,0:nCols].T.reshape(-1,2)
    worldPtsList = []
    imgPtsList = []

    # Find Corners
    for curImgPath in imgPathList:
        imgBGR = cv.imread(curImgPath) #read image
        imgGray = cv.cvtColor(imgBGR, cv.COLOR_BGR2GRAY) # convert to grey scale
        cornersFound, cornersOrg = cv.findChessboardCorners(imgGray,(nRows,nCols), None)

        if cornersFound == True:         #find all the corners and append them
            worldPtsList.append(worldPtsCur)
            cornersRefined = cv.cornerSubPix(imgGray,cornersOrg,(11,11),(-1,-1),termCriteria)
            imgPtsList.append(cornersRefined)

            if showPics:               #plot the corners by telling show pics
                cv.drawChessboardCorners(imgBGR,(nRows,nCols),cornersRefined,cornersFound)
                cv.imshow('Chessboard', imgBGR)
                cv.waitKey(500)
    cv.destroyAllWindows()

    # Calibrate
    repError,camMatrix,distCoeff,rvecs,tvecs = cv.calibrateCamera(worldPtsList, imgPtsList, imgGray.shape[::-1],None,None)
    print('Camera Matrix:\n',camMatrix)
    print("Reproj Error (pixels): {:.4f}".format(repError))

    # Save Calibration Parameters
    curFolder = os.path.dirname(os.path.abspath(_file_))
    paramPath = os.path.join(curFolder,'calibration.npz')
    np.savez(paramPath,
        repError=repError,
        camMatrix=camMatrix,
        distCoeff=distCoeff,
        rvecs=rvecs,
        tvecs=tvecs)

    return camMatrix,distCoeff

def removeDistortion(camMatrix,distCoeff):
    root = os.getcwd()
    imgPath = os.path.join(root,'/home/dolce/Pictures/.jpg')
    img = cv.imread(imgPath)

    height, width = img.shape[:2]
    camMatrixNew, roi = cv.getOptimalNewCameraMatrix(camMatrix,distCoeff,(width,height),1,(width,height))
    imgUndist = cv.undistort(img,camMatrix,distCoeff,None,camMatrixNew)

    # Draw Line to See Distortion Change
    cv.line(img,(1769,103),(1780,922),(255,255,255),2)
    cv.line(imgUndist,(1769,103),(1780,922),(255,255,255),2)

    plt.figure()
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(imgUndist)
    plt.show()

def runCalibration():
    calibrate(showPics=True)

def runRemoveDistortion():
    camMatrix, distCoeff = calibrate(showPics=False)
    removeDistortion(camMatrix,distCoeff)

if __name__ == '__main__':
    #runCalibration()
    runRemoveDistortion()
