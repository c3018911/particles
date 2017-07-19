# USAGE
# python object_size.py --image images/example_01.png --width 0.955
# python object_size.py --image images/example_02.png --width 0.955
# python object_size.py --image images/example_03.png --width 3.5


#http://docs.opencv.org/trunk/dd/d49/tutorial_py_contour_features.html

# import the necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
import matplotlib.pyplot as plt
import time
import csv
from itertools import izip


def in_circle(center_x, center_y, radius, x, y):
    square_dist = (center_x - x) ** 2 + (center_y - y) ** 2
    return square_dist <= radius ** 2


def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged

def inverte(imagem):
    imagem = (255-imagem)
    #cv2.imwrite(name, imagem)
    return imagem
# construct the argument parse and parse the arguments
    
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
    help="path to the input image")
ap.add_argument("-w", "--width", type=float, required=True,
    help="width of the left-most object in the image (in inches)")
args = vars(ap.parse_args())

# load the image, convert it to grayscale, and blur it slightly
image = cv2.imread((args["image"]))
orig=image.copy()
#cv2.imwrite('frames/0.jpg',image)

height, width = image.shape[:2]
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
'''
cv2.namedWindow('thresh',cv2.WINDOW_NORMAL)
cv2.resizeWindow('thresh', 900,900)
cv2.imshow("thresh", thresh)
cv2.waitKey(0)
'''

# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=3)

# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening,cv2.cv.CV_DIST_L2,5)
'''
cv2.namedWindow('dist_transform',cv2.WINDOW_NORMAL)
cv2.resizeWindow('dist_transform', 900,900)
cv2.imshow("dist_transform", dist_transform)
cv2.waitKey(0)
exit(0)
'''
ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)
'''
cv2.namedWindow('unknown',cv2.WINDOW_NORMAL)
cv2.resizeWindow('unknown', 900,900)
cv2.imshow("unknown", unknown)
cv2.waitKey(0)
exit(0)
'''

'''
# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)
# Add one to all labels so that sure background is not 0, but 1
markers = markers+1
# Now, mark the region of unknown with zero
markers[unknown==255] = 0

markers = cv2.watershed(orig,markers)
orig[markers == -1] = [255,0,0]
'''
'''
cv2.namedWindow('orig',cv2.WINDOW_NORMAL)
cv2.resizeWindow('orig', 900,900)
cv2.imshow("orig", orig)
cv2.waitKey(0)
exit(0)
'''
# perform edge detection, then perform a dilation + erosion to
# close gaps in between object edges
#edged = cv2.Canny(gray, 75, 100) #50,100 originally
edged=auto_canny(gray)
kernel = np.ones((2,2),np.uint8)
#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(119,119))

edged = cv2.dilate(edged, kernel, iterations=2)
edged = cv2.erode(edged, kernel, iterations=2)
'''
cv2.namedWindow('edged',cv2.WINDOW_NORMAL)
cv2.resizeWindow('edged', 900,900)
cv2.imshow("edged", edged)
cv2.waitKey(0)
'''

# find contours in the edge map
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]

'''
dummy=np.zeros((height, width))
#for c in cnts:
#    if cv2.contourArea(c) > 150:

cv2.drawContours(dummy,cnts,-1,(255,0,0),-1)

cv2.namedWindow('dummy',cv2.WINDOW_NORMAL)
cv2.resizeWindow('dummy', 900,900)
cv2.imshow("dummy", dummy)
cv2.waitKey(0)


cnts = cv2.findContours(dummy, cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
dummy=image.copy()
cv2.drawContours(dummy,cnts,-1,(0,255,0),5)
cv2.namedWindow('dummy',cv2.WINDOW_NORMAL)
cv2.resizeWindow('dummy', 900,900)
cv2.imshow("dummy", dummy)
cv2.waitKey(0)
exit(0)
'''



# sort the contours from left-to-right and initialize the
# 'pixels per metric' calibration variable
(cnts, _) = contours.sort_contours(cnts)
pixelsPerMetric = None
n=0 # number of pixels

#rectangular particle shape... see particle.png
diaA=[]
diaB=[]
Area=[]

img = np.zeros((height,width),np.uint8)
# loop over the contours individually
for c in cnts:
    # if the contour is not sufficiently large, ignore it
    print 'len c =%i'%len(c)
    if cv2.contourArea(c) < 50:
        continue
#===========================Box=================================================
    # compute the rotated bounding box of the contour

    box = cv2.minAreaRect(c)
    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    box = np.array(box, dtype="int")

    # order the points in the contour such that they appear
    # in top-left, top-right, bottom-right, and bottom-left
    # order, then draw the outline of the rotated bounding
    # box
    box = perspective.order_points(box)
    cv2.drawContours(image, [box.astype("int")], -1, (0, 255, 0), 2)
    
    
    '''
    # loop over the original points and draw them
    for (x, y) in box:
        cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)
    '''
    # unpack the ordered bounding box, then compute the midpoint
    # between the top-left and top-right coordinates, followed by
    # the midpoint between bottom-left and bottom-right coordinates
    (tl, tr, br, bl) = box
    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)

    # compute the midpoint between the top-left and top-right points,
    # followed by the midpoint between the top-righ and bottom-right
    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)
    
    '''
    # draw the midpoints on the image
    cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

    # draw lines between the midpoints
    cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
        (255, 0, 255), 2)
    cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
        (255, 0, 255), 2)
    '''
    # compute the Euclidean distance between the midpoints
    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

    #set A as the smallest dimension
    if dA>dB:
        dummy=dA
        dA=dB
        dB=dummy

    # if the pixels per metric has not been initialized, then
    # compute it as the ratio of pixels to supplied metric
    # (in this case, inches)
    if pixelsPerMetric is None:
        pixelsPerMetric = dB / args["width"]

    # compute the size of the object
    dimA = dA / pixelsPerMetric
    dimB = dB / pixelsPerMetric
    if n>0:
        diaA.append(dimA)
        diaB.append(dimB)
        Area.append(dimA*dimB)
    n+=1

    
    # draw the object sizes on the image
    cv2.putText(image, "A={:.3f}in".format(dimA),
        (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
        2, (255, 0, 0), 2)
    cv2.putText(image, "B={:.3f}in".format(dimB),
        (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
        2, (255, 0, 0), 2)
    cv2.drawContours(image, c, -1, (0, 255, 0), 2)
#===========================Hull================================================
    hull = cv2.convexHull(c)
    cv2.drawContours(image,[hull],-1,(147,0,255),2)
    
#===========================Minimum enclosing circle============================
    (x,y),radius = cv2.minEnclosingCircle(c)
    center = (int(x),int(y))
    radius = int(radius) # see http://code.opencv.org/issues/3362
    cv2.circle(image,center,radius,(0,255,0),2)

#===========================Max inscribed circle================================
    dist2=np.zeros((height,width))
    #http://answers.opencv.org/question/1494/pointpolygontest-is-not-working-properly/
    for ind_y in range(image.shape[0]):
        for ind_x in range(image.shape[1]):
            #pointPolygonTest quicker without distance calc ... i.e. False
            val=cv2.pointPolygonTest(c,(ind_y,ind_x),False)
            if val==1:
                dist2[ind_y,ind_x] = cv2.pointPolygonTest(c,(ind_y,ind_x),True)
            else:
                dist2[ind_y,ind_x]=-10e7 #set large dummay value
    '''
    cur_img=orig.copy()
    cv2.drawContours(cur_img, [box.astype("int")], -1, (0, 255, 0), 2)
    cv2.drawContours(cur_img, c, -1, (0, 255, 0), 4)
    cv2.namedWindow('cur_img',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('cur_img', 900,900)
    cv2.imshow("cur_img", cur_img)
    cv2.waitKey(0)
    '''
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(dist2)
    #print maxLoc, minVal, maxVal
    cv2.circle(image,(maxLoc[1],maxLoc[0]),int(abs((maxVal))),(0,255,255),2)
    cv2.putText(image, str(int(maxVal)),
        (maxLoc[1],maxLoc[0]), cv2.FONT_HERSHEY_SIMPLEX,
        1, (0, 255, 0), 3)
#========================== Particle characteristics ===========================

    roughness= cv2.arcLength(c,True)/cv2.arcLength(hull,True)
    area = cv2.contourArea(c)
    perimeter = 2*np.pi*radius #cv2.arcLength(c,True), similar error to http://code.opencv.org/issues/3362
    roundness=1/((perimeter**2)/(4*np.pi*area))
  
    print "n=%i, p=%.4f,A=%.5f, R=%.4f, roughness=%.4f"%(n,perimeter,area,roundness,roughness)
    st="%i,R=%.3f,r=%.3f"%(n,roundness,roughness)
    cv2.putText(image, st,
        center, cv2.FONT_HERSHEY_SIMPLEX,
        1, (0, 255, 0), 3)

    

values, baseA = np.histogram(diaA,bins=200)
cumulativeA = np.cumsum(values)
cumulativeA =cumulativeA/float(max(cumulativeA))

values, baseB = np.histogram(diaB,bins=200)
cumulativeB = np.cumsum(values)
cumulativeB =cumulativeB/float(max(cumulativeB))

#convert to mm
baseA*=25.4
baseB*=25.4

values, baseArea = np.histogram(Area,bins=200)
cumulativeArea=np.cumsum(values)
cumulativeArea=cumulativeArea/float(max(cumulativeArea))
#convert to mm
baseArea*=25.4**2


   
cv2.namedWindow('Image',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Image', 900,900)
cv2.imshow("Image", image)
cv2.waitKey(0)

cv2.imwrite('Contour.jpg',image)
cv2.destroyWindow("Image")
'''
#smallest circle
cv2.namedWindow('Image',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Image', 900,900)
cv2.imshow("Image", img)
cv2.waitKey(0)
'''

#output to file
with open('%s.csv' %args["image"].rstrip('.JPG'), 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['baseA','cumulativeA','baseB','cumulativeB','baseA','cumulativeArea','baseB','cumulativeArea'])
    writer.writerows(izip(baseA[:-1],cumulativeA*100,baseB[:-1],cumulativeB*100,baseA[:-1],cumulativeArea*100,baseB[:-1],cumulativeArea*100))


