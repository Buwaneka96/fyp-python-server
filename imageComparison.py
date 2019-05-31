import cv2
import numpy as np
import base64
from PIL import Image
from io import StringIO
import re

def readb64(base64_string):
    sbuf = StringIO()
    sbuf.write(base64.b64decode(base64_string).encode())
    pimg = Image.open(sbuf)
    return cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)

def compareImages(original, image_to_compare):

    # print('original ',original);

    # original = cv2.imread("images/login2.jpg")
    # image_to_compare = cv2.imread("images/login3.jpg")

    original = re.sub('data:image/jpeg;base64,', '', original)

    img = base64.b64decode(original); 
    npimg = np.fromstring(img, dtype=np.uint8); 
    original = cv2.imdecode(npimg, 1)

    img = base64.b64decode(image_to_compare); 
    npimg = np.fromstring(img, dtype=np.uint8); 
    image_to_compare = cv2.imdecode(npimg, 1)



    sift = cv2.xfeatures2d.SIFT_create()
    kp_1, desc_1 = sift.detectAndCompute(original, None)
    kp_2, desc_2 = sift.detectAndCompute(image_to_compare, None)

    index_params = dict(algorithm=0, trees=5)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(desc_1, desc_2, k=2)

    good_points = []
    for m, n in matches:
        if m.distance < 0.6*n.distance:
            good_points.append(m)

    # Define how similar they are
    number_keypoints = 0
    if len(kp_1) <= len(kp_2):
        number_keypoints = len(kp_1)
    else:
        number_keypoints = len(kp_2)


    print("Keypoints 1ST Image: " + str(len(kp_1)))
    print("Keypoints 2ND Image: " + str(len(kp_2)))
    print("Matched Keypoints: ", len(good_points))
    print("Match rating: ", len(good_points) / number_keypoints * 100)

    # result = cv2.drawMatches(original, kp_1, image_to_compare, kp_2, good_points, None)

    # cv2.imwrite("feature_matching.jpg", result)

    return {'rate': len(good_points) / number_keypoints * 100};