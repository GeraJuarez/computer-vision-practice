import numpy as np
from cv2 import cv2
from cv_helpers import cv2_show_img, plt_show_img, start_cv_video
import matplotlib.pyplot as plt

def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    h, w = image.shape
    if width is None:
        ratio = height / float(h)
        dim = ( int(w * ratio), height )
    else:
        ratio = width / float(w)
        dim = ( width, int(h * ratio) )

    return cv2.resize(image, dim, interpolation=inter)

def feature_match(img1, img2):
    originial = img2
    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    h, w = img1.shape
    h2, w2 = img2.shape

    if h > h2 and w > w2:
        img1 = resize_with_aspect_ratio(img1, height= h2 // 2)

    pts1, desc1 = ORB_descriptor(img1, 1000)
    pts2, desc2 = ORB_descriptor(img2, 10000)

    dmatches = get_matches(desc1, desc2)

    h, w = img1.shape
    dst = find_image_in_frame(dmatches, pts1, pts2, h, w)
    
    img2 = cv2.polylines(originial, [np.int32(dst)], True, (0,0,255), 10, cv2.LINE_AA)
    res = cv2.drawMatches(img1, pts1, img2, pts2, dmatches[:5], None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
   
    return res

def ORB_descriptor(img, num_features=1000):
    orb = cv2.ORB_create(nfeatures=num_features, scoreType=cv2.ORB_FAST_SCORE)
    points, desc = orb.detectAndCompute(img, None)  
   
    return points, desc 

def process_train_image(train_img):
    '''
        Reshape train image if it has a larger resolution than the input image
    '''
    cam = cv2.VideoCapture(0)
    in_height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
    in_width = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
    train_h, train_w = train_img.shape

    if (train_h > in_height and train_w > in_width):
        new_img = resize_with_aspect_ratio(train_img, height=train_h // 2)

    return new_img

def find_image_in_frame(dmatches, train_pts, new_pts, train_img_h, train_img_w):
    src_pts = np.float32([train_pts[m.queryIdx].pt for m in dmatches]).reshape(-1,1,2)
    dst_pts = np.float32([new_pts[m.trainIdx].pt for m in dmatches]).reshape(-1,1,2)

    homography_matrix, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    pts = np.float32([ [0, 0], [0, train_img_h - 1], [train_img_w - 1, train_img_h - 1], [train_img_w - 1, 0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts, homography_matrix)

    return dst

def get_matches(train_desc, new_desc):
    '''
        Match descriptors and sort them in the order of their distance
    '''
    #cv2.NORM_HAMMING2
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(train_desc, new_desc)
    dmatches = sorted(matches, key = lambda x:x.distance)

    return dmatches

def feature_track(frame, *params):
    red_BGR = (0, 0, 255)
    LINE_WIDTH = 1
    train_pts = params[0]
    train_desc = params[1]
    train_img = params[2]
    h, w = train_img.shape

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame_pts, frame_desc = ORB_descriptor(gray_frame, 10000)
    descriptor_matches = get_matches(train_desc, frame_desc)
    points_in_frame = find_image_in_frame(descriptor_matches, train_pts, frame_pts, h, w)

    new_frame = cv2.polylines(frame, [np.int32(points_in_frame)], True, red_BGR, LINE_WIDTH, cv2.LINE_AA)

    return new_frame

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Image descriptor')
    parser.add_argument('-train', type=str, action='store', dest='train_img', required=True, help='The image to apply the filter')
    parser.add_argument('-test', type=str, action='store', dest='test_img', help='The image to test')
    args = parser.parse_args()

    if args.test_img:
        try:
            img1 = cv2.imread(args.train_img, cv2.IMREAD_COLOR)
            img2 = cv2.imread(args.test_img, cv2.IMREAD_COLOR)

            new_img = feature_match(img1, img2)
            new_img = cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR)
            plt_show_img(new_img)

        except Exception as error:
            print(error)

    else:
        img = cv2.imread(args.train_img, cv2.IMREAD_GRAYSCALE)
        img = process_train_image(img)
        points, desc = ORB_descriptor(img)

        start_cv_video(0, feature_track, points, desc, img)
