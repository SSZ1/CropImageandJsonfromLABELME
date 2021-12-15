import json
import os
import cv2
import math
import numpy as np
######test

# your_dict = {"label":"insulator", "points":[1.2345, 2.3456]}

# print(your_dict["points"])

#测试一段代码

#列出文件夹中的json文件名
def list_json(path):
    sup_ext = ['.json']
    all_list = list(map(lambda x:os.path.join(path, x), os.listdir(path)))
    json_list = [x for x in all_list if os.path.splitext(x)[1]in sup_ext]
    return json_list


#列出文件夹中的jpg文件名
def list_jpg(path):
    sup_ext = ['.jpg']
    all_list = list(map(lambda x:os.path.join(path, x), os.listdir(path)))
    json_list = [x for x in all_list if os.path.splitext(x)[1]in sup_ext]
    return json_list


#获取输出路径
def get_outpath(out_dir, name):
    if not os.path.exists(out_dir):os.makedirs(out_dir)
    return os.path.join(out_dir, os.path.basename(name))


#旋转点集
def rotate(ps,m):
    pts = np.float32(ps).reshape([-1, 2])  # 要映射的点
    pts = np.hstack([pts, np.ones([len(pts), 1])]).T
    target_point = np.dot(m, pts)
    target_point = [[target_point[0][x],target_point[1][x]] for x in range(len(target_point[0]))]
    return target_point



#裁切最小矩形框，同时获取旋转坐标
def crop_minAreaRect(img, points):

    
    points = np.array(points)
    #print(points)
    rect = cv2.minAreaRect(points)
    #rotate img
    angle = rect[2]
    rows, cols = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle-90, 1)
    img_rot = cv2.warpAffine(img, M, (cols, rows))

    #rotate bounding box
    rect0 = (rect[0], rect[1], 0.0)
    box = cv2.boxPoints(rect0)
    pts = np.int0(cv2.transform(np.array([box]), M))[0]
    #pts[pts < 0] = 0

    #rotate the contour points
    points_seg = rotate(points, M)
    points_seg = [[points_seg[i][0] - pts[1][0], points_seg[i][1] - pts[1][1]] for i in range(len(points_seg))]



    #crop
    img_crop = img_rot[pts[1][1]:pts[0][1],pts[1][0]:pts[2][0]]
    #points_seg = np.dot(M, np.array([[points[0]], [points[1]], [[1]]]))

    return img_crop, points_seg

#根据影像和点集检测最小矩形框
def drow_box(img, cnt):
    rect_box = cv2.boundingRect(cnt)
    rotated_box = cv2.minAreaRect(cnt)

    cv2.rectangle(img, (rect_box[0], rect_box[1]), (rect_box[0] + rect_box[2], rect_box[1] + rect_box[3]), (0, 255, 0), 2)

    box = cv2.boxPoints(rotated_box)
    box = np.int0(box)
    cv2.drawContours(img, [box], 0, (0, 0, 255), 2)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # plt.imshow(img)
    # plt.show()

    return img, rotated_box, box

#裁剪影像
def crop1(img, cnt):
    horizon = True

    img, rotated_box, box = drow_box(img, cnt)

    #pts = rotated_box
    print(box)

    center, size, angle = rotated_box[0], rotated_box[1], rotated_box[2]
    center, size = tuple(map(int, center)), tuple(map(int, size))

    print(angle)

    if horizon:
        if size[0] < size[1]:
            angle -= 90
            w = size[1]
            h = size[0]
        else:
            w = size[0]
            h = size[1]
        size = (w, h)

    height, width = img.shape[0], img.shape[1]

    M = cv2.getRotationMatrix2D(center, angle, 1)
    img_rot = cv2.warpAffine(img, M, (width, height))
    img_crop = cv2.getRectSubPix(img_rot, size, center)
    out_points = rotate(cnt, M)
    out_box = rotate(box, M)
    print(out_box)

    #思路：先旋转检测框，获取检测框旋转之后的坐标，然后再使用旋转之后的点out_points减去旋转框之后的坐标的最小值。

    return img_crop, out_points



def main():
    jpg = cv2.imread("2.jpg")
    #points = [[1, 2], [3, 4], [5, 6], [0, 2], [5, 9], [10, 11], [8, 12], [3, 10], [15, 5]]

    jsn = "test2.json"
    with open(jsn, "r") as f:
        json_str = f.read()
    your_dict = json.loads(json_str)
    shapes = your_dict["shapes"]
    for label in range(len(shapes)): 
            #print(shapes[label]["points"])  
        pts = np.array(shapes[label]["points"])
        #print(type(pts[0][0]))
        #pts = np.array(points)



        pts = pts.astype(np.float32)
        #print(pts)
        #img_crop, points_seg = crop_minAreaRect(jpg, pts)
        img_crop, out_points= crop1(jpg, pts)
        


        #print(out_points)
        cv2.imshow('test', img_crop)
        cv2.waitKey(0)

if __name__ == "__main__":
    main()
