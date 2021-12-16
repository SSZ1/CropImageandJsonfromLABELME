'''
2021-12-16    SUN SHANGZHE
Crop the min rectangle of the insulator 
and generate rotated contour point coordinates by cv2.MinAreaRectangle(points coordinates of JSONS)

'''


import json
import os
import cv2
import math
import numpy as np
from numpy.core.fromnumeric import shape

#get the names of JSON files in the path
def list_json(path):
    sup_ext = ['.json']
    all_list = list(map(lambda x:os.path.join(path, x), os.listdir(path)))
    json_list = [x for x in all_list if os.path.splitext(x)[1]in sup_ext]
    return json_list

#get the names of JPG files in the path
def list_jpg(path):
    sup_ext = ['.jpg']
    all_list = list(map(lambda x:os.path.join(path, x), os.listdir(path)))
    json_list = [x for x in all_list if os.path.splitext(x)[1]in sup_ext]
    return json_list

#get the output path
def get_outpath(out_dir, name):
    if not os.path.exists(out_dir):os.makedirs(out_dir)
    return os.path.join(out_dir, os.path.basename(name))

#get the coordinates of rotated points
def rotate(ps,m):
    pts = np.float32(ps).reshape([-1, 2])  # 要映射的点
    pts = np.hstack([pts, np.ones([len(pts), 1])]).T
    target_point = np.dot(m, pts)
    target_point = [[target_point[0][x],target_point[1][x]] for x in range(len(target_point[0]))]
    return target_point

#get the image ,the box and the rotated box
def drow_box(img, cnt):
    rect_box = cv2.boundingRect(cnt)
    rotated_box = cv2.minAreaRect(cnt)

    #cv2.rectangle(img, (rect_box[0], rect_box[1]), (rect_box[0] + rect_box[2], rect_box[1] + rect_box[3]), (0, 255, 0), 2)

    box = cv2.boxPoints(rotated_box)
    box = np.int0(box)
    #cv2.drawContours(img, [box], 0, (0, 0, 255), 2)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # plt.imshow(img)
    # plt.show()

    return img, rotated_box, box

#crop the image
def crop1(img, cnt):
    horizon = True

    img, rotated_box, box = drow_box(img, cnt)

    #pts = rotated_box
    #print(box)

    center, size, angle = rotated_box[0], rotated_box[1], rotated_box[2]
    center, size = tuple(map(int, center)), tuple(map(int, size))

    #print(angle)

    if horizon:
        if size[0] < size[1]:
            angle -= 90
            w = size[1]
            h = size[0]
        else:
            w = size[0]
            h = size[1]
        size = (w, h)
        size1 = [w, h]

    height, width = img.shape[0], img.shape[1]

    M = cv2.getRotationMatrix2D(center, angle, 1)
    img_rot = cv2.warpAffine(img, M, (width, height))
    img_crop = cv2.getRectSubPix(img_rot, size, center)
    out_points = rotate(cnt, M)
    out_box = rotate(box, M)
    #print(out_box)

    #思路：先旋转检测框，获取检测框旋转之后的坐标，然后再使用旋转之后的点out_points减去旋转框之后的坐标的最小值。

    return img_crop, out_points, out_box, size1

#get the Min and Max value of the coordinates
def get_min_max(m):
    a = []
    b = []
    for i in range(len(m)):
        a.append(m[i][0])
        b.append(m[i][1])
    xmin = min(a)
    xmax = max(a)
    ymin = min(b)
    ymax = min(b)
    C = [[xmin, ymin], [xmax, ymax]]
    return C


#crop the images
def crop_imgs(load_data_dir, save_data_dir):
    print("Start cropping images!!!")
    jpg_list = list_jpg(load_data_dir)
    for img in range(len(jpg_list)):
        print(jpg_list[img])
        jpg = cv2.imread(jpg_list[img])
        ###########获取X,Y坐标的最值
       
        j = jpg_list[img][:-4] + ".json"
        with open(j, "r") as f:
            json_str = f.read()
        your_dict = json.loads(json_str)
        shapes = your_dict["shapes"]
        for label in range(len(shapes)):
            x = []
            y = []
            for k in range(len(shapes[label]["points"])):
                x.append(shapes[label]["points"][k][0])
                y.append(shapes[label]["points"][k][1])
            xmax = math.ceil(max(x))
            xmin = int(min(x))
            ymax = math.ceil(max(y))
            ymin = int(min(y))
            #print(type(jpg))
            cut_jpg = jpg[ymin:ymax, xmin:xmax]
            '''
            code to crop the image
            '''
            pts = np.array(shapes[label]["points"])
            pts = pts.astype(np.float32)
            img_crop, _, _, _= crop1(jpg, pts)

            outpath = get_outpath(save_data_dir, jpg_list[img])[:-4] + ".jpg"
            cv2.imwrite(outpath, img_crop)#, [100])
            print(jpg_list[img][:-4] + ".jpg 已写入！")
    print("Finish processing images!!!")

#edit the JSON files
def crop_jsons(load_data_dir, save_data_dir):
    print("Start cropping JSONs!!!")
    json_list = list_json(load_data_dir)
    for j in range(len(json_list)):
        #print("2222!!!")
        print(json_list[j])
        with open(json_list[j], "r") as f:
            json_str = f.read()
        your_dict = json.loads(json_str)
        your_dict["imageData"] = None
        #your_dict["imagePath"] = os.path.basename(json_list[j])[:-5] + '.jpg'
        #your_dict["imageHeight"] = y1 - y0
        #your_dict["imageWidth"] = x1 - x0

        shapes = your_dict["shapes"]
        for label in range(len(shapes)):
            pts = np.array(shapes[label]["points"])
            pts = pts.astype(np.float32)
            jpg = cv2.imread(json_list[j][:-5] + ".jpg")
            img_crop, out_points, out_box, size1= crop1(jpg, pts)
            minmaxpoints = get_min_max(out_box)
            for s in range(len(out_points)):
                out_points[s][0] = out_points[s][0] - minmaxpoints[0][0]
                out_points[s][1] = out_points[s][1] - minmaxpoints[0][1]
            
            





            # x = []
            # y = []
            # for i in range(len(shapes[label]["points"])):
            #     x.append(shapes[label]["points"][i][0])
            #     y.append(shapes[label]["points"][i][1])
            # xmax = math.ceil(max(x))
            # xmin = int(min(x))
            # ymax = math.ceil(max(y))
            # ymin = int(min(y))


            your_dict["imageHeight"] = size1[1]
            your_dict["imageWidth"] =  size1[0]
            #shapes[label]["label"] = shapes[label]["label"][:-1]
            # for k in range(len(shapes[label]["points"])):
            #     shapes[label]["points"][k][0] = shapes[label]["points"][k][0] - xmin
            #     shapes[label]["points"][k][1] = shapes[label]["points"][k][1] - ymin

            
            shapes[label]["points"] = out_points
            your_dict["shapes"] = [shapes[label]]
            # your_dict["imagePath"] = os.path.basename(json_list[j])[:-5] + shapes[label]["label"] +'.jpg'
            # save_dir = save_data_dir + os.path.basename(json_list[j])[:-5] + shapes[label]["label"] + ".json"
            your_dict["imagePath"] = os.path.basename(json_list[j])[:-5]+'.jpg'
            save_dir = save_data_dir + os.path.basename(json_list[j])[:-5]+ ".json"
            with open(save_dir, "w", encoding = "UTF-8") as f1:
                json.dump(your_dict, f1, ensure_ascii=False,indent = 2)
            print(os.path.basename(json_list[j])[:-5] + ".json已写入！！")
    print("Finish processing JSONs!!!")

def main():
    crop_imgs(load_data_dir, 'json_res7')
    crop_jsons(load_data_dir,save_data_dir)
    
if __name__ == "__main__":
    load_data_dir='json_res1' #原始图片和json文件的路径
    save_data_dir='E://科研/实验/标注数据/testjson/json_res7/' #裁剪后图片和json文件保存的路径
    main()
    print("finished all the program!!!")