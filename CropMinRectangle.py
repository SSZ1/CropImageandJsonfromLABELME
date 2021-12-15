import json
import glob
import cv2
import os
import math
import numpy as np
from numpy.core.fromnumeric import shape




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
    points = points.astype(np.float32)

    rect = cv2.minAreaRect(points)
    #rotate img
    angle = rect[2]
    rows, cols = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D((cols/2, rows/2), -angle, 1)
    img_rot = cv2.warpAffine(img, M, (cols, rows))

    #rotate bounding box
    rect0 = (rect[0], rect[1], 0.0)
    box = cv2.boxPoints(rect0)
    pts = np.int0(cv2.transform(np.array([box]), M))[0]
    pts[pts < 0] = 0

    #rotate the contour points
    points_seg = rotate(points, M)
    points_seg = [[points_seg[i][0] - pts[1][0], points_seg[i][1] - pts[1][1]] for i in range(len(points_seg))]



    #crop
    img_crop = img_rot[pts[1][1]:pts[0][1],pts[1][0]:pts[2][0]]
    #points_seg = np.dot(M, np.array([[points[0]], [points[1]], [[1]]]))

    return img_crop, points_seg



def crop_ImgsAndJSON(load_data_dir, save_data_dir):
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
            #print(shapes[label]["points"])
            img_crop, points_seg = crop_minAreaRect(jpg, shapes[label]["points"])
            #img_crop, points_seg = crop_minAreaRect(jpg, [[1, 2], [2, 5], [5, 6], [4, 6]])
            outpath = get_outpath(save_data_dir, jpg_list[img])[:-4] + ".jpg"
            cv2.imwrite(outpath, img_crop)#, [100])
            print(jpg_list[img][:-4] + ".jpg 已写入！")
            shapes[label]["points"] = points_seg
            your_dict["shapes"] = [shapes[label]]
            save_dir = save_data_dir + os.path.basename(jpg_list[img])[:-4] + ".json"
            with open(save_dir, "w", encoding = "UTF-8") as f1:
                json.dump(your_dict, f1, ensure_ascii=False,indent = 2)
            print(os.path.basename(jpg_list[img])[:-4] + ".json已写入！！")





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
            width = xmax - xmin
            height = ymax - ymin
            #print(type(jpg))



            if width >= height:
                tmp = int(width / 4)
                #新建4个列表用来存放不同范围的坐标
                tmp_points1 = []
                tmp_points2 = []
                tmp_points3 = []
                tmp_points4 = []

               
                #tmp_points = shapes[label]["points"][:]
                for k in range(len(shapes[label]["points"])):
                    if shapes[label]["points"][k][0] <= xmin + tmp:
                        tmp_points1.append(shapes[label]["points"][k])
                    elif shapes[label]["points"][k][0] > xmin + tmp and shapes[label]["points"][k][0] <= xmin + 2 * tmp:
                        tmp_points2.append(shapes[label]["points"][k])
                    elif shapes[label]["points"][k][0] > xmin + 2 * tmp and shapes[label]["points"][k][0] <= xmin + 3 * tmp:
                        tmp_points3.append(shapes[label]["points"][k])
                    else:
                        tmp_points4.append(shapes[label]["points"][k])

                x1 = []
                y1 = []
                for j in range(len(tmp_points1)):
                    x1.append(tmp_points1[j][0])
                    y1.append(tmp_points1[j][1])
                x1max = math.ceil(max(x1))
                x1min = int(min(x1))
                y1max = math.ceil(max(y1))
                y1min = int(min(y1))
                #width = x1max - x1min
                #height = y1max - y1min

                cut_jpg1 = jpg[y1min:y1max, x1min:x1max]
                outpath1 = get_outpath(save_data_dir, jpg_list[img])[:-4] + "_1.jpg"
                cv2.imwrite(outpath1, cut_jpg1)#, [100])
                print(jpg_list[img][:-4] + "_1.jpg 已写入！")

                x2 = []
                y2 = []
                for j in range(len(tmp_points2)):
                    x2.append(tmp_points2[j][0])
                    y2.append(tmp_points2[j][1])
                x2max = math.ceil(max(x2))
                x2min = int(min(x2))
                y2max = math.ceil(max(y2))
                y2min = int(min(y2))

                cut_jpg2 = jpg[y2min:y2max, x2min:x2max]
                outpath2= get_outpath(save_data_dir, jpg_list[img])[:-4] + "_2.jpg"
                cv2.imwrite(outpath2, cut_jpg2)#, [100])
                print(jpg_list[img][:-4] + "_2.jpg 已写入！")

                x3 = []
                y3 = []
                for j in range(len(tmp_points3)):
                    x3.append(tmp_points3[j][0])
                    y3.append(tmp_points3[j][1])
                x3max = math.ceil(max(x3))
                x3min = int(min(x3))
                y3max = math.ceil(max(y3))
                y3min = int(min(y3))

                cut_jpg3 = jpg[y3min:y3max, x3min:x3max]
                outpath3= get_outpath(save_data_dir, jpg_list[img])[:-4] + "_3.jpg"
                cv2.imwrite(outpath3, cut_jpg3)#, [100])
                print(jpg_list[img][:-4] + "_3.jpg 已写入！")

                x4 = []
                y4 = []
                for j in range(len(tmp_points4)):
                    x4.append(tmp_points4[j][0])
                    y4.append(tmp_points4[j][1])
                x4max = math.ceil(max(x4))
                x4min = int(min(x4))
                y4max = math.ceil(max(y4))
                y4min = int(min(y4))

                cut_jpg4 = jpg[y4min:y4max, x4min:x4max]
                outpath4= get_outpath(save_data_dir, jpg_list[img])[:-4] + "_4.jpg"
                cv2.imwrite(outpath4, cut_jpg4)#, [100])
                print(jpg_list[img][:-4] + "_4.jpg 已写入！")
            elif width < height:
                tmp = int(height / 4)
                #新建4个列表用来存放不同范围的坐标
                tmp_points1 = []
                tmp_points2 = []
                tmp_points3 = []
                tmp_points4 = []

               
                #tmp_points = shapes[label]["points"][:]
                for k in range(len(shapes[label]["points"])):
                    if shapes[label]["points"][k][1] <= ymin + tmp:
                        tmp_points1.append(shapes[label]["points"][k])
                    elif shapes[label]["points"][k][1] > ymin + tmp and shapes[label]["points"][k][1] <= ymin + 2 * tmp:
                        tmp_points2.append(shapes[label]["points"][k])
                    elif shapes[label]["points"][k][1] > ymin + 2 * tmp and shapes[label]["points"][k][1] <= ymin + 3 * tmp:
                        tmp_points3.append(shapes[label]["points"][k])
                    else:
                        tmp_points4.append(shapes[label]["points"][k])
                
                x1 = []
                y1 = []
                for j in range(len(tmp_points1)):
                    x1.append(tmp_points1[j][0])
                    y1.append(tmp_points1[j][1])
                x1max = math.ceil(max(x1))
                x1min = int(min(x1))
                y1max = math.ceil(max(y1))
                y1min = int(min(y1))
                #width = x1max - x1min
                #height = y1max - y1min

                cut_jpg1 = jpg[y1min:y1max, x1min:x1max]
                outpath1 = get_outpath(save_data_dir, jpg_list[img])[:-4] + "_1.jpg"
                cv2.imwrite(outpath1, cut_jpg1)#, [100])
                print(jpg_list[img][:-4] + "_1.jpg 已写入！")

                x2 = []
                y2 = []
                for j in range(len(tmp_points2)):
                    x2.append(tmp_points2[j][0])
                    y2.append(tmp_points2[j][1])
                x2max = math.ceil(max(x2))
                x2min = int(min(x2))
                y2max = math.ceil(max(y2))
                y2min = int(min(y2))

                cut_jpg2 = jpg[y2min:y2max, x2min:x2max]
                outpath2= get_outpath(save_data_dir, jpg_list[img])[:-4] + "_2.jpg"
                cv2.imwrite(outpath2, cut_jpg2)#, [100])
                print(jpg_list[img][:-4] + "_2.jpg 已写入！")

                x3 = []
                y3 = []
                for j in range(len(tmp_points3)):
                    x3.append(tmp_points3[j][0])
                    y3.append(tmp_points3[j][1])
                x3max = math.ceil(max(x3))
                x3min = int(min(x3))
                y3max = math.ceil(max(y3))
                y3min = int(min(y3))

                cut_jpg3 = jpg[y3min:y3max, x3min:x3max]
                outpath3= get_outpath(save_data_dir, jpg_list[img])[:-4] + "_3.jpg"
                cv2.imwrite(outpath3, cut_jpg3)#, [100])
                print(jpg_list[img][:-4] + "_3.jpg 已写入！")

                x4 = []
                y4 = []
                for j in range(len(tmp_points4)):
                    x4.append(tmp_points4[j][0])
                    y4.append(tmp_points4[j][1])
                x4max = math.ceil(max(x4))
                x4min = int(min(x4))
                y4max = math.ceil(max(y4))
                y4min = int(min(y4))

                cut_jpg4 = jpg[y4min:y4max, x4min:x4max]
                outpath4= get_outpath(save_data_dir, jpg_list[img])[:-4] + "_4.jpg"
                cv2.imwrite(outpath4, cut_jpg4)#, [100])
                print(jpg_list[img][:-4] + "_4.jpg 已写入！")


            '''  
            if width >= height: 
                tmp = int(width / 4) 
                cut_jpg1 = jpg[ymin:ymax, xmin:xmin + tmp]
                outpath1 = get_outpath(save_data_dir, jpg_list[img])[:-4] + "_1.jpg"
                cv2.imwrite(outpath1, cut_jpg1)#, [100])
                print(jpg_list[img][:-4] + "_1.jpg 已写入！")

                cut_jpg2 = jpg[ymin:ymax, xmin + tmp:xmin + 2 * tmp]
                outpath2 = get_outpath(save_data_dir, jpg_list[img])[:-4] + "_2.jpg"
                cv2.imwrite(outpath2, cut_jpg2)#, [100])
                print(jpg_list[img][:-4] + "_2.jpg 已写入！")

                cut_jpg3 = jpg[ymin:ymax, xmin + 2 * tmp:xmin + 3 * tmp]
                outpath3 = get_outpath(save_data_dir, jpg_list[img])[:-4] + "_3.jpg"
                cv2.imwrite(outpath3, cut_jpg3)#, [100])
                print(jpg_list[img][:-4] + "_3.jpg 已写入！")

                cut_jpg4 = jpg[ymin:ymax, xmin + 3 * tmp:xmax]
                outpath4 = get_outpath(save_data_dir, jpg_list[img])[:-4] + "_4.jpg"
                cv2.imwrite(outpath4, cut_jpg4)#, [100])
                print(jpg_list[img][:-4] + "_4.jpg 已写入！")
            else:
                tmp = int(height / 4) 
                cut_jpg1 = jpg[ymin:ymin + tmp, xmin:xmax]
                outpath1 = get_outpath(save_data_dir, jpg_list[img])[:-4] + "_1.jpg"
                cv2.imwrite(outpath1, cut_jpg1)#, [100])
                print(jpg_list[img][:-4] + "_1.jpg 已写入！")

                cut_jpg2 = jpg[ymin + tmp:ymin + 2 * tmp, xmin:xmax]
                outpath2 = get_outpath(save_data_dir, jpg_list[img])[:-4] + "_2.jpg"
                cv2.imwrite(outpath2, cut_jpg2)#, [100])
                print(jpg_list[img][:-4] + "_2.jpg 已写入！")

                cut_jpg3 = jpg[ymin + 2 * tmp:ymin + 3 * tmp, xmin:xmax]
                outpath3 = get_outpath(save_data_dir, jpg_list[img])[:-4] + "_3.jpg"
                cv2.imwrite(outpath3, cut_jpg3)#, [100])
                print(jpg_list[img][:-4] + "_3.jpg 已写入！")

                cut_jpg4 = jpg[ymin + 3 * tmp:ymax, xmin:xmax]
                outpath4 = get_outpath(save_data_dir, jpg_list[img])[:-4] + "_4.jpg"
                cv2.imwrite(outpath4, cut_jpg4)#, [100])
                print(jpg_list[img][:-4] + "_4.jpg 已写入！")
                '''
    print("Finish processing images!!!")





def crop_jsons(load_data_dir, save_data_dir):
    print("Start cropping JSONs!!!")
    json_list = list_json(load_data_dir)
    for j in range(len(json_list)):
        #print("2222!!!")
        #print(json_list[j])
        with open(json_list[j], "r") as f:
            json_str = f.read()
        your_dict = json.loads(json_str)
        your_dict["imageData"] = None
        #your_dict["imagePath"] = os.path.basename(json_list[j])[:-5] + '.jpg'
        #your_dict["imageHeight"] = y1 - y0
        #your_dict["imageWidth"] = x1 - x0

        shapes = your_dict["shapes"]
        for label in range(len(shapes)):
            x = []
            y = []
            for i in range(len(shapes[label]["points"])):
                x.append(shapes[label]["points"][i][0])
                y.append(shapes[label]["points"][i][1])
            xmax = math.ceil(max(x))
            xmin = int(min(x))
            ymax = math.ceil(max(y))
            ymin = int(min(y))

            width = xmax - xmin
            height = ymax - ymin
            #print(type(jpg))
            if width >= height: 
                tmp = int(width / 4)
                #新建4个列表用来存放不同范围的坐标
                tmp_points1 = []
                tmp_points2 = []
                tmp_points3 = []
                tmp_points4 = []

               
                #tmp_points = shapes[label]["points"][:]
                for k in range(len(shapes[label]["points"])):
                    if shapes[label]["points"][k][0] <= xmin + tmp:
                        tmp_points1.append(shapes[label]["points"][k])
                    elif shapes[label]["points"][k][0] > xmin + tmp and shapes[label]["points"][k][0] <= xmin + 2 * tmp:
                        tmp_points2.append(shapes[label]["points"][k])
                    elif shapes[label]["points"][k][0] > xmin + 2 * tmp and shapes[label]["points"][k][0] <= xmin + 3 * tmp:
                        tmp_points3.append(shapes[label]["points"][k])
                    else:
                        tmp_points4.append(shapes[label]["points"][k])
                
                #第一个切片
                x1 = []
                y1 = []
                for ji in range(len(tmp_points1)):
                    x1.append(tmp_points1[ji][0])
                    y1.append(tmp_points1[ji][1])
                x1max = math.ceil(max(x1))
                x1min = int(min(x1))
                y1max = math.ceil(max(y1))
                y1min = int(min(y1))
                #width = x1max - x1min
                #height = y1max - y1min

                for ji in range(len(tmp_points1)):
                    tmp_points1[ji][0] = tmp_points1[ji][0] - x1min
                    tmp_points1[ji][1] = tmp_points1[ji][1] - y1min

                your_dict["imageHeight"] = y1max - y1min
                your_dict["imageWidth"] =  x1max - x1min
                if len(tmp_points1) >= 4:
                   shapes[label]["points"] = tmp_points1
                else:
                    shapes[label]["points"] = [[0, 0], [0, 1], [2, 2], [1,0]]
                #shapes[label]["points"] = tmp_points
                your_dict["shapes"] = [shapes[label]]
                your_dict["imagePath"] = os.path.basename(json_list[j])[:-5] +'_1.jpg'
                save_dir = save_data_dir + os.path.basename(json_list[j])[:-5] + "_1.json"
                with open(save_dir, "w", encoding = "UTF-8") as f1:
                    json.dump(your_dict, f1, ensure_ascii=False,indent = 2)
                print(os.path.basename(json_list[j])[:-5] + "_1.json已写入！！")
                
                #第二个切片
                x2 = []
                y2 = []
                for ji in range(len(tmp_points2)):
                    x2.append(tmp_points2[ji][0])
                    y2.append(tmp_points2[ji][1])
                x2max = math.ceil(max(x2))
                x2min = int(min(x2))
                y2max = math.ceil(max(y2))
                y2min = int(min(y2))
                #width = x1max - x1min
                #height = y1max - y1min

                for ji in range(len(tmp_points2)):
                    tmp_points2[ji][0] = tmp_points2[ji][0] - x2min
                    tmp_points2[ji][1] = tmp_points2[ji][1] - y2min
                your_dict["imageHeight"] = y2max - y2min
                your_dict["imageWidth"] =  x2max - x2min
                if len(tmp_points2) >= 4:
                   shapes[label]["points"] = tmp_points2
                else:
                    shapes[label]["points"] = [[0, 0], [0, 1], [2, 2], [1,0]]
                #shapes[label]["points"] = tmp_points
                your_dict["shapes"] = [shapes[label]]
                your_dict["imagePath"] = os.path.basename(json_list[j])[:-5] +'_2.jpg'
                save_dir = save_data_dir + os.path.basename(json_list[j])[:-5] + "_2.json"
                with open(save_dir, "w", encoding = "UTF-8") as f1:
                    json.dump(your_dict, f1, ensure_ascii=False,indent = 2)
                print(os.path.basename(json_list[j])[:-5] + "_2.json已写入！！")

                #第三个切片
                x3 = []
                y3 = []
                for ji in range(len(tmp_points3)):
                    x3.append(tmp_points3[ji][0])
                    y3.append(tmp_points3[ji][1])
                x3max = math.ceil(max(x3))
                x3min = int(min(x3))
                y3max = math.ceil(max(y3))
                y3min = int(min(y3))
                #width = x1max - x1min
                #height = y1max - y1min

                for ji in range(len(tmp_points3)):
                    tmp_points3[ji][0] = tmp_points3[ji][0] - x3min
                    tmp_points3[ji][1] = tmp_points3[ji][1] - y3min
                your_dict["imageHeight"] = y3max - y3min
                your_dict["imageWidth"] =  x3max - x3min
                if len(tmp_points3) >= 4:
                   shapes[label]["points"] = tmp_points3
                else:
                    shapes[label]["points"] = [[0, 0], [0, 1], [2, 2], [1,0]]
                #shapes[label]["points"] = tmp_points
                your_dict["shapes"] = [shapes[label]]
                your_dict["imagePath"] = os.path.basename(json_list[j])[:-5] +'_3.jpg'
                save_dir = save_data_dir + os.path.basename(json_list[j])[:-5] + "_3.json"
                with open(save_dir, "w", encoding = "UTF-8") as f1:
                    json.dump(your_dict, f1, ensure_ascii=False,indent = 2)
                print(os.path.basename(json_list[j])[:-5] + "_3.json已写入！！")
                
                #第四个切片
                x4 = []
                y4 = []
                for ji in range(len(tmp_points4)):
                    x4.append(tmp_points4[ji][0])
                    y4.append(tmp_points4[ji][1])
                x4max = math.ceil(max(x4))
                x4min = int(min(x4))
                y4max = math.ceil(max(y4))
                y4min = int(min(y4))
                #width = x1max - x1min
                #height = y1max - y1min

                for ji in range(len(tmp_points4)):
                    tmp_points4[ji][0] = tmp_points4[ji][0] - x4min
                    tmp_points4[ji][1] = tmp_points4[ji][1] - y4min
                your_dict["imageHeight"] = y4max - y4min
                your_dict["imageWidth"] =  x4max - x4min
                if len(tmp_points4) >= 4:
                   shapes[label]["points"] = tmp_points4
                else:
                    shapes[label]["points"] = [[0, 0], [0, 1], [2, 2], [1,0]]
                #shapes[label]["points"] = tmp_points
                your_dict["shapes"] = [shapes[label]]
                your_dict["imagePath"] = os.path.basename(json_list[j])[:-5] +'_4.jpg'
                save_dir = save_data_dir + os.path.basename(json_list[j])[:-5] + "_4.json"
                with open(save_dir, "w", encoding = "UTF-8") as f1:
                    json.dump(your_dict, f1, ensure_ascii=False,indent = 2)
                print(os.path.basename(json_list[j])[:-5] + "_4.json已写入！！")
            elif width < height: 
                tmp = int(height / 4)
                #新建4个列表用来存放不同范围的坐标
                tmp_points1 = []
                tmp_points2 = []
                tmp_points3 = []
                tmp_points4 = []

               
                #tmp_points = shapes[label]["points"][:]
                for k in range(len(shapes[label]["points"])):
                    if shapes[label]["points"][k][1] <= ymin + tmp:
                        tmp_points1.append(shapes[label]["points"][k])
                    elif shapes[label]["points"][k][1] > ymin + tmp and shapes[label]["points"][k][1] <= ymin + 2 * tmp:
                        tmp_points2.append(shapes[label]["points"][k])
                    elif shapes[label]["points"][k][1] > ymin + 2 * tmp and shapes[label]["points"][k][1] <= ymin + 3 * tmp:
                        tmp_points3.append(shapes[label]["points"][k])
                    else:
                        tmp_points4.append(shapes[label]["points"][k])
                
                 #第一个切片
                x1 = []
                y1 = []
                for ji in range(len(tmp_points1)):
                    x1.append(tmp_points1[ji][0])
                    y1.append(tmp_points1[ji][1])
                x1max = math.ceil(max(x1))
                x1min = int(min(x1))
                y1max = math.ceil(max(y1))
                y1min = int(min(y1))
                #width = x1max - x1min
                #height = y1max - y1min

                for ji in range(len(tmp_points1)):
                    tmp_points1[ji][0] = tmp_points1[ji][0] - x1min
                    tmp_points1[ji][1] = tmp_points1[ji][1] - y1min
                your_dict["imageHeight"] = y1max - y1min
                your_dict["imageWidth"] =  x1max - x1min
                if len(tmp_points1) >= 4:
                   shapes[label]["points"] = tmp_points1
                else:
                    shapes[label]["points"] = [[0, 0], [0, 1], [2, 2], [1,0]]
                #shapes[label]["points"] = tmp_points
                your_dict["shapes"] = [shapes[label]]
                your_dict["imagePath"] = os.path.basename(json_list[j])[:-5] +'_1.jpg'
                save_dir = save_data_dir + os.path.basename(json_list[j])[:-5] + "_1.json"
                with open(save_dir, "w", encoding = "UTF-8") as f1:
                    json.dump(your_dict, f1, ensure_ascii=False,indent = 2)
                print(os.path.basename(json_list[j])[:-5] + "_1.json已写入！！")
                
                #第二个切片
                x2 = []
                y2 = []
                for ji in range(len(tmp_points2)):
                    x2.append(tmp_points2[ji][0])
                    y2.append(tmp_points2[ji][1])
                x2max = math.ceil(max(x2))
                x2min = int(min(x2))
                y2max = math.ceil(max(y2))
                y2min = int(min(y2))
                #width = x1max - x1min
                #height = y1max - y1min

                for ji in range(len(tmp_points2)):
                    tmp_points2[ji][0] = tmp_points2[ji][0] - x2min
                    tmp_points2[ji][1] = tmp_points2[ji][1] - y2min
                your_dict["imageHeight"] = y2max - y2min
                your_dict["imageWidth"] =  x2max - x2min
                if len(tmp_points2) >= 4:
                   shapes[label]["points"] = tmp_points2
                else:
                    shapes[label]["points"] = [[0, 0], [0, 1], [2, 2], [1,0]]
                #shapes[label]["points"] = tmp_points
                your_dict["shapes"] = [shapes[label]]
                your_dict["imagePath"] = os.path.basename(json_list[j])[:-5] +'_2.jpg'
                save_dir = save_data_dir + os.path.basename(json_list[j])[:-5] + "_2.json"
                with open(save_dir, "w", encoding = "UTF-8") as f1:
                    json.dump(your_dict, f1, ensure_ascii=False,indent = 2)
                print(os.path.basename(json_list[j])[:-5] + "_2.json已写入！！")

                #第三个切片
                x3 = []
                y3 = []
                for ji in range(len(tmp_points3)):
                    x3.append(tmp_points3[ji][0])
                    y3.append(tmp_points3[ji][1])
                x3max = math.ceil(max(x3))
                x3min = int(min(x3))
                y3max = math.ceil(max(y3))
                y3min = int(min(y3))
                #width = x1max - x1min
                #height = y1max - y1min

                for ji in range(len(tmp_points3)):
                    tmp_points3[ji][0] = tmp_points3[ji][0] - x3min
                    tmp_points3[ji][1] = tmp_points3[ji][1] - y3min
                your_dict["imageHeight"] = y3max - y3min
                your_dict["imageWidth"] =  x3max - x3min
                if len(tmp_points3) >= 4:
                   shapes[label]["points"] = tmp_points3
                else:
                    shapes[label]["points"] = [[0, 0], [0, 1], [2, 2], [1,0]]
                #shapes[label]["points"] = tmp_points
                your_dict["shapes"] = [shapes[label]]
                your_dict["imagePath"] = os.path.basename(json_list[j])[:-5] +'_3.jpg'
                save_dir = save_data_dir + os.path.basename(json_list[j])[:-5] + "_3.json"
                with open(save_dir, "w", encoding = "UTF-8") as f1:
                    json.dump(your_dict, f1, ensure_ascii=False,indent = 2)
                print(os.path.basename(json_list[j])[:-5] + "_3.json已写入！！")
                
                #第四个切片
                x4 = []
                y4 = []
                for ji in range(len(tmp_points4)):
                    x4.append(tmp_points4[ji][0])
                    y4.append(tmp_points4[ji][1])
                x4max = math.ceil(max(x4))
                x4min = int(min(x4))
                y4max = math.ceil(max(y4))
                y4min = int(min(y4))
                #width = x1max - x1min
                #height = y1max - y1min

                for ji in range(len(tmp_points4)):
                    tmp_points4[ji][0] = tmp_points4[ji][0] - x4min
                    tmp_points4[ji][1] = tmp_points4[ji][1] - y4min

                your_dict["imageHeight"] = y4max - y4min
                your_dict["imageWidth"] =  x4max - x4min
                if len(tmp_points4) >= 4:
                   shapes[label]["points"] = tmp_points4
                else:
                    shapes[label]["points"] = [[0, 0], [0, 1], [2, 2], [1,0]]
                #shapes[label]["points"] = tmp_points
                your_dict["shapes"] = [shapes[label]]
                your_dict["imagePath"] = os.path.basename(json_list[j])[:-5] +'_4.jpg'
                save_dir = save_data_dir + os.path.basename(json_list[j])[:-5] + "_4.json"
                with open(save_dir, "w", encoding = "UTF-8") as f1:
                    json.dump(your_dict, f1, ensure_ascii=False,indent = 2)
                print(os.path.basename(json_list[j])[:-5] + "_4.json已写入！！")   
                


        '''
            your_dict["imageHeight"] = ymax - ymin
            your_dict["imageWidth"] =  xmax - xmin
            #shapes[label]["label"] = shapes[label]["label"][:-1]
            for k in range(len(shapes[label]["points"])):
                shapes[label]["points"][k][0] = shapes[label]["points"][k][0] - xmin
                shapes[label]["points"][k][1] = shapes[label]["points"][k][1] - ymin

            your_dict["shapes"] = [shapes[label]]
            your_dict["imagePath"] = os.path.basename(json_list[j])[:-5] + shapes[label]["label"] +'.jpg'
            save_dir = save_data_dir + os.path.basename(json_list[j])[:-5] + shapes[label]["label"] + ".json"
            with open(save_dir, "w", encoding = "UTF-8") as f1:
                json.dump(your_dict, f1, ensure_ascii=False,indent = 2)   '''
    print("Finish processing JSONs!!!")

def main():
    crop_ImgsAndJSON(load_data_dir, 'json_res7')
    #crop_jsons(load_data_dir,save_data_dir)

if __name__ == "__main__":
    load_data_dir='json_res1' #原始图片和json文件的路径
    #save_data_dir='E://科研/实验/标注数据/testjson/json_res7/' #裁剪后图片和json文件保存的路径
    main()
    print("finished all the program!!!")
