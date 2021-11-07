import json
import glob
import cv2
import os
import math




#列出文件夹中的json文件名
def list_json(path):
    sup_ext = ['.json']
    all_list = list(map(lambda x:os.path.join(path, x), os.listdir(path)))
    json_list = [x for x in all_list if os.path.splitext(x)[1]in sup_ext]
    return json_list


#列出文件夹中的json文件名
def list_jpg(path):
    sup_ext = ['.jpg']
    all_list = list(map(lambda x:os.path.join(path, x), os.listdir(path)))
    json_list = [x for x in all_list if os.path.splitext(x)[1]in sup_ext]
    return json_list

#获取输出路径
def get_outpath(out_dir, name):
    if not os.path.exists(out_dir):os.makedirs(out_dir)
    return os.path.join(out_dir, os.path.basename(name))


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
            outpath = get_outpath(save_data_dir, jpg_list[img])[:-4] + shapes[label]["label"] + ".jpg"
            cv2.imwrite(outpath, cut_jpg)#, [100])
            print("已写入！"+ str(ymin) + " " + str(ymax)  + " " + str(xmin) + " " +str(xmax))
    print("Finish processing images!!!")





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
            x = []
            y = []
            for i in range(len(shapes[label]["points"])):
                x.append(shapes[label]["points"][i][0])
                y.append(shapes[label]["points"][i][1])
            xmax = math.ceil(max(x))
            xmin = int(min(x))
            ymax = math.ceil(max(y))
            ymin = int(min(y))


            your_dict["imageHeight"] = ymax - ymin
            your_dict["imageWidth"] =  xmax - xmin
            for k in range(len(shapes[label]["points"])):
                shapes[label]["points"][k][0] = shapes[label]["points"][k][0] - xmin
                shapes[label]["points"][k][1] = shapes[label]["points"][k][1] - ymin

            your_dict["shapes"] = [shapes[label]]
            your_dict["imagePath"] = os.path.basename(json_list[j])[:-5] + shapes[label]["label"] +'.jpg'
            save_dir = save_data_dir + os.path.basename(json_list[j])[:-5] + shapes[label]["label"] + ".json"
            with open(save_dir, "w", encoding = "UTF-8") as f1:
                json.dump(your_dict, f1, ensure_ascii=False,indent = 2)
    print("Finish processing JSONs!!!")

def main():
    crop_imgs(load_data_dir, 'json_res')
    crop_jsons(load_data_dir,save_data_dir)

if __name__ == "__main__":
    load_data_dir='json_ori' #原始图片和json文件的路径
    save_data_dir='E://科研/实验/标注数据/testjson/json_res/' #裁剪后图片和json文件保存的路径
    main()
    print("finished all the program!!!")





