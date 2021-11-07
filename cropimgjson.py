import json
import glob
import cv2
import os

load_data_dir = ''
save_data_dir = ''

def crop_imgs(load_data_dir, save_data_dir):
    for img in glob.glob(load_data_dir + '*.jpg'):
        print(img)
        jpg = cv2.imread(img)
        ###########获取X,Y坐标的最值
        x = []
        y = []
        f = img[:-4] + ".json"
        with open(j, "r") as f:
            json_str = f.read()
        your_dict = json.load(json_str)
        shapes = your_dict["shapes"]
        for label in range(len(shapes)):
            for k in range(len(shapes[label]["points"])):
                x.append(shapes[label]["points"][k][0])
                y.append(shapes[label]["points"][k][1])
            xmax = max(x)
            xmin = min(x)
            ymax = max(y)
            ymin = min(y)
            cut_jpg = jpg[ymin:ymax, xmin:xmax]
            save_data_dir=save_data_dir + os.path.basename(img)[:-4] + shapes[label]["label"] + ".jpg"
            cv2.imwrite(save_data_dir, cut_jpg, [100])





def crop_jsons(load_data_dir, save_data_dir):
    for j in glob.glob(load_data_dir + '*.json'):
        with open(j, "r") as f:
            json_str = f.read()
        your_dict = json.load(json_str)
        your_dict["imageData"] = None
        your_dict["imagePath"] = os.path.basename(j) + '.jpg'
        #your_dict["imageHeight"] = y1 - y0
        #your_dict["imageWidth"] = x1 - x0

        shapes = your_dict["shapes"]
        for label in range(len(shapes)):
            x = []
            y = []
            for i in range(len(shapes[label]["points"])):
                x.append(shapes[label]["points"][i][0])
                y.append(shapes[label]["points"][i][1])
            #xmax = max(x)
            xmin = min(x)
            #ymax = max(y)
            ymin = min(y)
            for k in range(len(shapes[label]["points"])):
                shapes[label]["points"][k][0] = shapes[label]["points"][k][0] - xmin
                shapes[label]["points"][k][1] = shapes[label]["points"][k][1] - ymin
            save_dir = save_data_dir + os.path.basename(j)[:-4] + shapes[label]["label"] + ".json"
            with open(save_dir, "w", encoding = "UTF-8") as f1:
                json.dump(your_dict, f1, ensure_ascii=False,indent = 2)

if __name__ == '__main__':
    json_dir = 'json_ori'
    out_dir = 'json_res'
    crop_imgs(json_dir, out_dir)
    crop_jsons(json_dir, out_dir)






