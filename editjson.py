import os, json
import numpy as np

#读取json文件
def read_json(json_path):
    f = open(json_path)
    json_file = json.load(f)
    return json_file

#保存json文件
def save_json(out_path, json_file):
    f = open(out_path, 'w', encoding='utf-8')
    json.dump(json_file, f)

#按照实例分割要求编辑json文件
def edit_labelme_json(labelme_json):
    #定义变量
    version = labelme_json['version']
    flags = labelme_json['flags']
    shapes = labelme_json['shapes']
    imagePath = labelme_json['imagePath']
    imageData = labelme_json['imageData']
    imageHeight = labelme_json['imageHeight']
    imageWidth = labelme_json['imageWidth']

    ###################编辑####################
    label_list = []
    label_dic = {}

    new_shapes = list([])
    for tmp_shape in shapes:
        for shape_key,shape_values in tmp_shape.items():
            if shape_key == 'label':
                if shape_values not in label_list:
                    label_list.append(shape_values)
                    label_dic.update({shape_values:0})
                    shape_values = shape_values + '0'
                else:
                    label_dic.update({shape_values:label_dic[shape_values] + 1})
                    shape_values = shape_values+str(label_dic[shape_values])
                label_values = shape_values
            elif shape_key == 'points':points_values = shape_values
            elif shape_key == 'group_id':group_id_values = shape_values
            elif shape_key == 'shape_type':shape_type_values = shape_values
            elif shape_key == 'flags':flags_values = shape_values
        new_shape = {'label':label_values, 'points':points_values, 'group_id':group_id_values, 'shape_type':shape_type_values, 'flags':flags_values}
        new_shapes.append(new_shape)
    new_json = {'version':version,
                'flags':flags,
                'shapes':new_shapes,
                'imagePath':imagePath,
                'imageData':imageData,
                'imageHeight':imageHeight,
                'imageWidth':imageWidth}
    return new_json

#列出文件夹中的json文件名
def list_json(path):
    sup_ext = ['.json']
    all_list = list(map(lambda x:os.path.join(path, x), os.listdir(path)))
    json_list = [x for x in all_list if os.path.splitext(x)[1]in sup_ext]
    return json_list

def get_outpath(out_dir, name):
    if not os.path.exists(out_dir):os.makedirs(out_dir)
    return os.path.join(out_dir, os.path.basename(name))

if __name__ == '__main__':
    json_dir = 'json_ori'
    out_dir = 'json_res'
    json_list = list_json(json_dir)
    for i in range(len(json_list)):
        json_r = read_json(json_list[i])
        json_ed = edit_labelme_json(json_r)
        outpath = get_outpath(out_dir, json_list[i])
        save_json(outpath, json_ed)
