import os
import random
p_path = os.path.join('path to visda','train')
dir_list = os.listdir(p_path)
print(dir_list)

class_list = ["bicycle", "bus", "car", "motorcycle", "train", "truck", "unk"]
path_source = "./source_list.txt"
path_target = "./target_list.txt"
write_source = open(path_source,"w")
write_target = open(path_target,"w")
for k, direc in enumerate(dir_list):
    if not '.txt' in direc:
        files = os.listdir(os.path.join(p_path, direc))
        for i, file in enumerate(files):
            if direc in class_list:
                class_name = direc
                file_name = os.path.join(p_path, direc, file)
                write_source.write('%s %s\n' % (file_name, class_list.index(class_name)))
            else:
                continue
p_path = os.path.join('path to visda','validation')
dir_list = os.listdir(p_path)
print(dir_list)
for k, direc in enumerate(dir_list):
    if not '.txt' in direc:
        files = os.listdir(os.path.join(p_path, direc))
        for i, file in enumerate(files):
            if direc in class_list:
                class_name = direc
            else:
                class_name = "unk"
            file_name = os.path.join(p_path, direc, file)
            write_target.write('%s %s\n' % (file_name, class_list.index(class_name)))

