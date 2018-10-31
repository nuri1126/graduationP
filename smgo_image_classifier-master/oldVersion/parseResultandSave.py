import re
import os
import shutil

# label name
# 0 faces
# 1 food
# 2 nature
# 3 pets

def main():
    file = open('/home/sm/result.txt', 'r')
    lines = file.readlines()
    resultDic = {}
    result = 0
    filename = ""
    save = False
    for line in lines:
        item = line.split(" ")
        if "#File" in line:
            filename = item[item.index("#File")+3]
            save = True
        if "Result" in line:
            i = int(re.findall('\d+', item[2])[0])
            result = i
            save = True
        if(save):
            resultDic[filename] = result
            save = False

    # Print result to terminal
    # print('[Predictions for %d images]' %resultDic.__len__())
    # for item in resultDic:
    #     print(item,'is',resultDic[item])

    # Classify and Save
    dataset_dir = '/home/sm/PycharmProjects/smproject/smgo/testimages/images'
    dirname_faces = os.path.join(dataset_dir, 'faces')
    dirname_food = os.path.join(dataset_dir, 'food')
    dirname_nature = os.path.join(dataset_dir, 'nature')
    dirname_pets = os.path.join(dataset_dir, 'pets')
    if not os.path.isdir(dirname_faces):
        os.mkdir(dirname_faces)
    if not os.path.isdir(dirname_food):
        os.mkdir(dirname_food)
    if not os.path.isdir(dirname_nature):
        os.mkdir(dirname_nature)
    if not os.path.isdir(dirname_pets):
        os.mkdir(dirname_pets)
    print('>> Predictions for [%d] images' % resultDic.__len__())
    count = 1
    for item in resultDic:
        filepath = os.path.join(dataset_dir, item)
        index = resultDic[item]
        if(index == 0):
            print(count,'-', item, 'is [faces]')
            shutil.copy(filepath, dirname_faces)
        elif(index == 1):
            print(count,'-',item, 'is [food]')
            shutil.copy(filepath, dirname_food)
        elif (index == 2):
            print(count,'-',item, 'is [nature]')
            shutil.copy(filepath, dirname_nature)
        else:
            print(count,'-',item, 'is [pets]')
            shutil.copy(filepath, dirname_pets)
        count+=1
    print('>> Complete!')

if __name__ == '__main__':
    main()