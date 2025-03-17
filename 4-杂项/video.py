import os

path = r'D:\SteamLibrary\steamapps\workshop\content\431960'  # 要遍历的目录
for root, dirs, names in os.walk(path):
    for name in names:
        oldname = os.path.join(root, name)
        name = name.replace(' ', '_')
        newname = os.path.join(root, name)
        os.rename(oldname, newname)
        ext = os.path.splitext(name)[1]  # 获取后缀名
        print(newname)
        print(ext)

        if ext == '.mp4':
           moveto = os.path.join(r'D:\av',name)  ##dirname 上一层目录
           os.rename(newname, moveto)  # 移动文件
        # if  ext == '.010' or ext == '.001' or ext == '.002' or ext == '.003' or ext == '.004'or ext == '.005' or ext == '.006'or ext == '.007' or ext == '.008' or ext == '.009' :
        #    moveto = os.path.join(r'D:\av\game',name)  ##dirname 上一层目录
        #    os.rename(newname, moveto)  # 移动文件


