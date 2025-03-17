import time
import pyautogui
import pyscreeze
import cv2


class find:
    def __init__(self,picture_No = 0):
        self.screenScale=1
        #事先读取按钮截图
        self.target= cv2.imread(f"./picture/{picture_No}.png",cv2.IMREAD_GRAYSCALE)
        self.theight, self.twidth = self.target.shape[:2]
    def run(self,):
        # 先截图
        screenshot=pyscreeze.screenshot('my_screenshot.png')
        # 读取图片 灰色会快
        temp = cv2.imread(r'my_screenshot.png',cv2.IMREAD_GRAYSCALE)
        tempheight, tempwidth = temp.shape[:2]
        print("目标图宽高："+str(self.twidth)+"-"+str(self.theight))
        print("模板图宽高："+str(tempwidth)+"-"+str(tempheight))
        # 先缩放屏幕截图 INTER_LINEAR INTER_AREA
        scaleTemp=cv2.resize(temp, (int(tempwidth / self.screenScale), int(tempheight / self.screenScale)))
        stempheight, stempwidth = scaleTemp.shape[:2]
        print("缩放后模板图宽高："+str(stempwidth)+"-"+str(stempheight))
        # 匹配图片
        res = cv2.matchTemplate(scaleTemp, self.target, cv2.TM_CCOEFF_NORMED)
        mn_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if(max_val>=0.9):
            # 计算出中心点
            top_left = max_loc
            bottom_right = (top_left[0] + self.twidth, top_left[1] + self.theight)
            tagHalfW=int(self.twidth/2)
            tagHalfH=int(self.theight/2)
            tagCenterX=top_left[0]+tagHalfW
            tagCenterY=top_left[1]+tagHalfH
            #左键点击屏幕上的这个位置
            pyautogui.moveTo(tagCenterX, tagCenterY)
            return tagCenterX,tagCenterY


        else:
            print ("没找到")

