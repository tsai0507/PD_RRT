import cv2
import numpy as np
import math
import random
import argparse
import os
from sklearn.neighbors import NearestNeighbors

class Nodes:
    """Class to store the RRT graph"""
    def __init__(self, x,y):
        self.x = x
        self.y = y
        self.parent_x = []
        self.parent_y = []

# return dist and angle b/w new point and nearest node
def dist_and_angle(x1,y1,x2,y2):
    dist = math.sqrt( ((x1-x2)**2)+((y1-y2)**2) )
    angle = math.atan2(y2-y1, x2-x1)
    return(dist,angle)

# return the neaerst node index
def nearest_node(x,y,node_list):
    temp_dist=[]
    for i in range(len(node_list)):
        dist,_ = dist_and_angle(x,y,node_list[i].x,node_list[i].y)
        temp_dist.append(dist)
    return temp_dist.index(min(temp_dist))


# generate a random point in the image space
def rnd_point(h,l):
    new_y = random.randint(0, h)
    new_x = random.randint(0, l)
    return (new_x,new_y)

# check collision
def collision(x1,y1,x2,y2,img):
    color=[]
    print(x1,x2)
    if((x2-x1)!=0):
        x = list(np.arange(x1,x2,(x2-x1)/100))
        y = list(((y2-y1)/(x2-x1))*(x-x1) + y1)
        # print("collision",x,y)
        for i in range(len(x)):
            # print(int(x[i]),int(y[i]))
            color.append(img[int(y[i]),int(x[i])])
        #若兩點之間有黑色代表有碰撞
        if (0 in color):
            return True #collision
        else:
            return False #no-collision
    else:
        print("ok")
        x = x1
        y = list(np.arange(y1,y2,(y2-y1)/100))
        # print("collision",x,y)
        for i in range(len(y)):
            # print(int(x[i]),int(y[i]))
            color.append(img[int(y[i]),int(x)])
        #若兩點之間有黑色代表有碰撞
        if (0 in color):
            return True #collision
        else:
            return False #no-collision

# check the collision with obstacle and trim
def extrend1(x1,y1,x2,y2,stepSize,img):
    
    _,theta = dist_and_angle(x2,y2,x1,y1)
    x=x2 + stepSize*np.cos(theta)
    y=y2 + stepSize*np.sin(theta)

    hy,hx=img.shape
    #先檢查有沒有超過邊界
    if y<0 or y>hy or x<0 or x>hx:
        # print("Point out of image bound")
        directCon = False
        nodeCon = False
    else:
        # print("p",x,y)
        # print("p",x2,y2)
        # check connection between two nodes
        if collision(x,y,x2,y2,img):
            nodeCon = False
        else:
            nodeCon = True

    return(x,y,nodeCon)

#找到a,b兩團點雲中最近的點
def extrend2(node_list_a,node_list_b,img):

    x1 = node_list_a[-1].x
    x2 = node_list_a[-1].y
    nearest_ind_b = nearest_node(x1,x2,node_list_b)
    nearest_x = node_list_b[nearest_ind_b].x
    nearest_y = node_list_b[nearest_ind_b].y

    # check direct connection
    if collision(x1,x2,nearest_x,nearest_y,img):
        directCon = False
    else:
        directCon=True

    # return(directCon,nearest_ind_a,nearest_ind_b)
    return(directCon,nearest_ind_b)



class RRTree:

    def __init__(self, stepsize, map_image,iter_num):
        img = cv2.imread(map_image,0) # load grayscale maze image
        img[np.where((img[:,:] != 255))] = 0
        kernel = np.ones((3,3), np.uint8)
        dilation = cv2.erode(img, kernel, iterations = 12)
        # cv2.imshow('dilation',dilation)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        self.img = dilation # load grayscale maze image
        self.img2 = cv2.imread(map_image) # load colored maze image 
        self.K = iter_num
        self.start = 0
        self.end = 0
        self.stepsize= stepsize
        self.node_list_start = [0]
        self.node_list_end = [0]
        self.end_final = " "

    def choice_final(self,end_final):
        # final = {"refrigerator":(255, 0, 0),"rack":(0, 255, 133),"cushion":(255, 9, 92),"lamp":(160, 150, 20),"cooktop":(7, 255, 224)}
        # r = final[end_final][0]
        # g = final[end_final][1]
        # b = final[end_final][2]
        # index = np.where((self.img2[:,:,0] == b)*(self.img2[:,:,1] == g)*(self.img2[:,:,2] == r))
        # x = int(np.mean(index[0]))
        # y = int(np.mean(index[1]))

        # h = [x,y+1]
        # d = [x, y-1]
        # l = [x-1,y]
        # r = [x+1,y]
        # # print(x,y)
        # while(1):
        #     k = 1
        #     h[1] = h[1]+k
        #     d[1] = d[1]-k
        #     l[0] = l[0]-k
        #     r[0] = r[0]+k
        #     if self.img[d[0],d[1]] == 255:
        #         x,y = d
        #         break
        #     elif self.img[h[0],h[1]] == 255:
        #         x,y = h
        #         break
        #     elif self.img[l[0],l[1]] == 255:
        #         x,y = l
        #         break
        #     elif self.img[r[0],r[1]] == 255:
        #         x,y = r
        #         break
        final = {"refrigerator":(453,458),"rack":(749,282),"cushion":(1041,452),"lamp":(892,655),"cooktop":(378,543)}
        x = final[end_final][0]
        y = final[end_final][1]
        self.end_final = end_final
        # cv2.circle(self.img2, (int(x),int(y)), 2,(0,0,255),thickness=3, lineType=8)
        # cv2.imshow('ddd',self.img2)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # self.end = (y,x)
        self.end = (x,y)

    def compute(self,end_final):
        if(end_final!=" "):
            self.choice_final(end_final)
            print("find object")
        h,l= self.img.shape # dim of the loaded image

        # insert the starting point in the node class       
        self.node_list_start[0] = Nodes(self.start[0],self.start[1])
        self.node_list_start[0].parent_x.append(self.start[0])
        self.node_list_start[0].parent_y.append(self.start[1])
        self.node_list_end[0] = Nodes(self.end[0],self.end[1])
        self.node_list_end[0].parent_x.append(self.end[0])
        self.node_list_end[0].parent_y.append(self.end[1])

        # display start and endcheck_collision(
        #用來判斷tatb
        flag = -1
        i = 1
        while i<self.K:
            if flag == -1:
                Ta = self.node_list_start.copy()
                Tb = self.node_list_end.copy()
            else:
                Ta = self.node_list_end.copy()
                Tb = self.node_list_start.copy()
            #在地圖中的隨機點
            nx,ny = rnd_point(h,l)
            #if extend1 trap
            nearest_ind = nearest_node(nx,ny,Ta)
            nearest_x = Ta[nearest_ind].x
            nearest_y = Ta[nearest_ind].y
            tx,ty,nodeCon = extrend1(nx,ny,nearest_x,nearest_y,self.stepsize,self.img)
            if nodeCon:
                ##把點加入Ta
                Ta.append(i)
                Ta[i] = Nodes(tx,ty)
                Ta[i].parent_x = Ta[nearest_ind].parent_x.copy()
                Ta[i].parent_y = Ta[nearest_ind].parent_y.copy()
                #parent裡面就是start到自己的路徑
                Ta[i].parent_x.append(tx)
                Ta[i].parent_y.append(ty)
                # display
                cv2.circle(self.img2, (int(tx),int(ty)), 2,(0,0,255),thickness=3, lineType=8)
                cv2.line(self.img2, (int(tx),int(ty)), (int(Ta[nearest_ind].x),int(Ta[nearest_ind].y)), (0,255,0), thickness=1, lineType=8)
                cv2.imwrite("media/"+str(i)+".jpg",self.img2)
                cv2.imshow("image",self.img2)
                cv2.waitKey(1)
                #if extend2 connect
                directCon,index = extrend2(Ta,Tb,self.img)
                if directCon :
                    print("Path has been found")
                    path = []
                    cv2.line(self.img2, (int(tx),int(ty)), (int(Tb[index].x),int(Tb[index].y)), (0,255,0), thickness=1, lineType=8)
                    if flag == -1:
                        for i in range(len(Ta[-1].parent_x)):
                            path.append((Ta[-1].parent_x[i],Ta[-1].parent_y[i]))
                            pass
                        Tb[index].parent_x.reverse()
                        Tb[index].parent_y.reverse()    
                        for i in range(len(Tb[index].parent_x)):
                            path.append((Tb[index].parent_x[i],Tb[index].parent_y[i]))
                    else:
                        for i in range(len(Tb[index].parent_x)):
                            path.append((Tb[index].parent_x[i],Tb[index].parent_y[i]))
                            pass
                        Ta[-1].parent_x.reverse()
                        Ta[-1].parent_y.reverse()
                        for i in range(len(Ta[-1].parent_x)):
                            path.append((Ta[-1].parent_x[i],Ta[-1].parent_y[i]))
                    for i in range(len(path)-1):
                        cv2.line(self.img2, (int(path[i][0]),int(path[i][1])), (int(path[i+1][0]),int(path[i+1][1])), (255,0,0), thickness=2, lineType=8)
                    cv2.waitKey(1)
                    cv2.imwrite("media/"+str(i)+".jpg",self.img2)
                    if(self.end_final == " "):
                        cv2.imwrite("birrt.jpg",self.img2)
                    else:
                        cv2.imwrite("path/"+self.end_final+".jpg",self.img2)
                    break
            else:
                continue
            if flag == -1:
                flag = flag*(-1)
                self.node_list_start = Ta.copy()
                self.node_list_end = Tb.copy()
            else:
                flag = flag*(-1)
                i = i+1
                self.node_list_start = Tb.copy()
                self.node_list_end = Ta.copy()
        if i==self.K:
            print("Can not find the path")
        print("number of iter: ",i*2)
        path = np.asarray(path)
        tran1 = l/17
        tran2 = h/11
        path[:,0] = path[:,0]/tran1-6
        path[:,1] = 7-path[:,1]/tran2
        return path

def draw_circle(event,x,y,flags,param):
    global coordinates
    if event == cv2.EVENT_LBUTTONDBLCLK:
    # if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(rrt_test.img2,(x,y),5,(255,0,0),-1)
        coordinates.append(x)
        coordinates.append(y)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Below are the params:')
    parser.add_argument('-p', type=str, default='mapp.png',metavar='ImagePath', action='store', dest='imagePath',
                    help='Path of the image containing mazes')
    parser.add_argument('-s', type=int, default=40,metavar='Stepsize', action='store', dest='stepSize',
                    help='Step-size to be used for RRT branches')
    parser.add_argument('-f', type=str, default=" ",metavar='END', action='store', dest='End',
                    help='Where want to go')
    args = parser.parse_args()

    # remove previously stored data
    try:
      os.system("rm -rf media")
    except:
      print("Dir already clean")
    os.mkdir("media")
    end = args.End
    rrt_test = RRTree(args.stepSize, args.imagePath,10000)

    coordinates=[]
    print("Select start and end points by double clicking, press 'escape' to exit")
    cv2.namedWindow('image')
    cv2.setMouseCallback('image',draw_circle)
    while(1):
        cv2.imshow('image',rrt_test.img2)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break
    rrt_test.start=(coordinates[0],coordinates[1])
    rrt_test.end=(coordinates[2],coordinates[3])
    path = rrt_test.compute(end)
    # print(path)
    np.save('path', path)

        
    