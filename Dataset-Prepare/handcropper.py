
# coding: utf-8

# In[103]:
import tf_pose
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil


# In[104]:


e = tf_pose.get_estimator()
w = 210
h = 260


# In[109]:


def get_hand(hand,w,h,frame):
    y = int(hand.y * h)
    x = int(hand.x * w)
    x_off = 0
    y_off = int(x_off * 1.42)
    if x < 46:
        x = 46
    if x > 210 - 46:
         x = 210-46
    if y < 66:
        y = 66
    if y > 260-66:
        y = 260 - 66
    f = frame[y-66:y+66,x-46:x+46,:]
    f = cv2.resize(f,(92,132))
    return f

def detect_and_crop(img):
    img1 = cv2.resize(cv2.imread(img),(210,260))
    img2 = img1
    humans = e.inference(img1,upsample_size=16.0)
    if len(humans) == 0:
        return False
    human= humans[0]
    body_parts = human.body_parts
    if 4 in body_parts or 7 in body_parts:
        if 4 in body_parts:
            right_hand = get_hand(body_parts[4],w,h,img1)
        else:
            right_hand = False
        if 7 in body_parts:
            left_hand = get_hand(body_parts[7],w,h,img2)
            left_hand = np.fliplr(left_hand)
        else:
            left_hand = False
        return right_hand,left_hand
    else:
        return False
        

    
def get_dir(path):
    dir_list = []
    for d in sorted(os.listdir(path)):
        if os.path.isdir(path + d): 
            dir_list.append(path + d + '/')
    return dir_list
def get_img(path):
    img_list = []
    for d in sorted(os.listdir(path)):
        if d.endswith('.png'):
            img_list.append(path + d)
    return img_list

def create_hands(path,target):
    dirs = get_dir(path)
    for d in dirs:
        img_list = get_img(d)
        os.mkdir(target+d)
        os.mkdir(target+ d+'right/')
        os.mkdir(target+ d+ 'left/')
        for img in img_list:
            hands = detect_and_crop(img)
            if hands is False:
                continue
            if hands[0] is not False:
                cv2.imwrite(target+d+'right/'+os.path.split(img)[-1],hands[0])
            if hands[1] is not False:
                cv2.imwrite(target+d+'left/'+os.path.split(img)[-1],hands[1])


# In[110]:

try:
    shutil.rmtree('/home/pilab/Desktop/train')
    os.mkdir('/home/pilab/Desktop/train')
except:
    os.mkdir('/home/pilab/Desktop/train')
    
create_hands('train/','/home/pilab/Desktop/')

try:
    shutil.rmtree('/home/pilab/Desktop/dev')
    os.mkdir('/home/pilab/Desktop/dev')
except:
    os.mkdir('/home/pilab/Desktop/dev')
    
create_hands('dev/','/home/pilab/Desktop/')

try:
    shutil.rmtree('/home/pilab/Desktop/test')
    os.mkdir('/home/pilab/Desktop/test')
except:
    os.mkdir('/home/pilab/Desktop/test')
    
create_hands('test/','/home/pilab/Desktop/')