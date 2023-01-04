import numpy as np
from PIL import Image
import numpy as np
import habitat_sim
from habitat_sim.utils.common import d3_40_colors_rgb
import cv2
import json
import math

point = np.load('path.npy')
print(point)


def driver(start,end):
    #part1 rotate
    x = start[1]-start[0]
    y = end[1]-end[0]
    #相對y軸轉atan(x/y)
    goal_ry = atan(x/y)
    if(goal_ry>=0):
        action = "turn_left"
    else:
        action = "turn_right"
    for i in range(int(goal_ry*10)):
        navigateAndSee(action)
    #part2 goforword
    action = "move_forward"
    while(1):
        agent_state = agent.get_state()
        x = sensor_state.sensor_state.position[0]
        z = sensor_state.sensor_state.position[2]
        distance = (x-end[0])**2+(z-end[1])**2
        distance = math.sqrt(distance)
        if(distance<0.01):
            break


