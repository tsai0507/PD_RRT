import numpy as np
from PIL import Image
import numpy as np
import habitat_sim
from habitat_sim.utils.common import d3_40_colors_rgb
import cv2
import json
import math
from mit_semseg.utils import colorEncode
from scipy.io import loadmat
import argparse

# This is the scene we are going to load.
# support a variety of mesh formats, such as .glb, .gltf, .obj, .ply
### put your scene path ###
test_scene = "apartment_0/habitat/mesh_semantic.ply"
path = "apartment_0/habitat/info_semantic.json"
colors = loadmat('color101.mat')['colors']
colors = np.insert(colors, 0, values=np.array([[0,0,0]]), axis=0)

#global test_pic
#### instance id to semantic id 
with open(path, "r") as f:
    annotations = json.load(f)

id_to_label = []
instance_id_to_semantic_label_id = np.array(annotations["id_to_label"])
for i in instance_id_to_semantic_label_id:
    if i < 0:
        id_to_label.append(0)
    else:
        id_to_label.append(i)
id_to_label = np.asarray(id_to_label)

######

sim_settings = {
    "scene": test_scene,  # Scene path
    "default_agent": 0,  # Index of the default agent
    "sensor_height": 1.5,  # Height of sensors in meters, relative to the agent
    "width": 512,  # Spatial resolution of the observations
    "height": 512,
    "sensor_pitch": 0,  # sensor pitch (x rotation in rads)
}

# This function generates a config for the simulator.
# It contains two parts:
# one for the simulator backend
# one for the agent, where you can attach a bunch of sensors

def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]

def transform_depth(image):
    depth_img = (image / 10 * 255).astype(np.uint8)
    return depth_img

def transform_semantic(semantic_obs):
    semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
    semantic_img.putpalette(colors.flatten())
    semantic_img.putdata(semantic_obs.flatten().astype(np.uint8))
    semantic_img = semantic_img.convert("RGB")
    semantic_img = cv2.cvtColor(np.asarray(semantic_img), cv2.COLOR_RGB2BGR)
    return semantic_img

def make_simple_cfg(settings):
    # simulator backend
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = settings["scene"]
    

    # In the 1st example, we attach only one sensor,
    # a RGB visual sensor, to the agent
    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "color_sensor"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [settings["height"], settings["width"]]
    rgb_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    rgb_sensor_spec.orientation = [
        settings["sensor_pitch"],
        0.0,
        0.0,
    ]
    rgb_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    #depth snesor
    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [settings["height"], settings["width"]]
    depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    depth_sensor_spec.orientation = [
        settings["sensor_pitch"],
        0.0,
        0.0,
    ]
    depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    #semantic snesor
    semantic_sensor_spec = habitat_sim.CameraSensorSpec()
    semantic_sensor_spec.uuid = "semantic_sensor"
    semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_sensor_spec.resolution = [settings["height"], settings["width"]]
    semantic_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    semantic_sensor_spec.orientation = [
        settings["sensor_pitch"],
        0.0,
        0.0,
    ]
    semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    # agent
    agent_cfg = habitat_sim.agent.AgentConfiguration()

    agent_cfg.sensor_specifications = [rgb_sensor_spec, depth_sensor_spec, semantic_sensor_spec]
    ##################################################################
    ### change the move_forward length or rotate angle
    ##################################################################
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.01) # 0.01 means 0.01 m
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=1.0) # 1.0 means 1 degree
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=1.0)
        ),
    }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


cfg = make_simple_cfg(sim_settings)
sim = habitat_sim.Simulator(cfg)


# initialize an agent
agent = sim.initialize_agent(sim_settings["default_agent"])

# Set agent state
point = np.load('path.npy') #load the path
start = point[0]
print(start)
agent_state = habitat_sim.AgentState()
agent_state.position = np.array([start[1], 0.0, start[0]])  # agent in world space
agent.set_state(agent_state)

# obtain the default, discrete actions that an agent can perform
# default action space contains 3 actions: move_forward, turn_left, and turn_right
action_names = list(cfg.agents[sim_settings["default_agent"]].action_space.keys())
print("Discrete action space: ", action_names)



def navigateAndSee(action=""):
    if action in action_names:
        observations = sim.step(action)
        #print("action: ", action)

        RGB_img = transform_rgb_bgr(observations["color_sensor"])
        SEIMEN_img =  transform_semantic(id_to_label[observations["semantic_sensor"]])
        index = np.where((SEIMEN_img[:,:,0]==b)*(SEIMEN_img[:,:,1]==g)*(SEIMEN_img[:,:,2]==r))
        if len(index[0]) != 0:
            RGB_img[index] = cv2.addWeighted(RGB_img[index], 0.6, SEIMEN_img[index], 0.4, 50)
        cv2.imshow("RGB", RGB_img)
        cv2.waitKey(1)
        videowriter.write(RGB_img)
        agent_state = agent.get_state()
        sensor_state = agent_state.sensor_states['color_sensor']
        print("camera pose: x y z rw rx ry rz")
        print(sensor_state.position[0],sensor_state.position[1],sensor_state.position[2],  sensor_state.rotation.w, sensor_state.rotation.x, sensor_state.rotation.y, sensor_state.rotation.z)
        return sensor_state


def driver(pre_node,start,end):
    #part1 rotate
    print("ok")
    if pre_node == []:
        v1 =np.array([-1,0])
    else:
        v1 = np.array([start[0]-pre_node[0],start[1]-pre_node[1]])
    v2 = np.array([end[0]-start[0],end[1]-start[1]])
    print(v1,v2)
    flag = v1[0]*v2[1]-v1[1]*v2[0]
    value = v1@v2
    v1 = math.sqrt((v1[0])**2+(v1[1])**2)
    v2 = math.sqrt((v2[0])**2+(v2[1])**2)

    goal_ry = int(math.acos(value/(v1*v2))/math.pi*180)
    print(goal_ry)
    print("rotate number",int(goal_ry))
    if(flag>=0):
        action = "turn_left"
    else:
        action = "turn_right"
    for i in range(int(abs(goal_ry))):
        sensor_state = navigateAndSee(action)
    #part2 goforword
    action = "move_forward"
    forward_distance = math.sqrt((end[0]-start[0])**2+(end[1]-start[1])**2)
    step = int(forward_distance/0.01)
    for i in range(step):
        sensor_state = navigateAndSee(action)
    x = sensor_state.position[0]
    z = sensor_state.position[2]
    return (z,x)



pre_node = []
start = point[0]
final = {"refrigerator":(255, 0, 0),"rack":(0, 255, 133),"cushion":(255, 9, 92),"lamp":(160, 150, 20),"cooktop":(7, 255, 224)}
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Below are the params:')
    parser.add_argument('-f', type=str, default=" ",metavar='END', action='store', dest='End',
                help='Where want to go')
    args = parser.parse_args()

    # save video initial
    path = "video/" + args.End + ".mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videowriter = cv2.VideoWriter(path, fourcc, 100, (512, 512))

    end_rgb = final[args.End]
    r = end_rgb[0]
    g = end_rgb[1]
    b = end_rgb[2]
    for i in range(len(point)-1):
        temp = start
        start = driver(pre_node,start,point[i+1])
        pre_node = temp
    videowriter.release()