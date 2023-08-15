import sys
import time
import argparse

sys.path.append('.')

from src.environments.dogfightEnv.dogfight_sandbox_hg2.network_client_example import dogfight_client as df

def parse_args():
    parser = argparse.ArgumentParser(description='TBD')
    parser.add_argument('--host', default='10.184.0.0', help='specifies Harfang host id')
    parser.add_argument('--port', default='50888', help='specifies Harfang port id')
    args = parser.parse_args()
    return args

args = parse_args()

df.connect(args.host, int(args.port))

planes = df.get_planes_list()

for i in planes:
    df.reset_machine(i)

df.set_plane_thrust(planes[1], 1)

df.set_client_update_mode(True)

# Thrust
t = 0
while t < 1:
    time.sleep(1/60)
    plane_state = df.get_plane_state(planes[1])
    df.update_scene()
    t = plane_state["thrust_level"]

df.activate_post_combustion(planes[1])

df.set_plane_pitch(planes[1], -0.5)

# Pitch
p = 0
while p < 15:
    time.sleep(1/60)
    plane_state = df.get_plane_state(planes[1])
    df.update_scene()
    p = plane_state["pitch_attitude"]

df.stabilize_plane(planes[1])

df.retract_gear(planes[1])

s = 0
while s < 100:
    time.sleep(1/60)
    plane_state = df.get_plane_state(planes[1])
    df.update_scene()
    s = plane_state["altitude"]

df.set_plane_pitch(planes[1], 0.1)

p = 15
while p > 0:
    time.sleep(1/60)
    plane_state = df.get_plane_state(planes[1])
    df.update_scene()
    p = plane_state["pitch_attitude"]

df.stabilize_plane(planes[1])

while True:
    time.sleep(1/60)
    df.update_scene()
    # Print anything you want to see here while controlling the plane with your keyboard
    # print('')
