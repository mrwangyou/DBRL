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
    df.display_2DText([0.4, 0.7], "Thrust", 0.1, [1, 0.5, 0, 1])
    time.sleep(1/60)
    plane_state = df.get_plane_state(planes[1])
    df.update_scene()
    t = plane_state["thrust_level"]

df.activate_post_combustion(planes[1])

df.set_plane_pitch(planes[1], -0.5)

# Pitch
p = 0
while p < 15:
    df.display_2DText([0.4, 0.7], "Pitch", 0.1, [1, 0.5, 0, 1])
    time.sleep(1/60)
    plane_state = df.get_plane_state(planes[1])
    df.update_scene()
    p = plane_state["pitch_attitude"]

df.stabilize_plane(planes[1])

df.retract_gear(planes[1])


s = 0
while s < 500:
    time.sleep(1/60)
    plane_state = df.get_plane_state(planes[1])
    df.update_scene()
    s = plane_state["altitude"]

plane_id = planes[1]

missiles = df.get_machine_missiles_list(plane_id)

missile_slot = 0
missile_id = missiles[missile_slot]


df.fire_missile(plane_id, missile_slot)

df.set_machine_custom_physics_mode(missile_id, True)

df.set_missile_life_delay(missile_id, 10)

df.update_scene()


missile_state = df.get_missile_state(missile_id)
x, y, z = missile_state["position"][0], missile_state["position"][1], missile_state["position"][2]
y_b = y
z_b = z
v_0 = df.get_plane_state(planes[1])['linear_speed']
y_0 = df.get_plane_state(planes[1])['vertical_speed']

df.set_renderless_mode(False)


missile_matrix = [1, 0, 0,
                  0, 1, 0,
                  0, 0, 1,
                  x, y, z]

# Linear displacement vector in m.s-1
missile_speed_vector = [1, 1, 1]

frame_time_step = 1/60

# Custom missile movements
t = 0
while not missile_state["wreck"]:
    df.display_2DText([0.4, 0.7], "Flare", 0.1, [1, 0.5, 0, 1])
    time.sleep(1/60)
    missile_state = df.get_missile_state(missile_id)
    missile_matrix[9] = x
    missile_matrix[10] = y
    missile_matrix[11] = z

    df.update_machine_kinetics(missile_id, missile_matrix, missile_speed_vector)
    df.update_scene()
    x = x
    y = y_b + y_0 * t - 0.5 * 10 * t * t
    z = z_b + v_0 * t
    

    # Compute speed vector, used by missile engine smoke
    missile_speed_vector = [(x-missile_matrix[9]) / frame_time_step, (y - missile_matrix[10]) / frame_time_step, (z - missile_matrix[11]) / frame_time_step]

    t += frame_time_step


# Custom physics off
df.set_machine_custom_physics_mode(missile_id, False)

df.set_plane_roll(planes[1], .5)

s = 0
while s < 120:
    df.display_2DText([0.4, 0.7], "Roll", 0.1, [1, 0.5, 0, 1])
    time.sleep(1/60)
    df.update_scene()
    s = s + 1

df.set_plane_roll(planes[1], -.5)

s = 0
while s < 120:
    df.display_2DText([0.4, 0.7], "Roll", 0.1, [1, 0.5, 0, 1])
    time.sleep(1/60)
    df.update_scene()
    s = s + 1


df.stabilize_plane(planes[1])

s = 0
while s < 120:
    time.sleep(1/60)
    df.update_scene()
    s = s + 1

df.set_plane_yaw(planes[1], .5)
s = 0
while s < 120:
    df.display_2DText([0.4, 0.7], "Yaw", 0.1, [1, 0.5, 0, 1])
    time.sleep(1/60)
    df.update_scene()
    s = s + 1

df.set_plane_yaw(planes[1], -.5)
s = 0
while s < 120:
    df.display_2DText([0.4, 0.7], "Yaw", 0.1, [1, 0.5, 0, 1])
    time.sleep(1/60)
    df.update_scene()
    s = s + 1

df.stabilize_plane(planes[1])

while s < 120:
    time.sleep(1/60)
    df.update_scene()
    s = s + 1

df.set_plane_flaps(planes[1], 1)
s = 0
while s < 120:
    df.display_2DText([0.4, 0.7], "Flaps", 0.1, [1, 0.5, 0, 1])
    time.sleep(1/60)
    df.update_scene()
    s = s + 1


df.set_plane_flaps(planes[1], -1)
s = 0
while s < 120:
    df.display_2DText([0.4, 0.7], "Flaps", 0.1, [1, 0.5, 0, 1])
    time.sleep(1/60)
    df.update_scene()
    s = s + 1


df.set_plane_brake(planes[1], 1)
df.set_plane_thrust(planes[1], 0)
s = 0
while s < 120:
    df.display_2DText([0.4, 0.7], "Speed brake", 0.1, [1, 0.5, 0, 1])
    time.sleep(1/60)
    df.update_scene()
    s = s + 1

df.deploy_gear(planes[1])


# Client update mode OFF
df.set_client_update_mode(False)

# Disconnect from the Dogfight server
df.disconnect()


