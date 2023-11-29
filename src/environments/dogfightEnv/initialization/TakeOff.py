
def start_on_carrier(
    df,
    task,
    planeID,
    enemyID,
    missile_slot,
    missileID,
):
    t = 0
    while t < 1:
        plane_state = df.get_plane_state(enemyID)
        df.update_scene()
        t = plane_state["thrust_level"]
    
    df.activate_post_combustion(enemyID)
    df.activate_post_combustion(planeID)

    df.set_plane_pitch(enemyID, -0.5)
    df.set_plane_pitch(planeID, -0.5)

    p = 0
    while p < 15:
        plane_state = df.get_plane_state(enemyID)
        df.update_scene()
        p = plane_state["pitch_attitude"]

    df.stabilize_plane(enemyID)
    df.stabilize_plane(planeID)

    df.retract_gear(enemyID)
    df.retract_gear(planeID)

    s = 0
    while s < 1000:
        plane_state = df.get_plane_state(enemyID)
        df.update_scene()
        s = plane_state["altitude"]
    
    df.set_plane_yaw(planeID, 1)

    if task == 'evade':
        # missiles = df.get_machine_missiles_list(enemyID)
        # missileID = missiles[missile_slot]

        df.fire_missile(enemyID, missile_slot)

        df.set_missile_target(missileID, planeID)
        df.set_missile_life_delay(missileID, 30)

    elif task == 'dogfight':

        df.set_target_id(enemyID, planeID)
        df.activate_IA(enemyID)


def start_in_sky(
    df,
    task,
    planeID,
    enemyID,
    missile_slot,
    missileID,
):
    t = 0
    while t < 1:
        plane_state = df.get_plane_state(planeID)
        df.update_scene()
        t = plane_state["thrust_level"]
    df.activate_post_combustion(planeID)

    if task == 'evade':

        df.fire_missile(enemyID, missile_slot)

        df.set_missile_target(missileID, planeID)
        df.set_missile_life_delay(missileID, 10)

    elif task == 'dogfight':

        df.set_target_id(enemyID, planeID)
        df.activate_IA(enemyID)










