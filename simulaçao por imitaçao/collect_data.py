import carla
import numpy as np
import cv2
import os
import time
import random
import pygame

# ========== CONFIGURAÇÕES ========== #
IMAGE_W, IMAGE_H = 800, 600
SAVE_W, SAVE_H = 160, 120
SIMULATION_DURATION = 100
SAVE_FOLDER = "dataset"
FPS = 20

# ========== PYGAME ========== #
pygame.init()
display = pygame.display.set_mode((IMAGE_W, IMAGE_H))
pygame.display.set_caption("CARLA - Câmera Frontal")

# ========== PREPARO ========== #
os.makedirs(f"{SAVE_FOLDER}/images", exist_ok=True)
log_path = f"{SAVE_FOLDER}/labels.csv"
log_file = open(log_path, "w")
log_file.write("frame,steer,throttle,brake,speed,x,y,yaw,on_road\n")

# ========== CONEXÃO COM CARLA ========== #
client = carla.Client("localhost", 2000)
client.set_timeout(10.0)
world = client.get_world()
bp_lib = world.get_blueprint_library()
tm = client.get_trafficmanager()
tm.set_synchronous_mode(False)
carla_map = world.get_map()

# ========== VEÍCULO ========== #
vehicle_bp = bp_lib.filter("model3")[0]
spawn_point = random.choice(carla_map.get_spawn_points())
vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
vehicle.set_autopilot(True, tm.get_port())
tm.ignore_lights_percentage(vehicle, 100.0)
tm.distance_to_leading_vehicle(vehicle, False)

# ========== CÂMERAS ========== #
def create_camera(transform):
    bp = bp_lib.find("sensor.camera.rgb")
    bp.set_attribute("image_size_x", str(IMAGE_W))
    bp.set_attribute("image_size_y", str(IMAGE_H))
    bp.set_attribute("fov", "100")
    return world.spawn_actor(bp, transform, attach_to=vehicle)

camera_data = {}
def make_callback(name):
    def callback(image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((IMAGE_H, IMAGE_W, 4))[:, :, :3]
        rgb_image = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
        camera_data[name] = rgb_image
    return callback

cameras_config = {
    "front": carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(pitch=-15)),
    "rear": carla.Transform(carla.Location(x=-5.0, z=2.4), carla.Rotation(yaw=180)),
    "left": carla.Transform(carla.Location(y=-1.5, z=2.0), carla.Rotation(yaw=-90)),
    "right": carla.Transform(carla.Location(y=1.5, z=2.0), carla.Rotation(yaw=90))
}

cameras = {}
for name, transform in cameras_config.items():
    cam = create_camera(transform)
    cam.listen(make_callback(name))
    cameras[name] = cam

# ========== LOOP DE COLETA ========== #
frame_counter = 0
start_time = time.time()
last_time = start_time
interval = 1.0 / FPS

try:
    while time.time() - start_time < SIMULATION_DURATION:
        now = time.time()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt

        if all(name in camera_data for name in cameras) and now - last_time >= interval:
            last_time = now
            control = vehicle.get_control()
            velocity = vehicle.get_velocity()
            speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

            transform = vehicle.get_transform()
            location = transform.location
            rotation = transform.rotation
            waypoint = carla_map.get_waypoint(location, project_to_road=False)
            on_road = int(waypoint is not None)

            noisy_steer = max(min(control.steer + random.uniform(-0.05, 0.05), 1.0), -1.0)

            for cam_name, image in camera_data.items():
                img_resized = cv2.resize(image, (SAVE_W, SAVE_H))
                filename = f"{frame_counter}_{cam_name}.png"
                cv2.imwrite(f"{SAVE_FOLDER}/images/{filename}", img_resized)

            surface = pygame.surfarray.make_surface(np.rot90(camera_data["front"]))
            display.blit(surface, (0, 0))
            pygame.display.update()

            log_file.write(f"{frame_counter},{noisy_steer:.3f},{control.throttle:.3f},{control.brake:.3f},{speed:.2f},"
                           f"{location.x:.2f},{location.y:.2f},{rotation.yaw:.2f},{on_road}\n")

            print(f"[{frame_counter:04d}] Speed: {speed:.2f} m/s | Pos: ({location.x:.1f}, {location.y:.1f}) | On Road: {on_road}")
            frame_counter += 1

        time.sleep(0.005)

except KeyboardInterrupt:
    print("🛑 Coleta interrompida pelo usuário.")

finally:
    for cam in cameras.values():
        cam.stop()
        cam.destroy()
    vehicle.destroy()
    log_file.close()
    pygame.quit()
    print(f"\n✅ Coleta finalizada com {frame_counter} frames.")
