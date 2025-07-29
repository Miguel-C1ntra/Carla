import carla
import torch
import numpy as np
import cv2
import pygame
from PIL import Image
from torchvision import transforms
from train_model import SteeringCNN
import time

# ========== CONFIG ========== #
IMG_WIDTH = 840
IMG_HEIGHT = 680
MAX_RUNTIME = 400
MIN_SPEED = 0.1
STUCK_TIMEOUT = 3
REVERSE_DURATION = 4
REVERSE_THROTTLE = 0.6
AUTOPILOT_DURATION = 5
SPEED_LEVEL = 6
FPS = 30

# ========== TRANSFORMAÇÃO ========== #
transform = transforms.Compose([
    transforms.Resize((66, 200)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ========== PYGAME ========== #
pygame.init()
win = pygame.display.set_mode((IMG_WIDTH, IMG_HEIGHT))
pygame.display.set_caption("AutoDrive View")
clock = pygame.time.Clock()
pygame.font.init()
font = pygame.font.SysFont("Arial", 28)

# ========== CONEXÃO COM CARLA ========== #
client = carla.Client("localhost", 2000)
client.set_timeout(20.0)
world = client.get_world()
bp_lib = world.get_blueprint_library()

# ========== SPAWN VEÍCULO ========== #
vehicle_bp = bp_lib.filter("model3")[0]
spawn_point = world.get_map().get_spawn_points()[1]
vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
vehicle.set_autopilot(False)

# ========== CÂMERA FRONTAL ========== #
camera_bp = bp_lib.find("sensor.camera.rgb")
camera_bp.set_attribute("image_size_x", f"{IMG_WIDTH}")
camera_bp.set_attribute("image_size_y", f"{IMG_HEIGHT}")
camera_bp.set_attribute("fov", "100")

camera_transform = carla.Transform(
    carla.Location(x=1.5, z=2.4),
    carla.Rotation(pitch=-15)
)
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

# ========== UTILITÁRIOS ========== #
def get_speed():
    v = vehicle.get_velocity()
    return np.sqrt(v.x**2 + v.y**2 + v.z**2)

def update_spectator():
    transform = vehicle.get_transform()
    location = transform.location + transform.get_forward_vector() * -8 + carla.Location(z=4)
    rotation = transform.rotation
    rotation.pitch = -15
    world.get_spectator().set_transform(carla.Transform(location, rotation))

# ========== MODELO ========== #
model = SteeringCNN()
model.load_state_dict(torch.load("steering_model.pth", map_location=torch.device('cpu')))
model.eval()

# ========== CONTROLE ========== #
last_move_time = time.time()
in_reverse = False
reverse_start_time = 0
autopilot_active = False
autopilot_start_time = 0

# ========== CALLBACK ========== #
def process_image(image):
    global last_move_time, in_reverse, reverse_start_time
    global autopilot_active, autopilot_start_time

    array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((IMG_HEIGHT, IMG_WIDTH, 4))[:, :, :3]
    array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)

    pil = Image.fromarray(array)
    tensor = transform(pil).unsqueeze(0)

    with torch.no_grad():
        steer = model(tensor).item()
    steer = max(min(steer, 1.0), -1.0)

    speed = get_speed()
    current_time = time.time()

    # === Controle inteligente === #
    if not in_reverse and not autopilot_active and speed < MIN_SPEED and (current_time - last_move_time > STUCK_TIMEOUT):
        print("⚠️ Travado! Iniciando ré.")
        in_reverse = True
        reverse_start_time = current_time

    if in_reverse and (current_time - reverse_start_time > REVERSE_DURATION):
        print("🔄 Ré finalizada. Corrigindo com autopilot.")
        in_reverse = False
        autopilot_active = True
        autopilot_start_time = current_time
        vehicle.set_autopilot(True)

    if autopilot_active and (current_time - autopilot_start_time > AUTOPILOT_DURATION):
        print("✅ Correção concluída. Retornando ao controle IA.")
        vehicle.set_autopilot(False)
        autopilot_active = False
        last_move_time = current_time

    if not autopilot_active:
        control = carla.VehicleControl()

        if in_reverse:
            control.reverse = True
            control.throttle = REVERSE_THROTTLE
            control.steer = 0.3 if int(current_time * 2) % 2 == 0 else -0.3
            control.brake = 0.0
        else:
            control.reverse = False
            control.steer = steer
            base_throttle = 0.4 + (0.3 * (1.0 - abs(steer)))
            speed_scale = SPEED_LEVEL / 10.0
            control.throttle = max(0.3, base_throttle * speed_scale)
            control.brake = 0.0

        vehicle.apply_control(control)
        print(f"🧠 Steer: {steer:.2f} | Throttle: {control.throttle:.2f} | Speed: {speed:.2f} m/s")

    update_spectator()

    # === Interface gráfica === #
    surface = pygame.surfarray.make_surface(np.rot90(array))
    surface = pygame.transform.scale(surface, (IMG_WIDTH, IMG_HEIGHT))
    win.blit(surface, (0, 0))

    kmh = speed * 3.6
    vel_text = font.render(f"Velocidade: {kmh:.1f} km/h", True, (255, 255, 255))
    win.blit(vel_text, (20, 20))
    pygame.display.update()

# ========== LOOP PRINCIPAL ========== #
try:
    print("🚘 IA conduzindo. Pressione ESC ou feche a janela para sair.")
    start_time = time.time()
    last_print = -1
    camera.listen(lambda image: process_image(image))

    while time.time() - start_time < MAX_RUNTIME:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                raise KeyboardInterrupt

        elapsed = int(time.time() - start_time)
        if elapsed != last_print:
            print(f"⏱️  Tempo de simulação: {elapsed}s")
            last_print = elapsed

        clock.tick(FPS)

except KeyboardInterrupt:
    print("🛑 Simulação encerrada pelo usuário.")

finally:
    camera.stop()
    camera.destroy()
    vehicle.destroy()
    pygame.quit()
    print("✅ Recursos liberados com sucesso.")
