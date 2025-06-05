# Carla

# 1. Exemplo: spawn veiculos
 No primeiro exemplo, mostramos como spawnar veículos nos pontos pré-definidos do mapa.
Você pode escolher o modelo do veículo (por exemplo, vehicle.tesla.model3).
Os veículos podem ser posicionados automaticamente em até 155 spawn points existentes no mapa.

O que você aprende:

- Usar get_spawn_points()

- Usar world.try_spawn_actor()

- Controlar spawn por índice (0 a 154)



# 2. Exemplo: Ativar AutoPilot
Neste exemplo, mostramos como ativar o piloto automático (autopilot) para que o veículo navegue sozinho pela cidade.

 O que você aprende:

- Ativar vehicle.set_autopilot(True)

- Deixar o carro navegar pelo tráfego automaticamente

- Observar o comportamento da IA no mapa



# 3. Exemplo: Mudar o Clima
O CARLA permite simular diferentes condições climáticas, como chuva, neblina, céu nublado, etc. Esse exemplo mostra como alterar dinamicamente o clima do ambiente.

O que você aprende:

- Usar carla.WeatherParameters

- Aplicar mudanças como:

- ClearNoon

- CloudySunset

- WetSunset

- HardRainNoon

# 4. Exemplo: Trocar de Cidade / Mapa
CARLA suporta vários mapas, como Town01, Town02, até Town10HD. Você pode carregar outro mapa com um simples comando.

O que você aprende:

Usar:

- client.load_world("Town03")

- Reiniciar o ambiente em uma cidade diferente

