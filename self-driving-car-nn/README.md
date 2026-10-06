# Self-Driving Car (Neural Network + Evolution)

A self-driving car simulation built with **Python**, **Kivy** and **NumPy**.
A population of cars, each controlled by a small neural network, learns to stay on a winding road through a simple genetic algorithm.

## How it works

- **Sensors:** each car casts 5 rays (-55°, -27°, 0°, +27°, +55°) and measures the distance to the road edge.
- **Neural network:** 5 inputs → 8 hidden neurons (tanh) → 1 output (steering).
- **Evolution:** when every car has crashed, the best car (by distance travelled) is kept (elitism) and the rest of the next generation are mutated copies of it.
- **Population:** 25 cars per generation, running at about 30 FPS.

## Requirements

- Python 3.9+
- Kivy
- NumPy

## Installation

```bash
git clone https://github.com/<your-username>/self-driving-car-nn.git
cd self-driving-car-nn

python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

## Run

```bash
python main.py
```

## Project structure

```
self-driving-car-nn/
├── main.py            # Simulation (network, car, road, evolution, Kivy app)
├── requirements.txt   # Dependencies
├── README.md
├── LICENSE
└── .gitignore
```

## Tweaking

| Setting | Location | Default |
|---|---|---|
| Population size | `SelfDrivingCar.population_size` | 25 |
| Mutation strength | `evolve()` (the `* 0.25` factors) | 0.25 |
| Car speed | `Car.speed` | 3 |
| Road width | `Road.road_width` | 300 |
| Sensor range | `Car.sensors()` | 180 px |

## License

MIT — see [LICENSE](LICENSE).
