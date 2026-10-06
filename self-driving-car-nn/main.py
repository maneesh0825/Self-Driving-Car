import math
import numpy as np

from kivy.app import App
from kivy.clock import Clock

from kivy.graphics import (
    Color,
    Rectangle,
    Line,
    PushMatrix,
    PopMatrix,
    Rotate
)

from kivy.uix.widget import Widget
from kivy.uix.label import Label


# ============================================================
# 1. NEURAL NETWORK
# ============================================================

class NeuralNetwork:

    def __init__(self):

        # 5 sensor inputs
        # 8 hidden neurons
        # 1 steering output

        self.W1 = np.random.randn(5, 8) * 0.7
        self.b1 = np.zeros(8)

        self.W2 = np.random.randn(8, 1) * 0.7
        self.b2 = np.zeros(1)

    def predict(self, sensors):

        x = np.array(sensors, dtype=float)

        # Input -> hidden layer
        hidden = np.tanh(
            np.dot(x, self.W1) + self.b1
        )

        # Hidden -> output layer
        output = np.tanh(
            np.dot(hidden, self.W2) + self.b2
        )

        return float(output[0])


# ============================================================
# 2. CAR
# ============================================================

class Car:

    def __init__(self, nn):

        self.nn = nn

        # Starting position
        self.x = 400
        self.y = 80

        # 90 degrees = upward
        self.angle = 90

        self.speed = 3

        # Fitness
        self.distance = 0

        self.alive = True

        # Car dimensions
        self.width = 24
        self.height = 42

    # --------------------------------------------------------
    # SENSOR SYSTEM
    # --------------------------------------------------------

    def sensors(self, road):

        # Five sensors:
        #
        # left       = -55 degrees
        # front-left = -27 degrees
        # front      = 0 degrees
        # front-right= +27 degrees
        # right      = +55 degrees

        angles = [-55, -27, 0, 27, 55]

        values = []

        for relative_angle in angles:

            # Absolute angle of this sensor
            a = math.radians(
                self.angle + relative_angle
            )

            # Start with maximum distance
            distance = 180

            # Cast ray
            for d in range(10, 181, 5):

                sx = (
                    self.x
                    + math.cos(a) * d
                )

                sy = (
                    self.y
                    + math.sin(a) * d
                )

                # Boundary detected
                if not road.is_inside(sx, sy):
                    distance = d
                    break

            # Normalize to approximately 0 -> 1
            values.append(distance / 180.0)

        return values

    # --------------------------------------------------------
    # UPDATE CAR
    # --------------------------------------------------------

    def update(self, road):

        if not self.alive:
            return

        # Read environment
        sensor_values = self.sensors(road)

        # Neural network decides steering
        steering = self.nn.predict(sensor_values)

        # Steering
        self.angle += steering * 5

        # Convert degrees to radians
        radians = math.radians(self.angle)

        # Move forward
        self.x += math.cos(radians) * self.speed
        self.y += math.sin(radians) * self.speed

        # Fitness
        self.distance += self.speed

        # Crash detection
        if not road.is_inside(self.x, self.y):

            self.alive = False

    # --------------------------------------------------------
    # DRAW CAR
    # --------------------------------------------------------

    def draw(self):

        PushMatrix()

        Rotate(
            angle=self.angle - 90,
            origin=(self.x, self.y)
        )

        # Car body
        Color(
            0.1,
            0.4,
            1
        )

        Rectangle(
            pos=(
                self.x - self.width / 2,
                self.y - self.height / 2
            ),
            size=(
                self.width,
                self.height
            )
        )

        # Windshield
        Color(
            0.7,
            0.9,
            1
        )

        Rectangle(
            pos=(
                self.x - 8,
                self.y + 5
            ),
            size=(
                16,
                12
            )
        )

        # Wheels
        Color(
            0.05,
            0.05,
            0.05
        )

        # Left rear
        Rectangle(
            pos=(
                self.x - 16,
                self.y - 13
            ),
            size=(6, 13)
        )

        # Right rear
        Rectangle(
            pos=(
                self.x + 10,
                self.y - 13
            ),
            size=(6, 13)
        )

        # Left front
        Rectangle(
            pos=(
                self.x - 16,
                self.y + 2
            ),
            size=(6, 13)
        )

        # Right front
        Rectangle(
            pos=(
                self.x + 10,
                self.y + 2
            ),
            size=(6, 13)
        )

        PopMatrix()


# ============================================================
# 3. ROAD
# ============================================================

class Road:

    def __init__(self):

        self.points = []

        # Generate road center points
        for y in range(0, 2500, 50):

            x = (
                400
                + math.sin(y * 0.004) * 180
                + math.sin(y * 0.009) * 80
            )

            self.points.append((x, y))

        self.road_width = 300

    # --------------------------------------------------------
    # FIND ROAD CENTER
    # --------------------------------------------------------

    def road_center(self, y):

        index = int(y / 50)

        index = max(
            0,
            min(
                index,
                len(self.points) - 1
            )
        )

        return self.points[index][0]

    # --------------------------------------------------------
    # CHECK WHETHER POSITION IS ON ROAD
    # --------------------------------------------------------

    def is_inside(self, x, y):

        if y < 0 or y > 2450:
            return False

        center = self.road_center(y)

        return (
            abs(x - center)
            < self.road_width / 2
        )

    # --------------------------------------------------------
    # DRAW ROAD
    # --------------------------------------------------------

    def draw(self):

        # Grass
        Color(
            0.15,
            0.5,
            0.15
        )

        Rectangle(
            pos=(0, 0),
            size=(800, 600)
        )

        # Road
        Color(
            0.15,
            0.15,
            0.15
        )

        for y in range(0, 2450, 20):

            center = self.road_center(y)

            Rectangle(
                pos=(
                    center - self.road_width / 2,
                    y
                ),
                size=(
                    self.road_width,
                    25
                )
            )

        # Center line
        Color(
            1,
            1,
            0
        )

        for y in range(0, 2450, 80):

            center = self.road_center(y)

            Rectangle(
                pos=(
                    center - 3,
                    y
                ),
                size=(6, 40)
            )


# ============================================================
# 4. SELF-DRIVING ENVIRONMENT
# ============================================================

class SelfDrivingCar(Widget):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        # Road
        self.road = Road()

        # Generation number
        self.generation = 1

        # Population size
        self.population_size = 25

        # Create random brains
        self.population = [
            NeuralNetwork()
            for _ in range(self.population_size)
        ]

        # Cars
        self.cars = []

        # Best distance ever
        self.best_distance = 0

        # Create first generation
        self.create_generation()

        # Information label
        self.info = Label(
            text="Generation: 1",
            font_size=18,
            pos=(10, 540),
            size=(500, 60)
        )

        self.add_widget(self.info)

        # Run simulation at ~30 FPS
        Clock.schedule_interval(
            self.update,
            1 / 30
        )

    # --------------------------------------------------------
    # CREATE GENERATION
    # --------------------------------------------------------

    def create_generation(self):

        self.cars = []

        for nn in self.population:

            car = Car(nn)

            self.cars.append(car)

    # --------------------------------------------------------
    # EVOLUTION
    # --------------------------------------------------------

    def evolve(self):

        # Calculate fitness
        fitness = np.array([
            car.distance
            for car in self.cars
        ])

        # Best car
        best_index = np.argmax(fitness)

        best_network = (
            self.cars[best_index].nn
        )

        # Save best result
        self.best_distance = max(
            self.best_distance,
            fitness[best_index]
        )

        # Elitism
        new_population = [
            best_network
        ]

        # Mutation
        while len(new_population) < self.population_size:

            child = NeuralNetwork()

            # Mutate weights
            child.W1 = (
                best_network.W1
                + np.random.randn(
                    *best_network.W1.shape
                ) * 0.25
            )

            # Mutate hidden biases
            child.b1 = (
                best_network.b1
                + np.random.randn(
                    *best_network.b1.shape
                ) * 0.25
            )

            # Mutate output weights
            child.W2 = (
                best_network.W2
                + np.random.randn(
                    *best_network.W2.shape
                ) * 0.25
            )

            # Mutate output bias
            child.b2 = (
                best_network.b2
                + np.random.randn(
                    *best_network.b2.shape
                ) * 0.25
            )

            new_population.append(child)

        # Replace population
        self.population = new_population

        # Next generation
        self.generation += 1

        self.create_generation()

    # --------------------------------------------------------
    # MAIN GAME LOOP
    # --------------------------------------------------------

    def update(self, dt):

        alive = False

        # Update every car
        for car in self.cars:

            car.update(self.road)

            if car.alive:
                alive = True

        # If everyone crashed,
        # evolve the population
        if not alive:

            self.evolve()

        # Redraw
        self.canvas.clear()

        with self.canvas:

            self.road.draw()

            # Draw cars
            for car in self.cars:

                car.draw()

                # Draw sensors
                if car.alive:

                    sensors = car.sensors(
                        self.road
                    )

                    angles = [
                        -55,
                        -27,
                        0,
                        27,
                        55
                    ]

                    for value, offset in zip(
                        sensors,
                        angles
                    ):

                        a = math.radians(
                            car.angle + offset
                        )

                        length = value * 180

                        Color(
                            1,
                            0.3,
                            0.1
                        )

                        Line(
                            points=[
                                car.x,
                                car.y,

                                car.x
                                + math.cos(a)
                                * length,

                                car.y
                                + math.sin(a)
                                * length
                            ],
                            width=1
                        )

        # Current best
        best_current = max(
            car.distance
            for car in self.cars
        )

        # Update information
        self.info.text = (
            f"Generation: {self.generation}\n"
            f"Current best distance: "
            f"{int(best_current)}\n"
            f"Overall best: "
            f"{int(self.best_distance)}"
        )


# ============================================================
# 5. KIVY APP
# ============================================================

class CarApp(App):

    def build(self):

        return SelfDrivingCar()


# ============================================================
# 6. RUN
# ============================================================

if __name__ == "__main__":
    CarApp().run()
