"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R """

import math
import random
from collections import namedtuple


# Predicting ENDPOINTS
# TODO: Define corresponding endpoints to connect with Bonsai
ENDPOINT_CART_POSITION = "HTTP:..."
ENDPOINT_CART_VELOCITY = "HTTP:..."
ENDPOINT_POLE_ANGLE = "HTTP:..."
ENDPOINT_POLE_ANGULAR_VELOCITY = "HTTP:..."

# Constants that we might want to let people change
DEFAULT_CART_MASS = 0.31  # kg
DEFAULT_POLE_MASS = 0.055  # kg
DEFAULT_POLE_LENGTH = 0.4  # m

# Constants not exposed for changing for now
GRAVITY = 9.8  # a classic...
FORCE_MAG = 1.0
STEP_DURATION = 0.02  # seconds between state updates (20ms)
TRACK_WIDTH = 2.0  # m
FORCE_NOISE = 0.02  # % of FORCE_MAG



def normalize_angle(x):
    """
    convert an angle to [-pi, pi]
    """
    return (x+math.pi) % (2*math.pi) - math.pi


class CartPole:
    """
    Model for the dynamics of an inverted pendulum
    """

    def __init__(self):
        # Set a constants needed by renderer
        self.x_threshold = TRACK_WIDTH / 2

        # Default Static Initializations
        self._cart_mass = DEFAULT_CART_MASS  # kg
        self._pole_mass = DEFAULT_POLE_MASS  # (kg)
        self._pole_length = DEFAULT_POLE_LENGTH  # (m)

        # ENDPOINTS available to query next environment state.
        self.predict_cart_position = ENDPOINT_CART_POSITION
        self.predict_cart_velocity = ENDPOINT_CART_VELOCITY
        self.predict_pole_angle = ENDPOINT_POLE_ANGLE
        self.predict_pole_angular_velocity = ENDPOINT_POLE_ANGULAR_VELOCITY

        # DEFINE WHETHER TO RUN WITH ENDPOINT PREDICTION OR SIM PREDICTIONS
        self.run_with_sim = False

        # Initialize Simulation
        self.reset(
            initial_pole_angle=random.uniform(-0.05, 0.05),
            initial_angular_velocity=random.uniform(-0.05, 0.05),
        )

    def reset(
        self,
        initial_cart_position: float = 0,
        initial_cart_velocity: float = 0,
        initial_pole_angle: float = 0,
        initial_angular_velocity: float = 0,
        target_pole_position: float = 0,
    ):
        self._cart_position = initial_cart_position  # (m)
        self._cart_velocity = initial_cart_velocity  # (m/s)
        self._pole_angle = normalize_angle(initial_pole_angle)  # (rad)
        self._pole_angular_velocity = initial_angular_velocity  # (rad/s)
        self._target_pole_position = target_pole_position  # (m)

        # Compute
        self._update_pole_center_state()

    def _update_pole_center_state(self):
        """
        Update _pole_center_position and _pole_center_velocity.
        """
        # Use the pole center, not the cart center, for tracking
        # pole center velocity.
        pole_half_length = self._pole_length / 2
        self._pole_center_position = (
            self._cart_position + math.sin(self._pole_angle) * pole_half_length
        )
        self._pole_center_velocity = (
            self._cart_velocity
            + math.sin(self._pole_angular_velocity) * pole_half_length
        )

    def step(self, action: float):
        """ 
        Move the state of the cartpole simulation forward one time unit.

        Args:
            action: a dict with a key 'command', with value between -1 and 1.
        """

        
        if not self.run_with_sim:
            # PREPARE INPUT TO ML MODEL PREDICTORS.
            pred_input = self.get_predictor_input(action)

            # QUERY ENDPOINTS TO GET NEXT ENVIRONMENT STATE.
            # TODO: Requires proper call to ENDPOINTS URL using the formatted input 'pred_input'.
            self._cart_position = self.predict_cart_position(pred_input)
            self._cart_velocity = self.predict_cart_velocity(pred_input)
            self._pole_angle = self.predict_pole_angle(pred_input)
            self._pole_angular_velocity = self.predict_pole_angular_velocity(pred_input)

            # Compute additional state space based on predicted state space.
            self._update_pole_center_state()
        

        if self.run_with_sim:
            # We are expecting the input command to be -1 or 1,
            # but we'll support a continuous action space.
            # Add a small amount of random noise to the force so
            # the policy can't succeed by simply applying zero
            # force each time.
            force = FORCE_MAG * (action + random.uniform(-0.02, 0.02))

            # Precompute some helpful quantities
            total_mass = self._cart_mass + self._pole_mass
            pole_half_length = self._pole_length / 2
            pole_mass_length = self._pole_mass * pole_half_length

            cosTheta = math.cos(self._pole_angle)
            sinTheta = math.sin(self._pole_angle)

            temp = (
                force + pole_mass_length * self._pole_angular_velocity ** 2 * sinTheta
            ) / total_mass
            angularAccel = (GRAVITY * sinTheta - cosTheta * temp) / (
                pole_half_length
                * (4.0 / 3.0 - (self._pole_mass * cosTheta ** 2) / total_mass)
            )
            linearAccel = temp - (pole_mass_length * angularAccel * cosTheta) / total_mass

            self._cart_position = self._cart_position + STEP_DURATION * self._cart_velocity
            self._cart_velocity = self._cart_velocity + STEP_DURATION * linearAccel

            self._pole_angle = (
                self._pole_angle + STEP_DURATION * self._pole_angular_velocity
            )
            self._pole_angle = normalize_angle(self._pole_angle)

            self._pole_angular_velocity = (
                self._pole_angular_velocity + STEP_DURATION * angularAccel
            )

            self._update_pole_center_state()

    @property
    def state(self):
        return {
            "cart_position": self._cart_position,
            "cart_velocity": self._cart_velocity,
            "pole_angle": self._pole_angle,
            "pole_angular_velocity": self._pole_angular_velocity,
            "pole_center_position": self._pole_center_position,
            "pole_center_velocity": self._pole_center_velocity,
            "target_pole_position": self._target_pole_position,
            # also return the constants from the config
            "cart_mass": self._cart_mass,
            "pole_mass": self._pole_mass,
            "pole_length": self._pole_length,
        }
    
    def get_predictor_input(self, action):

        # PREPARE INPUT TO ML MODELS

        env_input_list = []
        env_input_list.append(self._cart_position)
        env_input_list.append(self._cart_velocity)
        env_input_list.append(self._pole_angle)
        env_input_list.append(self._pole_angular_velocity)
        env_input_list.append(action)

        # TODO: Requires transformation to ML MODEL Endpoint input format.

        return env_input_list


def create_viewer(model):
    from render import Viewer

    viewer = Viewer()
    viewer.model = model
    return viewer


if __name__ == "__main__":

    import random
    import sys

    model = CartPole()
    viewer = create_viewer(model)
    number_iterations = 300
    for i in range(number_iterations):
        print(", ".join([f"{k}: {v:.3f}" for k,v in model.state.items()]))
        actions = [-1, 1]
        action = random.sample(actions, 1)[0]
        state = viewer.model.step(action)
        viewer.update()
        if viewer.has_exit:
            viewer.close()
            sys.exit(0)
