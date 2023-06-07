
# - example_test_trajectory


from utils import QuadState,Model
from controller import AdapLowLevelControl


low_level_controller = AdapLowLevelControl()
cur_state = QuadState()

motor_spd_command = low_level_controller.run(cur_state)