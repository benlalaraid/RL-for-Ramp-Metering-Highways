import os
import numpy as np
import traci  # SUMO Python API
import traci.constants as tc

# Load the Q-learning model (Q-table) trained with traffic lights
Q_TABLE_PATH = 'Notebooks/q_table_with_traffic_lights.npy'
q_table = np.load(Q_TABLE_PATH)

# Simulation parameters
SUMO_CFG_FILE = "simulation.sumocfg"  # Your SUMO configuration file
NUM_STEPS = 1000  # Number of simulation steps
ACTION_MAPPING = {
    0: "green",  # Ramp metering open
    1: "red"     # Ramp metering closed
}

def get_current_state():
    """
    Extract the current state representation from SUMO.
    Example state: vehicle density on ramps, average speed, etc.
    """
    # Define state parameters
    ramp_vehicle_ids = traci.lane.getLastStepVehicleIDs("ramp_0")  # Replace with your ramp lane ID
    main_road_vehicle_ids = traci.lane.getLastStepVehicleIDs("main_road_0")  # Replace with your main road lane ID

    ramp_density = len(ramp_vehicle_ids)  # Number of vehicles on the ramp
    main_road_density = len(main_road_vehicle_ids)  # Number of vehicles on the main road

    # Return the state as an integer (assuming discrete state space)
    # You may encode this differently if you used a more complex state representation
    return ramp_density, main_road_density

def select_action(state):
    """
    Select the best action for the given state using the Q-table.
    """
    # Convert state to a discrete index (assuming states are encoded as tuples)
    state_index = encode_state(state)  # Implement encoding if needed
    return np.argmax(q_table[state_index])  # Action with the maximum Q-value

def apply_action(action):
    """
    Apply the selected action (ramp metering control) in the simulation.
    """
    if action == 0:
        traci.trafficlight.setPhase("ramp_meter", 0)  # Set ramp meter to green
    elif action == 1:
        traci.trafficlight.setPhase("ramp_meter", 1)  # Set ramp meter to red

def encode_state(state):
    """
    Encode the state into an index to match the Q-table.
    Example: Assuming state space is small, flatten the tuple into an integer.
    """
    # Adjust encoding based on your Q-table's structure
    ramp_density, main_road_density = state
    return ramp_density * 10 + main_road_density  # Example encoding

def main():
    """
    Main simulation loop.
    """
    # Start SUMO simulation
    traci.start(["sumo", "-c", SUMO_CFG_FILE])

    try:
        for step in range(NUM_STEPS):
            # Advance the simulation step
            traci.simulationStep()

            # Get the current state
            current_state = get_current_state()

            # Select the best action using the Q-table
            action = select_action(current_state)

            # Apply the selected action
            apply_action(action)

            # Optional: Collect metrics for analysis
            print(f"Step {step}: State={current_state}, Action={ACTION_MAPPING[action]}")
    finally:
        # Close the simulation
        traci.close()

if __name__ == "__main__":
    main()
