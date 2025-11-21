import os
import torch
import numpy as np
import traci  # SUMO Python API
import traci.constants as tc

# Load the trained DQN model
MODEL_PATH = 'Notebooks/q_network_with_traffic_lights.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dqn_model = torch.load(MODEL_PATH, map_location=device)
dqn_model.eval()

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
    Example state: vehicle density on ramps, traffic light phases, etc.
    """
    # Define state parameters
    ramp_vehicle_ids = traci.lane.getLastStepVehicleIDs("ramp_0")  # Replace with your ramp lane ID
    main_road_vehicle_ids = traci.lane.getLastStepVehicleIDs("main_road_0")  # Replace with your main road lane ID
    traffic_light_phase = traci.trafficlight.getPhase("intersection_0")  # Replace with your traffic light ID

    ramp_density = len(ramp_vehicle_ids)  # Number of vehicles on the ramp
    main_road_density = len(main_road_vehicle_ids)  # Number of vehicles on the main road

    # Return the state as a normalized vector (if required for the DQN input)
    return np.array([ramp_density, main_road_density, traffic_light_phase], dtype=np.float32)

def select_action(state):
    """
    Select the best action for the given state using the trained DQN model.
    """
    # Convert state to tensor and pass it through the neural network
    state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        q_values = dqn_model(state_tensor)
    return torch.argmax(q_values).item()  # Action with the highest Q-value

def apply_action(action):
    """
    Apply the selected action (ramp metering control) in the simulation.
    """
    if action == 0:
        traci.trafficlight.setPhase("ramp_meter", 0)  # Set ramp meter to green
    elif action == 1:
        traci.trafficlight.setPhase("ramp_meter", 1)  # Set ramp meter to red

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

            # Select the best action using the DQN model
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
