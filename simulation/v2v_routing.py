"""
v2v_routing.py
Vehicle-to-Vehicle cooperative routing using SUMO TraCI.

Pioneer vehicles broadcast congestion warnings when their fused classifier
score exceeds PIONEER_THRESHOLD. Follower vehicles within COMM_RANGE_M
reroute if the mean received score exceeds FOLLOWER_THRESHOLD.

Packet loss is modelled by randomly discarding messages at send time.

Usage:
    python v2v_routing.py --config njnyc.sumocfg --packet_loss 0.0 --n_trips 500
"""

import random
import math
import argparse

try:
    import traci
    TRACI_AVAILABLE = True
except ImportError:
    TRACI_AVAILABLE = False
    print("WARNING: TraCI not found. Install SUMO and add to PYTHONPATH.")

PIONEER_THRESHOLD  = 0.50   # broadcast if fused_score > 50%
FOLLOWER_THRESHOLD = 0.60   # reroute if mean received score > 60%
COMM_RANGE_M       = 300.0  # V2V communication range in metres
CONGESTED_PENALTY  = 10.0   # travel time multiplier for congested edges


def euclidean_distance(pos1: tuple, pos2: tuple) -> float:
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)


def broadcast_warning(pioneer_id: str, score: float, all_vehicles: list,
                      packet_loss: float) -> list:
    """
    Send congestion warning from pioneer to all vehicles within COMM_RANGE_M.
    Applies packet loss stochastically.
    Returns list of vehicle IDs that successfully received the warning.
    """
    pioneer_pos = traci.vehicle.getPosition(pioneer_id)
    received_by = []

    for vid in all_vehicles:
        if vid == pioneer_id:
            continue
        if random.random() < packet_loss:
            continue   # packet dropped
        follower_pos = traci.vehicle.getPosition(vid)
        if euclidean_distance(pioneer_pos, follower_pos) <= COMM_RANGE_M:
            received_by.append(vid)

    return received_by


def reroute_vehicle(vid: str, congested_edges: set):
    """
    Trigger Dijkstra rerouting on dynamically updated edge weights.
    Congested edges get a CONGESTED_PENALTY x travel time multiplier.
    """
    for edge_id in congested_edges:
        try:
            base_time = traci.edge.getTraveltime(edge_id)
            traci.edge.adaptTraveltime(edge_id, base_time * CONGESTED_PENALTY)
        except traci.TraCIException:
            pass
    traci.vehicle.rerouteTraveltime(vid)


def run_simulation(config_path: str, packet_loss: float = 0.0) -> dict:
    """
    Run one full simulation and return journey time statistics.
    """
    if not TRACI_AVAILABLE:
        raise RuntimeError("SUMO TraCI is required. Install SUMO and set SUMO_HOME.")

    traci.start(['sumo', '-c', config_path, '--no-warnings', 'true'])

    journey_times    = {}   # vid -> start_step
    completed_times  = []
    congested_edges  = set()
    follower_scores  = {}   # vid -> list of received scores

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        step = traci.simulation.getTime()

        # Track departures
        for vid in traci.simulation.getDepartedIDList():
            journey_times[vid] = step

        # Track arrivals
        for vid in traci.simulation.getArrivedIDList():
            if vid in journey_times:
                completed_times.append(step - journey_times[vid])

        all_vehicles = list(traci.vehicle.getIDList())

        for vid in all_vehicles:
            # Estimate congestion score from edge density
            edge_id  = traci.vehicle.getRoadID(vid)
            n_vehs   = traci.edge.getLastStepVehicleNumber(edge_id) if edge_id else 0
            raw_score = min(1.0, n_vehs / 20.0)   # normalise: 20 vehicles = max

            if raw_score > PIONEER_THRESHOLD:
                congested_edges.add(edge_id)
                receivers = broadcast_warning(vid, raw_score, all_vehicles, packet_loss)
                for recv_id in receivers:
                    if recv_id not in follower_scores:
                        follower_scores[recv_id] = []
                    follower_scores[recv_id].append(raw_score)

            # Reroute followers above threshold
            if vid in follower_scores:
                mean_score = sum(follower_scores[vid]) / len(follower_scores[vid])
                if mean_score > FOLLOWER_THRESHOLD:
                    reroute_vehicle(vid, congested_edges)
                    del follower_scores[vid]

    traci.close()

    import statistics
    if not completed_times:
        return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'n': 0}

    return {
        'mean':    statistics.mean(completed_times),
        'std':     statistics.stdev(completed_times) if len(completed_times) > 1 else 0,
        'min':     min(completed_times),
        'max':     max(completed_times),
        'n':       len(completed_times),
        'n_rerouted': sum(1 for s in follower_scores.values() if s),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',       default='njnyc.sumocfg')
    parser.add_argument('--packet_loss',  type=float, default=0.0)
    args = parser.parse_args()

    stats = run_simulation(args.config, args.packet_loss)
    print(f"Journey times (seconds): mean={stats['mean']:.1f}, "
          f"std={stats['std']:.1f}, n={stats['n']}")
