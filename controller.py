import numpy as np
from numpy.typing import ArrayLike

from racetrack import RaceTrack

# ============================================================================
# Helper functions
# ============================================================================

def resample_path(path: np.ndarray, new_len: int) -> np.ndarray:
    old_len = len(path)
    old_idx = np.linspace(0, 1, old_len)
    new_idx = np.linspace(0, 1, new_len)
    return np.column_stack([
        np.interp(new_idx, old_idx, path[:, 0]),
        np.interp(new_idx, old_idx, path[:, 1])
    ])


def compute_arc_length(path: np.ndarray) -> np.ndarray:
    """Compute cumulative arc length along a path."""
    diffs = np.diff(path, axis=0)
    seg_lengths = np.sqrt(np.sum(diffs**2, axis=1))
    return np.concatenate([[0.0], np.cumsum(seg_lengths)])


def find_closest_point(state: ArrayLike, path: ArrayLike) -> tuple[int, float]:
    car_pos = state[0:2]
    distances = np.linalg.norm(path - car_pos, axis=1)
    idx = int(np.argmin(distances))
    return idx, float(distances[idx])


def find_lookahead_point(state: ArrayLike, path: ArrayLike, lookahead_distance: float):
    closest_idx, _ = find_closest_point(state, path)
    cumulative_dist = 0.0
    lookahead_idx = closest_idx

    for i in range(1, len(path)):
        prev = (closest_idx + i - 1) % len(path)
        nxt = (closest_idx + i) % len(path)
        cumulative_dist += np.linalg.norm(path[nxt] - path[prev])
        if cumulative_dist >= lookahead_distance:
            lookahead_idx = nxt
            break

    return lookahead_idx, path[lookahead_idx]


def estimate_curvature(path: ArrayLike, idx: int, step: int = 5) -> float:
    N = len(path)
    i1, i2, i3 = (idx-step) % N, idx % N, (idx+step) % N
    p1, p2, p3 = path[i1], path[i2], path[i3]

    area = 0.5 * abs(
        (p2[0]-p1[0])*(p3[1]-p1[1]) -
        (p3[0]-p1[0])*(p2[1]-p1[1])
    )

    a = np.linalg.norm(p2 - p1)
    b = np.linalg.norm(p3 - p2)
    c = np.linalg.norm(p1 - p3)

    denom = a*b*c
    curvature = 0 if denom < 1e-9 else 4*area/denom

    # Slightly higher scaling so tight stuff looks tighter
    curvature *= 0.50          # was 0.35
    return max(curvature, 0.0005)

# ============================================================================
# S1 – Velocity reference
# ============================================================================

def compute_safe_corner_speed(curvature: float, v_max: float) -> float:
    if curvature < 1e-6:
        return v_max

    # More conservative lateral accel limits
    if curvature < 0.008:
        a_lat = 17.0      # was 19.0
    elif curvature < 0.020:
        a_lat = 14.0      # was 16.5
    elif curvature < 0.035:
        a_lat = 11.5      # was 14.0
    else:
        a_lat = 9.0       # was 11.0

    return min(np.sqrt(a_lat / curvature), v_max)


def compute_braking_distance(v_current, v_target, a_brake):
    if v_current <= v_target:
        return 0.0
    return max((v_current**2 - v_target**2) / (2*a_brake), 0.0)


def compute_reference_velocity(state, path, parameters, raceline_mode=False):
    v_min = float(parameters[2])
    v_max = float(parameters[5])
    a_max = float(parameters[10])

    current_speed = abs(state[3])
    closest_idx, _ = find_closest_point(state, path)

    base_look = 40
    if current_speed > 60:
        extra = ((current_speed - 60)**1.3) / 2
        # allow a slightly longer preview so we see the chicane in time
        look = min(base_look + current_speed/2.5 + extra, 120)  # was 100
    else:
        look = min(base_look + current_speed/2.5, 90)           # was 80

    look = int(look)

    min_safe = v_max
    dist_to_corner = 0
    found_corner = False
    cumulative = 0.0

    for i in range(look):
        check_idx = (closest_idx + i) % len(path)
        if i > 0:
            prev = (closest_idx + i - 1) % len(path)
            cumulative += np.linalg.norm(path[check_idx] - path[prev])

        curv = estimate_curvature(path, check_idx, step=2)
        if curv > 0.0007:
            safe = compute_safe_corner_speed(curv, v_max)
            if safe < min_safe:
                min_safe = safe
                dist_to_corner = cumulative
                found_corner = True

    if not found_corner or min_safe >= current_speed:
        return v_max

    # More conservative braking model → earlier braking
    brake_accel = 0.70 * a_max                  # was 0.85 * a_max
    needed = compute_braking_distance(current_speed, min_safe, brake_accel) * 1.20  # was *1.05

    if dist_to_corner <= needed:
        # Start actually targeting the lower speed
        return min_safe
    elif dist_to_corner < needed * 1.3:
        # Coasting / gentle transition
        return current_speed
    else:
        return v_max

# ============================================================================
# S2 – Steering reference
# ============================================================================

def compute_reference_steering(state, path, parameters):
    wheelbase = float(parameters[0])
    delta_max = float(parameters[4])

    v = abs(state[3])
    speed = max(v, 1.0)

    # Smoother lookahead (slightly larger)
    if speed < 50:
        lad = 7.0 + 0.15*speed   # was 6.0 + 0.15*speed
    else:
        lad = 14.0 + 0.20*(speed - 50)  # was 13.0

    lad *= 0.75

    _, lookahead_point = find_lookahead_point(state, path, lad)

    sx, sy = state[0], state[1]
    heading = state[4]

    dx = lookahead_point[0] - sx
    dy = lookahead_point[1] - sy

    cos_h = np.cos(-heading)
    sin_h = np.sin(-heading)

    lx = dx*cos_h - dy*sin_h
    ly = dx*sin_h + dy*cos_h
    lx = max(lx, 0.5)

    Ld = max(np.hypot(lx, ly), 1.0)
    alpha = np.arctan2(ly, lx)

    # Pure pursuit slightly less aggressive (3.0 → 2.5)
    pp = np.arctan(2.5 * wheelbase * np.sin(alpha) / Ld)

    closest_idx, _ = find_closest_point(state, path)
    nxt = (closest_idx + 1) % len(path)

    path_vec = path[nxt] - path[closest_idx]
    norm = np.linalg.norm(path_vec)
    path_vec = path_vec / norm if norm > 1e-6 else np.array([1.0, 0.0])

    to_car = np.array([sx, sy]) - path[closest_idx]
    cross_track = path_vec[0]*to_car[1] - path_vec[1]*to_car[0]

    # Stanley softened (1.8 → 1.5)
    stan = np.arctan(1.5 * cross_track / max(speed, 2.0))

    # Combine with gentler weighting (0.4 → 0.30)
    delta_ref = pp + 0.30 * stan

    return float(np.clip(delta_ref, -delta_max, delta_max))

# ============================================================================
# Low level
# ============================================================================

def velocity_controller(state, v_ref, parameters):
    v = float(state[3])
    a_max = float(parameters[10])

    error = v_ref - v

    if error < 0:
        if v < 50: kp = 4.5
        elif v > 90: kp = 5.8
        else: kp = 4.5 + (v - 50)/40 * 1.3
    else:
        kp = 3.5

    a = kp * error
    return float(np.clip(a, -a_max, a_max))


def steering_controller(state, delta_ref, parameters, prev_error=0.0, dt=0.1):
    delta = float(state[2])
    v = abs(state[3])

    v_delta_min = float(parameters[7])
    v_delta_max = float(parameters[9])

    error = delta_ref - delta
    d_error = (error - prev_error) / dt

    if v < 20:
        kp, kd = 3.0, 0.35
    elif v > 70:
        kp, kd = 3.0, 0.65
    else:
        kd = 0.35 + (v - 20)/50 * 0.30
        kp = 3.0

    v_delta = kp*error + kd*d_error
    v_delta = np.clip(v_delta, v_delta_min, v_delta_max)

    return float(v_delta), error

# ============================================================================
# High level controller – with **correct progress-based spatial blending**
# ============================================================================

def controller(state: ArrayLike, parameters: ArrayLike, racetrack: RaceTrack) -> ArrayLike:

    if racetrack.raceline is None:
        raise ValueError("Raceline not found")

    raceline = racetrack.raceline

    # --- compute arc length once ---
    if not hasattr(controller, "arc_len"):
        controller.arc_len = compute_arc_length(raceline)
        controller.total_len = controller.arc_len[-1]

    # --- compute car progress ---
    idx, _ = find_closest_point(state, raceline)
    progress_m = controller.arc_len[idx]   # meters from start of lap

    # --- compute blending weight ---
    TRANSITION_M = 200.0   # distance to shift centerline → raceline
    w = max(0.0, min(1.0, 1.0 - progress_m / TRANSITION_M))

    # --- compute centerline and resample ---
    center_raw = (racetrack.left_boundary + racetrack.right_boundary) / 2.0
    center_resampled = resample_path(center_raw, len(raceline))

    # --- blended path ---
    path = w * center_resampled + (1 - w) * raceline

    # --- compute references ---
    v_ref = compute_reference_velocity(state, path, parameters)
    delta_ref = compute_reference_steering(state, path, parameters)

    return np.array([delta_ref, v_ref])


_prev_steering_error = 0.0

def lower_controller(state, desired, parameters):
    global _prev_steering_error

    delta_ref, v_ref = float(desired[0]), float(desired[1])
    v_delta, err = steering_controller(state, delta_ref, parameters, _prev_steering_error)
    _prev_steering_error = err

    a = velocity_controller(state, v_ref, parameters)
    return np.array([v_delta, a])
