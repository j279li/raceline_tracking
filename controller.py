import numpy as np
from numpy.typing import ArrayLike

from simulator import RaceTrack

# ============================================================================
# Helper functions
# ============================================================================

def find_closest_point(state: ArrayLike, path: ArrayLike) -> tuple[int, float]:
    """
    Find the closest point on the path to the car position.
    state: [sx, sy, δ, v, ϕ]
    path:  N x 2 array of [x, y] points
    """
    car_pos = state[0:2]
    distances = np.linalg.norm(path - car_pos, axis=1)
    idx = int(np.argmin(distances))
    return idx, float(distances[idx])


def find_lookahead_point(
    state: ArrayLike,
    path: ArrayLike,
    lookahead_distance: float
) -> tuple[int, ArrayLike]:
    """
    Walk forward along the path from the closest point until we accumulate
    approximately lookahead_distance in arc length.
    """
    closest_idx, _ = find_closest_point(state, path)
    cumulative_dist = 0.0
    lookahead_idx = closest_idx

    for i in range(1, len(path)):
        prev_idx = (closest_idx + i - 1) % len(path)
        next_idx = (closest_idx + i) % len(path)
        segment_dist = np.linalg.norm(path[next_idx] - path[prev_idx])
        cumulative_dist += segment_dist

        if cumulative_dist >= lookahead_distance:
            lookahead_idx = next_idx
            break

    return lookahead_idx, path[lookahead_idx]


def estimate_curvature(path: ArrayLike, idx: int, step: int = 5) -> float:
    """
    Estimate curvature at path index using Menger curvature formula.
    Returns curvature in 1/meters (higher = tighter corner).
    """
    N = len(path)
    i1 = (idx - step) % N
    i2 = idx % N
    i3 = (idx + step) % N

    p1 = path[i1]
    p2 = path[i2]
    p3 = path[i3]

    area = 0.5 * abs(
        (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])
    )

    a = np.linalg.norm(p2 - p1)
    b = np.linalg.norm(p3 - p2)
    c = np.linalg.norm(p1 - p3)

    denom = a * b * c
    if denom < 1e-9:
        return 0.0

    curvature = 4.0 * area / denom
    return float(curvature)


# ============================================================================
# Reference generators: S1 (velocity) and S2 (steering angle)
# ============================================================================

def compute_safe_corner_speed(
    curvature: float,
    v_max: float,
    raceline_mode: bool = False
) -> float:
    """
    Calculate safe corner speed with curvature-dependent lateral acceleration.

    raceline_mode=True → allow higher lateral acceleration (more grip / risk).
    """
    if curvature < 1e-6:
        return v_max

    if raceline_mode:
        # Raceline: slightly higher allowable lateral g
        if curvature < 0.008:
            a_lateral_max = 19.0  # m/s²
        elif curvature < 0.020:
            a_lateral_max = 16.5
        elif curvature < 0.035:
            a_lateral_max = 14.0
        else:
            a_lateral_max = 11.0
    else:
        # Centerline: a bit more conservative
        if curvature < 0.008:
            a_lateral_max = 17.0  # m/s²
        elif curvature < 0.020:
            a_lateral_max = 14.5
        elif curvature < 0.035:
            a_lateral_max = 12.0
        else:
            a_lateral_max = 9.5

    v_safe = np.sqrt(a_lateral_max / curvature)
    return min(v_safe, v_max)


def compute_braking_distance(v_current: float, v_target: float, a_brake: float) -> float:
    """
    Calculate braking distance needed to slow from v_current to v_target.

    Physics: d = (v_current² - v_target²) / (2 * a_brake)
    """
    if v_current <= v_target:
        return 0.0

    distance = (v_current**2 - v_target**2) / (2.0 * a_brake)
    return max(distance, 0.0)


def compute_reference_velocity(
    state: ArrayLike,
    path: ArrayLike,
    parameters: ArrayLike,
    raceline_mode: bool = False
) -> float:
    """
    S1: Physics-based velocity planning.

    1. Scan ahead on path to find tightest upcoming corner.
    2. Compute safe speed from curvature (with raceline vs centerline limits).
    3. Use braking distance to decide if we must slow down now.
    """
    v_min = float(parameters[2])
    v_max = float(parameters[5])
    a_max = float(parameters[10])

    current_speed = abs(float(state[3]))
    closest_idx, _ = find_closest_point(state, path)

    base_lookahead = 40
    if current_speed > 60:
        extra = ((current_speed - 60) ** 1.3) / 2.0
        lookahead_points = int(min(base_lookahead + current_speed / 2.5 + extra, 100))
    else:
        lookahead_points = int(min(base_lookahead + current_speed / 2.5, 80))

    min_safe_speed = v_max
    distance_to_corner = 0.0
    found_corner = False
    last_corner_sign = None
    last_corner_distance = None

    cumulative_distance = 0.0
    for i in range(lookahead_points):
        check_idx = (closest_idx + i) % len(path)

        if i > 0:
            prev_idx = (closest_idx + i - 1) % len(path)
            seg_len = np.linalg.norm(path[check_idx] - path[prev_idx])
            cumulative_distance += seg_len

        curv = estimate_curvature(path, check_idx, step=3)

        if curv > 0.0008:  # catch even slight curves
            safe_speed = compute_safe_corner_speed(
                curv, v_max, raceline_mode=raceline_mode
            )

            # Determine turn direction for S-curve detection
            turn_sign = 0.0
            if curv > 0.010:
                prev_idx = (check_idx - 1) % len(path)
                next_idx = (check_idx + 1) % len(path)
                v1 = path[check_idx] - path[prev_idx]
                v2 = path[next_idx] - path[check_idx]
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                if abs(cross) > 1e-9:
                    turn_sign = np.sign(cross)

            # Extra safety margins, lighter for raceline
            if raceline_mode:
                if curv > 0.030:
                    safe_speed *= 0.80
                elif curv > 0.020:
                    safe_speed *= 0.86
                elif curv > 0.012:
                    safe_speed *= 0.92
                elif curv > 0.006:
                    safe_speed *= 0.96
                elif curv > 0.003:
                    safe_speed *= 0.985
                else:
                    safe_speed *= 0.998
            else:
                if curv > 0.030:
                    safe_speed *= 0.76
                elif curv > 0.020:
                    safe_speed *= 0.82
                elif curv > 0.012:
                    safe_speed *= 0.88
                elif curv > 0.006:
                    safe_speed *= 0.94
                elif curv > 0.003:
                    safe_speed *= 0.97
                else:
                    safe_speed *= 0.995

            # Mild S-curve penalty ONLY for centerline mode
            if (
                not raceline_mode
                and turn_sign != 0.0
                and last_corner_sign is not None
                and last_corner_distance is not None
            ):
                gap = cumulative_distance - last_corner_distance
                if (turn_sign * last_corner_sign) < 0 and gap < 40.0:
                    safe_speed *= 0.94

            if turn_sign != 0.0:
                last_corner_sign = turn_sign
                last_corner_distance = cumulative_distance

            if safe_speed < min_safe_speed:
                min_safe_speed = safe_speed
                distance_to_corner = cumulative_distance
                found_corner = True

    if not found_corner or min_safe_speed >= current_speed:
        v_ref = v_max
    else:
        brake_accel = 0.85 * a_max
        required_brake_distance = compute_braking_distance(
            current_speed, min_safe_speed, brake_accel
        )

        safety_margin = 1.22
        required_brake_distance *= safety_margin

        if distance_to_corner <= required_brake_distance:
            v_ref = min_safe_speed
            # otherwise maintain or go fast, as before
        elif distance_to_corner < required_brake_distance * 1.3:
            v_ref = current_speed
        else:
            v_ref = v_max

    return float(np.clip(v_ref, v_min, v_max))


def compute_reference_steering(
    state: ArrayLike,
    centerline: ArrayLike,
    parameters: ArrayLike,
    raceline_mode: bool = False
) -> float:
    """
    S2: Pure pursuit steering plus a light Stanley-style cross-track correction.
    Lookahead adapts to speed; raceline_mode uses slightly shorter lookahead.
    """
    wheelbase = float(parameters[0])
    delta_max = float(parameters[4])

    v = float(state[3])
    speed = max(abs(v), 1.0)

    # Speed-based lookahead
    if speed < 50:
        lookahead_distance = 8.0 + 0.30 * speed
    else:
        lookahead_distance = 23.0 + 0.45 * (speed - 50)

    if raceline_mode:
        lookahead_distance *= 0.9

    _, lookahead_point = find_lookahead_point(state, centerline, lookahead_distance)

    sx, sy = state[0], state[1]
    heading = state[4]

    dx = lookahead_point[0] - sx
    dy = lookahead_point[1] - sy

    cos_h = np.cos(-heading)
    sin_h = np.sin(-heading)

    lx = dx * cos_h - dy * sin_h
    ly = dx * sin_h + dy * cos_h

    if lx < 0.5:
        lx = 0.5

    Ld = np.hypot(lx, ly)
    if Ld < 1.0:
        Ld = 1.0

    alpha = np.arctan2(ly, lx)

    pure_pursuit_steer = np.arctan2(2.0 * wheelbase * np.sin(alpha), Ld)

    # Stanley correction
    closest_idx, _ = find_closest_point(state, centerline)
    next_idx = (closest_idx + 1) % len(centerline)

    path_vec = centerline[next_idx] - centerline[closest_idx]
    path_len = np.linalg.norm(path_vec)
    if path_len > 1e-6:
        path_vec = path_vec / path_len
    else:
        path_vec = np.array([1.0, 0.0])

    to_car = np.array([sx, sy]) - centerline[closest_idx]

    cross_track_error = path_vec[0] * to_car[1] - path_vec[1] * to_car[0]

    k_stanley = 0.8 if raceline_mode else 0.5
    stanley_correction = np.arctan(
        k_stanley * cross_track_error / max(abs(speed), 2.0)
    )

    correction_weight = 0.22 if raceline_mode else 0.15
    delta_ref = pure_pursuit_steer + correction_weight * stanley_correction

    delta_ref = float(np.clip(delta_ref, -0.9 * delta_max, 0.9 * delta_max))
    return delta_ref


# ============================================================================
# Controllers: C1 (velocity) and C2 (steering rate)
# ============================================================================

def velocity_controller(
    state: ArrayLike,
    reference_velocity: float,
    parameters: ArrayLike
) -> float:
    """
    C1: Longitudinal controller.
    a = Kp * (v_ref - v), clipped to ±max_acceleration.
    """
    v = float(state[3])
    a_max = float(parameters[10])

    v_ref = float(reference_velocity)
    v_error = v_ref - v

    if v_error < 0:  # braking
        if v < 50:
            kp_v = 4.5
        elif v > 90:
            kp_v = 5.8
        else:
            alpha = (v - 50) / 40.0
            kp_v = 4.5 + alpha * 1.3
    else:  # accelerating
        kp_v = 3.5

    a_cmd = kp_v * v_error
    a_cmd = float(np.clip(a_cmd, -a_max, a_max))
    return a_cmd


def steering_controller(
    state: ArrayLike,
    reference_steering: float,
    parameters: ArrayLike,
    prev_error: float = 0.0,
    dt: float = 0.1
) -> tuple[float, float]:
    """
    C2: Lateral low level controller with PD control on steering angle.
    """
    delta = float(state[2])
    delta_ref = float(reference_steering)
    v = float(state[3])

    v_delta_min = float(parameters[7])
    v_delta_max = float(parameters[9])

    error = delta_ref - delta
    d_error = (error - prev_error) / dt

    speed = abs(v)
    if speed < 20:
        kp_delta = 2.4
        kd_delta = 0.35
    elif speed > 70:
        kp_delta = 2.1
        kd_delta = 0.85
    else:
        alpha = (speed - 20) / 50.0
        kp_delta = 2.4 - alpha * 0.3
        kd_delta = 0.35 + alpha * 0.5

    v_delta_cmd = kp_delta * error + kd_delta * d_error
    v_delta_cmd = float(np.clip(v_delta_cmd, v_delta_min, v_delta_max))

    return v_delta_cmd, error


# ============================================================================
# High level and low level controller interfaces
# ============================================================================

def compute_safe_raceline(
    raceline: ArrayLike,
    racetrack: RaceTrack,
    safety_margin: float = 0.7
) -> ArrayLike:
    """
    (Currently UNUSED in controller)
    Compute a "safe" raceline that maintains a margin from track boundaries.
    """
    safe_raceline = raceline.copy()

    for i in range(len(raceline)):
        point = raceline[i]

        centerline_dists = np.linalg.norm(racetrack.centerline - point, axis=1)
        closest_center_idx = np.argmin(centerline_dists)

        right_boundary = racetrack.right_boundary[closest_center_idx]
        left_boundary = racetrack.left_boundary[closest_center_idx]
        center_point = racetrack.centerline[closest_center_idx]

        dist_to_right = np.linalg.norm(point - right_boundary)
        dist_to_left = np.linalg.norm(point - left_boundary)

        if dist_to_right < safety_margin:
            direction_to_center = center_point - point
            direction_to_center /= (np.linalg.norm(direction_to_center) + 1e-6)
            safe_raceline[i] = point + direction_to_center * (safety_margin - dist_to_right)
        elif dist_to_left < safety_margin:
            direction_to_center = center_point - point
            direction_to_center /= (np.linalg.norm(direction_to_center) + 1e-6)
            safe_raceline[i] = point + direction_to_center * (safety_margin - dist_to_left)

    return safe_raceline


def controller(
    state: ArrayLike,
    parameters: ArrayLike,
    racetrack: RaceTrack,
    raceline: ArrayLike | None = None
) -> ArrayLike:
    """
    High level controller:
      - Uses raceline if provided, otherwise centerline.
      - Raceline mode allows higher lateral g and slightly tighter tracking.
    """
    using_raceline = raceline is not None
    if using_raceline:
        path = raceline          # use raw raceline (no reshaping)
    else:
        path = racetrack.centerline

    v_ref = compute_reference_velocity(state, path, parameters, raceline_mode=using_raceline)
    delta_ref = compute_reference_steering(state, path, parameters, raceline_mode=using_raceline)

    return np.array([delta_ref, v_ref])


_prev_steering_error = 0.0

def lower_controller(
    state: ArrayLike,
    desired: ArrayLike,
    parameters: ArrayLike
) -> ArrayLike:
    """
    Low level controller:
      desired = [δ_r, v_ref]
      output  = [v_δ, a]
    """
    global _prev_steering_error

    assert desired.shape == (2,)

    delta_ref = float(desired[0])
    v_ref = float(desired[1])

    v_delta, error = steering_controller(
        state, delta_ref, parameters, _prev_steering_error
    )
    _prev_steering_error = error

    a = velocity_controller(state, v_ref, parameters)

    return np.array([v_delta, a])
