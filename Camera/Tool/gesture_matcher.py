import os
import numpy as np

# Gesture matching utilities for ManuMano (template + LSTM fusion).

POSE_SIZE = 33 * 4
HAND_SIZE = 21 * 3
FEATURE_SIZE = POSE_SIZE + HAND_SIZE + HAND_SIZE


def split_hands_by_handedness(hand_landmarks, handedness):
    if not hand_landmarks:
        return None, None

    left = None
    right = None
    unknown = []

    for i, hand in enumerate(hand_landmarks):
        label = ""
        score = 0.0

        if handedness and i < len(handedness) and handedness[i]:
            category = handedness[i][0]
            label = (category.category_name or "").lower()
            score = float(category.score)

        if label == "left":
            if left is None or score > left[1]:
                left = (hand, score)
            else:
                unknown.append(hand)
        elif label == "right":
            if right is None or score > right[1]:
                right = (hand, score)
            else:
                unknown.append(hand)
        else:
            unknown.append(hand)

    if unknown:
        unknown = sorted(unknown, key=lambda h: h[0].x)
        for hand in unknown:
            if left is None:
                left = (hand, 0.0)
            elif right is None:
                right = (hand, 0.0)

    return (left[0] if left else None), (right[0] if right else None)


def _hand_to_vec(hand):
    if hand is None:
        return np.zeros(HAND_SIZE, dtype=np.float32)

    return np.array([[lm.x, lm.y, lm.z] for lm in hand], dtype=np.float32).flatten()


def extract_keypoints(pose_landmarks, left_hand, right_hand):
    pose = np.array(
        [
            [lm.x, lm.y, lm.z, getattr(lm, "visibility", 0)]
            for pose in (pose_landmarks or [])
            for lm in pose
        ],
        dtype=np.float32,
    ).flatten()

    if pose.size == 0:
        pose = np.zeros(POSE_SIZE, dtype=np.float32)

    lh = _hand_to_vec(left_hand)
    rh = _hand_to_vec(right_hand)
    return np.concatenate([pose, lh, rh]).astype(np.float32)


def _split_frame(frame_vec):
    pose = frame_vec[:POSE_SIZE].reshape(33, 4)
    lh = frame_vec[POSE_SIZE : POSE_SIZE + HAND_SIZE].reshape(21, 3)
    rh = frame_vec[POSE_SIZE + HAND_SIZE :].reshape(21, 3)
    return pose, lh, rh


def _safe_face_points(pose):
    nose = pose[0, :3]
    mouth = (pose[9, :3] + pose[10, :3]) * 0.5
    left_ear = pose[7, :3]
    right_ear = pose[8, :3]

    if np.linalg.norm(nose) < 1e-6 and np.linalg.norm(mouth) < 1e-6:
        nose = np.array([0.5, 0.5, 0.0], dtype=np.float32)
        mouth = np.array([0.5, 0.55, 0.0], dtype=np.float32)
        left_ear = np.array([0.35, 0.5, 0.0], dtype=np.float32)
        right_ear = np.array([0.65, 0.5, 0.0], dtype=np.float32)

    return nose, mouth, left_ear, right_ear


def _frame_signature(frame_vec):
    pose, lh, rh = _split_frame(frame_vec)
    nose, mouth, left_ear, right_ear = _safe_face_points(pose)

    lw = lh[0]
    rw = rh[0]

    l_tips = lh[[4, 8, 12, 16, 20]]
    r_tips = rh[[4, 8, 12, 16, 20]]

    l_shape = (l_tips - lw).flatten()
    r_shape = (r_tips - rw).flatten()

    face_rel = np.concatenate(
        [
            lw - mouth,
            rw - mouth,
            lw - left_ear,
            rw - right_ear,
        ]
    )

    return np.concatenate(
        [
            l_shape,
            r_shape,
            face_rel,
            lw - rw,
            nose,
            mouth,
        ]
    ).astype(np.float32)


def sequence_signature(sequence):
    base = np.array([_frame_signature(frame) for frame in sequence], dtype=np.float32)
    velocity = np.diff(base, axis=0, prepend=base[:1])
    return np.concatenate([base, 0.45 * velocity], axis=1)


def sequence_profile(sequence):
    left_traj = []
    right_traj = []
    mouth_traj = []
    left_ear_traj = []
    right_ear_traj = []

    for frame_vec in sequence:
        pose, lh, rh = _split_frame(frame_vec)
        _, mouth, left_ear, right_ear = _safe_face_points(pose)

        left_traj.append(lh[0])
        right_traj.append(rh[0])
        mouth_traj.append(mouth)
        left_ear_traj.append(left_ear)
        right_ear_traj.append(right_ear)

    left_traj = np.array(left_traj, dtype=np.float32)
    right_traj = np.array(right_traj, dtype=np.float32)
    mouth_traj = np.array(mouth_traj, dtype=np.float32)
    left_ear_traj = np.array(left_ear_traj, dtype=np.float32)
    right_ear_traj = np.array(right_ear_traj, dtype=np.float32)

    left_path = float(np.sum(np.linalg.norm(np.diff(left_traj, axis=0), axis=1)))
    right_path = float(np.sum(np.linalg.norm(np.diff(right_traj, axis=0), axis=1)))

    if left_path >= right_path:
        dom = left_traj
    else:
        dom = right_traj

    path = float(np.sum(np.linalg.norm(np.diff(dom, axis=0), axis=1)))
    displacement = float(np.linalg.norm(dom[-1] - dom[0]))

    mouth_d = np.linalg.norm(dom - mouth_traj, axis=1)
    ear_d = np.minimum(
        np.linalg.norm(dom - left_ear_traj, axis=1),
        np.linalg.norm(dom - right_ear_traj, axis=1),
    )

    min_mouth = float(np.min(mouth_d))
    min_ear = float(np.min(ear_d))

    return np.array([path, displacement, min_mouth, min_ear], dtype=np.float32)


def sequence_distance(sig_a, sig_b):
    return float(np.mean(np.linalg.norm(sig_a - sig_b, axis=1)))


def _load_action_sequences(action_path, sequence_length):
    seq_map = {}

    for name in os.listdir(action_path):
        if not name.endswith(".npy"):
            continue

        stem = os.path.splitext(name)[0]
        parts = stem.split("_")
        if len(parts) != 2:
            continue

        try:
            seq_id = int(parts[0])
            frame_id = int(parts[1])
        except ValueError:
            continue

        file_path = os.path.join(action_path, name)
        vec = np.load(file_path)

        if vec.shape[0] != FEATURE_SIZE:
            continue

        seq_map.setdefault(seq_id, {})[frame_id] = vec.astype(np.float32)

    sequences = []
    for seq_id in sorted(seq_map.keys()):
        frames = seq_map[seq_id]
        if not all(i in frames for i in range(sequence_length)):
            continue

        seq = np.array([frames[i] for i in range(sequence_length)], dtype=np.float32)
        sequences.append(seq)

    return sequences


def _template_cache_fingerprint(data_path, actions, sequence_length, max_templates_per_action):
    parts = [f"seq={sequence_length}", f"k={max_templates_per_action}"]

    for action in sorted(actions):
        action_path = os.path.join(data_path, action)
        file_count = 0
        newest = 0.0

        if os.path.isdir(action_path):
            try:
                for entry in os.scandir(action_path):
                    if not entry.is_file() or not entry.name.endswith(".npy"):
                        continue

                    file_count += 1
                    try:
                        newest = max(newest, float(entry.stat().st_mtime))
                    except OSError:
                        continue
            except OSError:
                pass

        parts.append(f"{action}:{file_count}:{newest:.6f}")

    return "|".join(parts)


def _load_template_cache(cache_path, fingerprint, sequence_length):
    if not os.path.isfile(cache_path):
        return None

    try:
        blob = np.load(cache_path, allow_pickle=True)
    except Exception:
        return None

    try:
        if "fingerprint" not in blob or str(blob["fingerprint"][0]) != fingerprint:
            return None

        actions = blob["actions"].tolist()
        sigs = blob["sigs"].tolist()
        profiles = blob["profiles"].tolist()

        template_signatures = {}
        template_profiles = {}

        for action, action_sigs, action_profile in zip(actions, sigs, profiles):
            if action_sigs is None:
                continue

            try:
                sig_list = list(action_sigs)
            except TypeError:
                sig_list = []

            clean_sigs = []
            for sig in sig_list:
                arr = np.asarray(sig, dtype=np.float32)
                if arr.ndim == 2 and arr.shape[0] == sequence_length:
                    clean_sigs.append(arr)

            if clean_sigs:
                template_signatures[action] = clean_sigs

            if action_profile is not None:
                profile_arr = np.asarray(action_profile, dtype=np.float32)
                if profile_arr.size == 4:
                    template_profiles[action] = profile_arr

        if template_signatures:
            return template_signatures, template_profiles
    except Exception:
        return None

    return None


def _save_template_cache(cache_path, fingerprint, actions, template_signatures, template_profiles):
    try:
        actions_arr = np.array(actions, dtype=object)
        sigs_arr = np.empty(len(actions), dtype=object)
        profiles_arr = np.empty(len(actions), dtype=object)

        for i, action in enumerate(actions):
            sigs_arr[i] = template_signatures.get(action, [])
            profiles_arr[i] = template_profiles.get(action, None)

        np.savez_compressed(
            cache_path,
            fingerprint=np.array([fingerprint], dtype=object),
            actions=actions_arr,
            sigs=sigs_arr,
            profiles=profiles_arr,
        )
    except Exception:
        pass


def load_action_templates(data_path, actions, sequence_length, max_templates_per_action=10):
    template_signatures = {}
    template_profiles = {}

    fingerprint = _template_cache_fingerprint(
        data_path=data_path,
        actions=actions,
        sequence_length=sequence_length,
        max_templates_per_action=max_templates_per_action,
    )
    cache_path = os.path.join(
        data_path,
        f"_template_cache_s{sequence_length}_k{max_templates_per_action}.npz",
    )

    cached = _load_template_cache(cache_path, fingerprint, sequence_length)
    if cached is not None:
        return cached

    for action in actions:
        action_path = os.path.join(data_path, action)
        if not os.path.isdir(action_path):
            continue

        sequences = _load_action_sequences(action_path, sequence_length)
        if not sequences:
            continue

        sequences = sequences[-max_templates_per_action:]
        sigs = [sequence_signature(seq) for seq in sequences]
        profiles = np.array([sequence_profile(seq) for seq in sequences], dtype=np.float32)

        template_signatures[action] = sigs
        template_profiles[action] = np.mean(profiles, axis=0)

    _save_template_cache(cache_path, fingerprint, actions, template_signatures, template_profiles)

    return template_signatures, template_profiles

def match_sequence(
    sequence,
    model_probs,
    actions,
    template_signatures,
    template_profiles,
    model_weight=0.35,
    template_weight=0.65,
):
    if len(sequence) == 0:
        return {
            "action": actions[0] if actions else "",
            "combined": 0.0,
            "model_prob": 0.0,
            "template_dist": 999.0,
        }

    model_probs = np.array(model_probs, dtype=np.float32)
    usable = min(len(actions), model_probs.shape[0])

    if usable == 0:
        return {
            "action": "",
            "combined": 0.0,
            "model_prob": 0.0,
            "template_dist": 999.0,
        }

    query_seq = np.asarray(sequence, dtype=np.float32)
    query_sig = sequence_signature(query_seq)
    query_profile = sequence_profile(query_seq)

    best_action = actions[int(np.argmax(model_probs[:usable]))]
    best_combined = -1.0
    best_model = 0.0
    best_dist = 999.0

    for i in range(usable):
        action = actions[i]
        model_prob = float(model_probs[i])

        refs = template_signatures.get(action, None)
        profile_ref = template_profiles.get(action, None)

        if refs:
            point_dist = min(sequence_distance(query_sig, ref_sig) for ref_sig in refs)
        else:
            point_dist = 1.0

        profile_dist = 0.0
        if profile_ref is not None:
            denom = np.maximum(np.abs(profile_ref), 1e-4)
            profile_dist = float(np.mean(np.abs(query_profile - profile_ref) / denom))

        template_dist = point_dist + (0.35 * profile_dist)

        if profile_ref is not None:
            expects_contact = profile_ref[2] < 0.18
            expects_motion = profile_ref[0] > 0.18

            if expects_contact and query_profile[2] > profile_ref[2] + 0.08:
                template_dist += 0.08
            if expects_motion and query_profile[0] < profile_ref[0] * 0.55:
                template_dist += 0.08

        template_sim = 1.0 / (1.0 + 6.0 * template_dist)
        combined = (model_weight * model_prob) + (template_weight * template_sim)

        if combined > best_combined:
            best_combined = combined
            best_action = action
            best_model = model_prob
            best_dist = template_dist

    return {
        "action": best_action,
        "combined": float(best_combined),
        "model_prob": float(best_model),
        "template_dist": float(best_dist),
    }

