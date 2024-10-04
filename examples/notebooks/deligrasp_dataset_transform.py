def euler_to_rmat(euler):
    return tfg.rotation_matrix_3d.from_euler(euler)

def mat_to_rot6d(mat):
    r6 = mat[..., :2, :]
    r6_0, r6_1 = r6[..., 0, :], r6[..., 1, :]
    r6_flat = tf.concat([r6_0, r6_1], axis=-1)
    return r6_flat

def deligrasp_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # every input feature is batched, ie has leading batch dimension
    T = trajectory["action"][:, :3]
    R = mat_to_rot6d(euler_to_rmat(trajectory["action"][:, 3:6]))
    trajectory["action"] = tf.concat(
        (
            T,
            R,
            tf.expand_dims(trajectory["action"][:, 6], axis=-1), # delta gripper position
            tf.expand_dims(trajectory["action"][:, 7], axis=-1), # delta applied force
        ),
        axis=-1,
    )
    return trajectory
