from robomimic.scripts.config_gen.helper import *
import random
import json
import numpy as np
from collections import OrderedDict

#############################################################################
# *************** Replace with your paths/config information ****************

# Note: Assumes naming of dataset in "datasets" for the full DROID dataset is
# droid

DATA_PATH = "C:/Users/willi/tensorflow_datasets"    # UPDATE WITH PATH TO RLDS DATASETS
EXP_LOG_PATH = "C:/workspace/droid_policy_learning/logs" # UPDATE WITH PATH TO DESIRED LOGGING DIRECTORY
EXP_NAMES = OrderedDict(
    [
        # Note: you can add co-training dataset here appending
        # a new dataset to "datasets" and adjusting "sample_weights"
        # accordingly
        ("deligrasp", {"datasets": ["deligrasp_dataset"],
                   "sample_weights": [1]})                                    
    ])

#############################################################################

def make_generator_helper(args):
    algo_name_short = "diffusion_policy"

    generator = get_generator(
        algo_name="diffusion_policy",
        config_file=os.path.join(base_path, 'robomimic/exps/templates/diffusion_policy.json'),
        args=args,
        exp_log_path=EXP_LOG_PATH,
        algo_name_short=algo_name_short,
        pt=True,
    )
    if args.ckpt_mode is None:
        args.ckpt_mode = "off"

    generator.add_param(
        key="train.data_format",
        name="",
        group=-1,
        values=[
            "dg_rlds"
        ],
    )

    generator.add_param(
        key="train.num_epochs",
        name="",
        group=-1,
        values=[30],
    )

    generator.add_param(
        key="experiment.epoch_every_n_steps",
        name="",
        group=-1,
        values=[100],
    )

    generator.add_param(
        key="experiment.save.every_n_epochs",
        name="",
        group=-1,
        values=[10],
    )

    generator.add_param(
        key="train.data_path",
        name="",
        group=-1,
        values=[DATA_PATH],
    )

    generator.add_param(
        key="train.shuffle_buffer_size",
        name="",
        group=-1,
        values=[250],
    )

    generator.add_param(
        key="train.batch_size",
        name="bz",
        group=1212111,
        values=[16],
        hidename=False,
    )

    generator.add_param(
        key="train.subsample_length",
        name="subsample_length",
        group=7070707,
        values=[
            50
        ],
        hidename=True,
    )

    generator.add_param(
        key="train.num_parallel_calls",
        name="num_parallel_calls",
        group=404040404,
        values=[
            200
        ],
        hidename=True,
    )

    generator.add_param(
        key="train.traj_transform_threads",
        name="traj_transform_threads",
        group=303030303,
        values=[
            48
        ],
        hidename=True,
    )

    generator.add_param(
        key="train.traj_read_threads",
        name="traj_read_threads",
        group=908090809,
        values=[
            48
        ],
        hidename=True,
    )

    generator.add_param(
        key="algo.noise_samples",
        name="noise_samples",
        group=1010101,
        values=[8],
        value_names=["8"]
    )

    # use ddim by default
    generator.add_param(
        key="algo.ddim.enabled",
        name="ddim",
        group=1001,
        values=[
            True,
            # False,
        ],
        hidename=True,
    )
    generator.add_param(
        key="algo.ddpm.enabled",
        name="ddpm",
        group=1001,
        values=[
            False,
            # True,
        ],
        hidename=True,
    )
    if args.env == "droid":
        generator.add_param(
            key="train.action_config",
            name="",
            group=-1,
            values=[
                {
                    "action/cartesian_position":{
                        "normalization": "min_max",
                    },
                    "action/abs_pos":{
                        "normalization": "min_max",
                    },
                    "action/abs_rot_6d":{
                        "normalization": "min_max",
                        "format": "rot_6d",
                        "convert_at_runtime": "rot_euler",
                    },
                    "action/abs_rot_euler":{
                        "normalization": "min_max",
                        "format": "rot_euler",
                    },
                    "action/gripper_position":{
                        "normalization": None,
                    },
                    "action/gripper_force":{
                        "normalization": None,
                    },
                    "action/cartesian_velocity":{
                        "normalization": None,
                    },
                    "action/rel_pos":{
                        "normalization": None,
                    },
                    "action/rel_rot_6d":{
                        "format": "rot_6d",
                        "normalization": None,
                        "convert_at_runtime": "rot_euler",
                    },
                    "action/rel_rot_euler":{
                        "format": "rot_euler",
                        "normalization": None,
                    },
                    "action/gripper_velocity":{
                        "normalization": None,
                    },
                }
            ],
        )
        generator.add_param(
            key="train.sample_weights",
            name="sample_weights",
            group=24988,
            values=[
                EXP_NAMES[k]["sample_weights"] for k in EXP_NAMES.keys()
            ],
        )
        generator.add_param(
            key="train.dataset_names",
            name="dataset_names",
            group=24988,
            values=[
                EXP_NAMES[k]["datasets"] for k in EXP_NAMES.keys()
            ],
            value_names=list(EXP_NAMES.keys())
        )
        generator.add_param(
            key="train.action_keys",
            name="ac_keys",
            group=-1,
            values=[
                [
                    "action/rel_pos",
                    "action/rel_rot_6d",
                    "action/gripper_position",
                    "action/gripper_force",
                ],
            ],
            value_names=[
                "rel",
            ],
            hidename=True,
        )
        generator.add_param(
            key="train.action_shapes",
            name="ac_shapes",
            group=-1,
            values=[
                [
                    (1, 3),
                    (1, 6),
                    (1, 1),
                    (1, 1),
                ],
            ],
            value_names=[
                "ac_shapes",
            ],
            hidename=True,
        )
        generator.add_param(
            key="observation.image_dim",
            name="",
            group=-1,
            values=[
                [128, 128],
            ],
            hidename=True,
        )
        generator.add_param(
            key="observation.modalities.obs.rgb",
            name="cams",
            group=130,
            values=[
                ["camera/image/varied_camera_1_left_image", "camera/image/varied_camera_2_left_image"],
            ],
            value_names=[
                "workspace_wrist",
            ]
        )
        generator.add_param(
            key="observation.encoder.rgb.obs_randomizer_class",
            name="obsrand",
            group=130,
            values=[
                # "ColorRandomizer", # jitter only
                ["ColorRandomizer", "CropRandomizer"], # jitter, followed by crop
            ],
            hidename=True,
        )
        generator.add_param(
            key="observation.encoder.rgb.obs_randomizer_kwargs",
            name="obsrandargs",
            group=130,
            values=[
                # {}, # jitter only
                [{}, {"crop_height": 116, "crop_width": 116, "num_crops": 1, "pos_enc": False}], # jitter, followed by crop
            ],
            hidename=True,
        )

        ### CONDITIONING
        generator.add_param(
            key="train.goal_mode",
            name="goal_mode",
            group=24986,
            values = [
                # "geom",
                None, # Change this to "geom" to do goal conditioning

            ]
        )
        generator.add_param(
            key="train.truncated_geom_factor",
            name="truncated_geom_factor",
            group=5555,
            values = [
                0.3,
                # 0.5
            ]
        )
        generator.add_param(
            key="observation.modalities.obs.low_dim",
            name="ldkeys",
            group=24986,
            values=[
                ["robot_state/cartesian_position", "robot_state/gripper_position", 
                 "robot_state/applied_force", "robot_state/contact_force"],
            ],
            value_names=[
                "proprio-lang",
            ],
            hidename=False,
        )
        generator.add_param(
            key="observation.encoder.rgb.core_kwargs.backbone_kwargs.use_cam",
            name="",
            group=2498,
            values=[
                False,
                # True,
            ],
            hidename=True,
        )
        generator.add_param(
            key="observation.encoder.rgb.core_kwargs.backbone_kwargs.pretrained",
            name="",
            group=2498,
            values=[
                # False,
                True,
            ],
            hidename=True,
        )
        generator.add_param(
            key="observation.encoder.rgb.core_class",
            name="visenc",
            group=-1,
            values=["VisualCore"],
        )
        generator.add_param(
            key="observation.encoder.rgb.core_kwargs.backbone_class",
            name="",
            group=-1,
            values=["ResNet50Conv"],
            hidename=True,
        )
        generator.add_param(
            key="observation.encoder.rgb.core_kwargs.feature_dimension",
            name="visdim",
            group=1234,
            values=[
                512,
                # None,
                # None
            ],
            hidename=True,
        )
        generator.add_param(
            key="observation.encoder.rgb.core_kwargs.pool_class",
            name="poolclass",
            group=1234,
            values=[
                # "SpatialSoftmax",
                None,
                # None
            ],
            hidename=True,
        )

        generator.add_param(
            key="observation.encoder.rgb.core_kwargs.flatten",
            name="flatten",
            group=1234,
            values=[
                True,
                # False,
                # False
            ],
            hidename=True,
        )
        generator.add_param(
            key="observation.encoder.rgb.fuser",
            name="fuser",
            group=1234,
            values=[
                None,
                # "transformer",
                # "perceiver"
            ],
            hidename=False,
        )
        generator.add_param(
            key="observation.encoder.rgb.core_kwargs.backbone_kwargs.downsample",
            name="",
            group=1234,
            values=[
                False,
            ],
            hidename=False,
        )
    else:
        raise ValueError
    
    generator.add_param(
        key="train.output_dir",
        name="",
        group=-1,
        values=[
            "{exp_log_path}/{env}/{mod}/{algo_name_short}".format(
                exp_log_path=EXP_LOG_PATH,
                env=args.env,
                mod=args.mod, 
                algo_name_short=algo_name_short,
            )
        ],
    )

    return generator

if __name__ == "__main__":
    parser = get_argparser()

    args = parser.parse_args()
    make_generator(args, make_generator_helper)