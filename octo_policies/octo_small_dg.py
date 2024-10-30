from absl import app, flags, logging
import flax
import jax
import optax
import tensorflow as tf
import tqdm
import wandb

from octo.data.dataset import make_single_dataset
from octo.model.components.action_heads import L1ActionHead, DiffusionActionHead
from octo.model.components.tokenizers import LowdimObsTokenizer
from octo.model.octo_model import OctoModel
from octo.utils.jax_utils import initialize_compilation_cache
from octo.utils.spec import ModuleSpec
from octo.utils.train_utils import (
    freeze_weights,
    merge_params,
    process_text,
    TrainState,
)

DATA_PATH = "C:/Users/willi/tensorflow_datasets/"    # UPDATE WITH PATH TO RLDS DATASETS
EXP_LOG_PATH = "C:/workspace/deligrasp_policy_learning/logs/octo" # UPDATE WITH PATH TO DESIRED LOGGING DIRECTORY
OCTO_CKPT_SMALL = "C:/Users/willi/.cache/huggingface/hub/models--rail-berkeley--octo-small-1.5/snapshots/dc9aa3019f764726c770814b27e4ab0fc6e32a58"
OCTO_CKPT_BASE = "C:/Users/willi/.cache/huggingface/hub/models--rail-berkeley--octo-base-1.5/snapshots/ee3c10e8edd6ce2e8b1e8744d3c6fba4097bed48"

DATA_PATH = "/mnt/c/Users/willi/tensorflow_datasets/"
EXP_LOG_PATH = "/mnt/c/workspace/deligrasp_policy_learning/logs/octo"
OCTO_CKPT_SMALL = "/mnt/c/Users/willi/.cache/huggingface/hub/models--rail-berkeley--octo-small-1.5/snapshots/dc9aa3019f764726c770814b27e4ab0fc6e32a58"
OCTO_CKPT_BASE = "/mnt/c/Users/willi/.cache/huggingface/hub/models--rail-berkeley--octo-base-1.5/snapshots/ee3c10e8edd6ce2e8b1e8744d3c6fba4097bed48"

FLAGS = flags.FLAGS

flags.DEFINE_string("f", "", "notebook path hack.")
flags.DEFINE_string(
    "pretrained_path",OCTO_CKPT_SMALL, "Path to pre-trained Octo checkpoint directory."
)
flags.DEFINE_string("data_dir", DATA_PATH, "Path to finetuning dataset, in RLDS format.")
flags.DEFINE_string("save_dir", EXP_LOG_PATH, "Directory for saving finetuning checkpoints.")
# flags.DEFINE_integer("verbosity", 0, "0 is info, 1 is debug. not having this flag is breaking jax")
flags.DEFINE_integer("batch_size", 16, "Batch size for finetuning.")
flags.DEFINE_bool(
    "freeze_transformer",
    True,
    "Whether pre-trained transformer weights should be frozen.",
)
# import sys
# FLAGS(sys.argv)

def main(_):
    assert (
        FLAGS.batch_size % jax.device_count() == 0
    ), "Batch size must be divisible by device count."

    initialize_compilation_cache()
    # prevent tensorflow from using GPU memory since it's only used for data loading
    tf.config.set_visible_devices([], "GPU")

    # setup wandb for logging
    wandb.init(name="octo_sm_dg", project="jaf")

    # load pre-trained model
    logging.info("Loading pre-trained model...")
    pretrained_model = OctoModel.load_pretrained(FLAGS.pretrained_path)

    # make finetuning dataset
    # apply Gaussian normalization, load chunks of 50 actions since we'll train with action chunking
    # delete goal images in the data loader since we will train a language-conditioned-only policy
    # TODO: directly load this from raw data to make it less opaque?
    logging.info("Loading finetuning dataset...")
    dataset = make_single_dataset(
        dataset_kwargs=dict(
            name="deligrasp_dataset",
            data_dir=FLAGS.data_dir,
            image_obs_keys={"primary": "image", "wrist": "wrist_image"},
            proprio_obs_key="state",
            language_key="language_instruction",
        ),
        traj_transform_kwargs=dict(
            window_size=2,
            action_horizon=8,
            subsample_length=20,

        ),
        frame_transform_kwargs=dict(
            resize_size={"primary": (256, 256), "wrist": (128, 128)},
        ),
        train=True,
    )
    train_data_iter = (
        dataset.repeat()
        .unbatch()
        .shuffle(100)  # can reduce this if RAM consumption too high
        .batch(FLAGS.batch_size)
        .iterator()
    )

    # run text tokenizer over batch (this needs to happen before training / sharding) + delete unused keys
    text_processor = pretrained_model.text_processor

    def process_batch(batch):
        batch = process_text(batch, text_processor)
        del batch["dataset_name"]
        return batch

    train_data_iter = map(process_batch, train_data_iter)
    example_batch = next(train_data_iter)

    # load pre-training config and modify --> remove wrist cam, add proprio input, change action head
    # following Zhao et al. we use "action chunks" of length 50 and L1 loss for ALOHA
    config = pretrained_model.config
    del config["model"]["observation_tokenizers"]["wrist"]
    ###
    config = pretrained_model.config
    # config["model"]["heads"]["action"]["kwargs"]["action_dim"] = 8
    # Fully override the old action head with a new one (for smaller changes, you can use update_config)
    config["model"]["heads"]["action"] = ModuleSpec.create(
        DiffusionActionHead,
        use_map=False,
        action_horizon=8,
        action_dim=9,
        readout_key="readout_action",
    )

    # initialize weights for modified Octo model, then merge in all applicable pre-trained weights
    # new position encodings for proprio inputs & weights for new action head will remain "from scratch"
    logging.info("Updating model for new observation & action space...")
    model = OctoModel.from_config(
        config,
        example_batch,
        text_processor,
        verbose=True,
        dataset_statistics=dataset.dataset_statistics,
    )
    merged_params = merge_params(model.params, pretrained_model.params)
    # can perform any additional parameter surgery here...
    # ...
    model = model.replace(params=merged_params)
    del pretrained_model

    # create optimizer & train_state, optionally freeze keys for pre-trained transformer
    # train_state bundles parameters & optimizers
    learning_rate = optax.join_schedules(
        [optax.linear_schedule(0, 3e-5, 100), optax.constant_schedule(3e-5)], [100]
    )
    tx = optax.adamw(learning_rate)
    frozen_keys = model.config["optimizer"]["frozen_keys"]
    if FLAGS.freeze_transformer:
        frozen_keys.append("BlockTransformer_0")
    tx = freeze_weights(tx, model.params, frozen_keys)
    train_state = TrainState.create(
        rng=jax.random.PRNGKey(1234),
        model=model,
        tx=tx,
    )

    # define loss function and train step
    def loss_fn(params, batch, rng, train=True):
        bound_module = model.module.bind({"params": params}, rngs={"dropout": rng})
        transformer_embeddings = bound_module.octo_transformer(
            batch["observation"],
            batch["task"],
            batch["observation"]["timestep_pad_mask"],
            train=train,
        )
        action_loss, action_metrics = bound_module.heads["action"].loss(
            transformer_embeddings,  # Action head knows to pull out the action readout_key
            batch["action"],
            batch["observation"]["timestep_pad_mask"],
            batch["action_pad_mask"],
            train=train,
        )
        return action_loss, action_metrics

    @jax.jit
    def train_step(state, batch):
        rng, dropout_rng = jax.random.split(state.rng)
        (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.model.params, batch, dropout_rng, train=True
        )
        new_state = state.apply_gradients(grads=grads, rng=rng)
        return new_state, info

    # run finetuning loop
    n_epochs = 30
    n_steps = 100
    total_steps = n_epochs * n_steps
    save_every_n_epochs = 10
    save_every_n_steps = save_every_n_epochs * n_steps
    logging.info("Starting finetuning...")
    for i in tqdm.tqdm(range(total_steps), total=total_steps, dynamic_ncols=True):
        batch = next(train_data_iter)
        train_state, update_info = train_step(train_state, batch)
        if (i + 1) % n_steps == 0:
            update_info = jax.device_get(update_info)
            wandb.log(
                flax.traverse_util.flatten_dict({"training": update_info}, sep="/"),
                step=i,
            )
        if (i + 1) % save_every_n_steps == 0:
            # save checkpoint
            train_state.model.save_pretrained(step=i, checkpoint_path=FLAGS.save_dir)


if __name__ == "__main__":
    app.run(main)

