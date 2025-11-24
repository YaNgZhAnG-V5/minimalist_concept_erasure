import os

from diffsolver.hooks.cross_attn_hooks import CrossAttentionExtractionHook
from diffsolver.hooks.ff_hooks import FeedForwardHooker
from diffsolver.hooks.linear_layer_hooks import LinearLayerHooker
from diffsolver.hooks.norm_hook import NormHooker


def init_hooker(cfg, pipe, torch_dtype, project_folder):
    """
    Initialize hookers for training
    return:
        hookers: list of hookers
        hookers_tuple: list of tuples of hookers and their names
        lr_list: list of learning rates
    """
    if any(model in cfg.trainer.model for model in ["flux", "sd3"]):
        # TODO: temporary solusion, fix this in the future
        if cfg.trainer.model in ["sd3"]:
            cfg.trainer.n_lr = 0
        return init_attn_ffn_norm_hooker(cfg, pipe, torch_dtype, project_folder)
    else:
        return init_linear_hooker(cfg, pipe, torch_dtype, project_folder)


def init_attn_ffn_norm_hooker(cfg, pipe, torch_dtype, project_folder):
    """
    Initialize cross attention, feedforward, and norm hookers for training
    return:
        hookers: list of hookers
        hookers_tuple: list of tuples of hookers and their names
        lr_list: list of learning rates
    """
    cross_attn_hooker = CrossAttentionExtractionHook(
        pipe,
        regex=cfg.trainer.regex,
        dtype=torch_dtype,
        head_num_filter=cfg.trainer.head_num_filter,
        masking=cfg.trainer.masking,
        dst=os.path.join(project_folder, cfg.logger.save_lambda_path.attn),
        epsilon=cfg.trainer.epsilon,
        model_name=cfg.trainer.model,
        attn_name=cfg.trainer.attn_name,
        use_log=cfg.trainer.use_log,
        eps=cfg.trainer.masking_eps,
    )
    cross_attn_hooker.add_hooks(init_value=cfg.trainer.init_lambda)

    # initialize feedforward hooks
    ff_hooker = FeedForwardHooker(
        pipe,
        regex=cfg.trainer.regex,
        dtype=torch_dtype,
        masking=cfg.trainer.masking,
        dst=os.path.join(project_folder, cfg.logger.save_lambda_path.ffn),
        epsilon=cfg.trainer.epsilon,
        eps=cfg.trainer.masking_eps,
        use_log=cfg.trainer.use_log,
    )
    ff_hooker.add_hooks(init_value=cfg.trainer.init_lambda)
    hookers = [cross_attn_hooker, ff_hooker]
    hookers_tuple = [(cross_attn_hooker, "attn"), (ff_hooker, "ff")]
    lr_list = [cfg.trainer.attn_lr, cfg.trainer.ff_lr]

    # initialize norm hooks if lr is not 0
    if cfg.trainer.n_lr != 0:
        norm_hooker = NormHooker(
            pipe,
            regex=cfg.trainer.regex,
            dtype=torch_dtype,
            masking=cfg.trainer.masking,
            dst=os.path.join(project_folder, cfg.logger.save_lambda_path.norm),
            epsilon=cfg.trainer.epsilon,
            eps=cfg.trainer.masking_eps,
            use_log=cfg.trainer.use_log,
        )
        norm_hooker.add_hooks(init_value=cfg.trainer.init_lambda)
        hookers.append(norm_hooker)
        hookers_tuple.append((norm_hooker, "norm"))
        lr_list.append(cfg.trainer.n_lr)
    return hookers, hookers_tuple, lr_list


def init_linear_hooker(cfg, pipe, torch_dtype, project_folder):
    """
    Initialize linear hooker for training
    return:
        hookers: list of hookers
        hookers_tuple: list of tuples of hookers and their names
        lr_list: list of learning rates
    """
    linear_hooker = LinearLayerHooker(
        pipe,
        regex=cfg.trainer.regex,
        dtype=torch_dtype,
        masking=cfg.trainer.masking,
        dst=os.path.join(project_folder, cfg.logger.save_lambda_path.ffn),
        epsilon=cfg.trainer.epsilon,
        eps=cfg.trainer.masking_eps,
        use_log=cfg.trainer.use_log,
    )
    hookers = [linear_hooker]
    hookers_tuple = [(linear_hooker, "linear")]
    linear_hooker.add_hooks(init_value=cfg.trainer.init_lambda)
    lr_list = [cfg.trainer.ff_lr]
    return hookers, hookers_tuple, lr_list
