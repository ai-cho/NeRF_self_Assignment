"""
train.py

A script for training.
"""

from pathlib import Path
from typing import Dict, Tuple

import hydra
from hydra.core.hydra_config import HydraConfig
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
import torch.utils.data as data
import torchmetrics
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from tqdm import tqdm


from torch_nerf.runners.utils import (
    init_cuda,
    init_dataset_and_loader,
    init_optimizer_and_scheduler,
    init_renderer,
    init_scene,
    init_tensorboard,
    init_torch,
    init_loss_func,
    load_ckpt,
    save_ckpt,
)
import torch_nerf.src.cameras.cameras as cameras
from torch_nerf.src.utils.data import NeRFBlenderDataset, LLFFDataset


def init_dataset_and_loader(cfg: DictConfig):
    """
    Initializes the dataset and loader.
    """
    dataset_type = str(cfg.data.dataset_type)

    if dataset_type == "nerf_synthetic":
        train_dataset = NeRFBlenderDataset(
            cfg.data.data_root,
            scene_name=cfg.data.scene_name,
            data_type="train",
            half_res=cfg.data.half_res,
            white_bg=cfg.data.white_bg,
        )
        train_loader = data.DataLoader(
            train_dataset,
            batch_size=cfg.data.batch_size,
            shuffle=cfg.data.shuffle,
            num_workers=1,
        )

        val_dataset = NeRFBlenderDataset(
            cfg.data.data_root,
            scene_name=cfg.data.scene_name,
            data_type="val",
            half_res=cfg.data.half_res,
            white_bg=cfg.data.white_bg,
        )
        val_loader = data.DataLoader(
            val_dataset,
            batch_size=cfg.data.batch_size,
            shuffle=False,
            num_workers=1,
        )

    elif dataset_type == "nerf_llff":
        raise NotImplementedError(f"Unsupported dataset type: {dataset_type}")
    elif dataset_type == "nerf_deepvoxels":
        raise NotImplementedError(f"Unsupported dataset type: {dataset_type}")
    else:
        raise NotImplementedError(f"Unsupported dataset type: {dataset_type}")

    return train_dataset, train_loader, val_dataset, val_loader


def train_one_epoch(
    epoch,
    cfg,
    default_scene,
    renderer,
    dataset,
    loader,
    loss_func,
    optimizer,
    scheduler,
    fine_scene=None,
):
    """
    Training routine for one epoch.
    """

    device_idx = torch.cuda.current_device()
    device = torch.device(device_idx)

    loss_dict = {}

    for batch in loader:

        loss = 0.0

        # parse batch
        pixel_gt, extrinsic = batch
        pixel_gt = pixel_gt.squeeze()
        pixel_gt = pixel_gt.reshape(-1, 3)  # (H, W, 3) -> (H * W, 3)
        extrinsic = extrinsic.squeeze()

        # initialize gradients
        optimizer.zero_grad()

        # create a camera for the current viewpoint
        camera = cameras.Camera(
            extrinsic,
            dataset.focal_length,
            dataset.focal_length,
            dataset.img_width / 2.0,
            dataset.img_height / 2.0,
            cfg.renderer.t_near,
            cfg.renderer.t_far,
            dataset.img_width,
            dataset.img_height,
            device,
        )

        # randomly sample pixels from which rays will be casted
        pixel_indices = torch.tensor(
            np.random.choice(
                camera.image_height.item() * camera.image_width.item(),
                size=[cfg.renderer.num_pixels],
                replace=False,
            ),
            device=camera.device,
        )
        if epoch < 5:  # warm-up during early stage of training

            # sample pixels around the center
            center_i = (dataset.img_height - 1) // 2
            center_j = (dataset.img_width - 1) // 2

            center_is = torch.arange(
                center_i - center_i // 2,
                center_i + center_i // 2,
            )
            center_js = torch.arange(
                center_j - center_j // 2,
                center_j + center_j // 2,
            )

            center_indices = torch.cartesian_prod(center_is, center_js)
            center_indices = center_indices[:, 0] * dataset.img_width + center_indices[:, 1]
            pixel_indices = center_indices[
                torch.randperm(len(center_indices))[: cfg.renderer.num_pixels]
            ]

        # forward prop.
        coarse_pred, coarse_weights, coarse_t_samples = renderer.render_scene(
            default_scene,
            camera,
            pixel_indices,
            cfg.renderer.num_samples_coarse,
            cfg.renderer.project_to_ndc,
        )
        coarse_loss = loss_func(pixel_gt[pixel_indices, ...].to(coarse_pred), coarse_pred)
        loss += coarse_loss
        if "coarse_loss" not in loss_dict:
            loss_dict["coarse_loss"] = coarse_loss.item()
        else:
            loss_dict["coarse_loss"] += coarse_loss.item()

        if not fine_scene is None:

            # forward prop.
            fine_pred, _, _ = renderer.render_scene(
                fine_scene,
                camera,
                pixel_indices,
                cfg.renderer.num_samples_fine,
                cfg.renderer.project_to_ndc,
                weights=coarse_weights.detach(),
                prev_t_samples=coarse_t_samples.detach(),
            )

            fine_loss = loss_func(pixel_gt[pixel_indices, ...].to(fine_pred), fine_pred)
            loss += fine_loss
            if "fine_loss" not in loss_dict:
                loss_dict["fine_loss"] = fine_loss.item()
            else:
                loss_dict["fine_loss"] += fine_loss.item()

        if "loss" not in loss_dict:
            loss_dict["loss"] = loss.item()
        else:
            loss_dict["loss"] += loss.item()

        # back prop.
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

    # compute average loss over all batches
    for key in loss_dict.keys():
        loss_dict[key] /= len(loader)

    return loss_dict


@torch.no_grad()
def validate_one_epoch(
    cfg,
    default_scene,
    renderer,
    dataset,
    loader,
    fine_scene=None,
 ):
    """
    Validation routine for one epoch.
    """

    device_idx = torch.cuda.current_device()
    device = torch.device(device_idx)

    # initialize metrics
    lpips = LearnedPerceptualImagePatchSimilarity(
        net_type='vgg',
        normalize=True,
    ).to(device)
    psnr = torchmetrics.PeakSignalNoiseRatio().to(device)
    ssim = torchmetrics.StructuralSimilarityIndexMeasure().to(device)

    val_images = []
    metric_dict = {}

    num_sample = 0
    for batch_index, batch in enumerate(loader):

        if batch_index >= cfg.train_params.validation.num_batch:
            break

        # parse batch
        pixel_gt, extrinsic = batch
        batch_size = len(extrinsic)
        pixel_gt = pixel_gt.squeeze()
        pixel_gt = pixel_gt.reshape(-1, 3)  # (H, W, 3) -> (H * W, 3)
        extrinsic = extrinsic.squeeze()

        # count the number of samples used for validation
        num_sample += batch_size

        # create a camera for the current viewpoint
        camera = cameras.Camera(
            extrinsic,
            dataset.focal_length,
            dataset.focal_length,
            dataset.img_width / 2.0,
            dataset.img_height / 2.0,
            cfg.renderer.t_near,
            cfg.renderer.t_far,
            dataset.img_width,
            dataset.img_height,
            device,
        )

        # select all pixels
        pixel_indices = torch.arange(
            0,
            camera.image_height * camera.image_width,
            device=camera.device,
        )

        pixel_pred, coarse_weights, coarse_t_samples = renderer.render_scene(
            default_scene,
            camera,
            pixel_indices,
            cfg.renderer.num_samples_coarse,
            cfg.renderer.project_to_ndc,
            num_ray_batch=len(pixel_indices) // cfg.renderer.num_pixels,
        )

        if not fine_scene is None:

            pixel_pred, _, _ = renderer.render_scene(
                fine_scene,
                camera,
                pixel_indices,
                cfg.renderer.num_samples_fine,
                cfg.renderer.project_to_ndc,
                weights=coarse_weights,
                prev_t_samples=coarse_t_samples,
                num_ray_batch=len(pixel_indices) // cfg.renderer.num_pixels,
            )

        # (H * W, C) -> (1, C, H, W)
        pixel_pred = pixel_pred.reshape(dataset.img_height, dataset.img_width, -1)
        pixel_pred = pixel_pred.permute(2, 0, 1)[None]
        pixel_gt = pixel_gt.reshape(dataset.img_height, dataset.img_width, -1)
        pixel_gt = pixel_gt.permute(2, 0, 1)[None]

        # clamp values to [0, 1]
        pixel_pred = torch.clamp(pixel_pred, 0.0, 1.0)
        pixel_gt = torch.clamp(pixel_gt, 0.0, 1.0)

        # collect image(s)
        pixel_all = torch.cat([pixel_pred.cpu(), pixel_gt], dim=3)
        val_images.append(pixel_all)

        # compute LPIPS
        lpips_value = lpips(pixel_pred, pixel_gt.to(pixel_pred)).item()
        if "lpips" not in metric_dict:
            metric_dict["lpips"] = lpips_value
        else:
            metric_dict["lpips"] += lpips_value

        # compute PSNR
        psnr_value = psnr(pixel_pred, pixel_gt.to(pixel_pred)).item()
        if "psnr" not in metric_dict:
            metric_dict["psnr"] = psnr_value
        else:
            metric_dict["psnr"] += psnr_value

        # compute SSIM
        ssim_value = ssim(pixel_pred, pixel_gt.to(pixel_pred)).item()
        if "ssim" not in metric_dict:
            metric_dict["ssim"] = ssim_value
        else:
            metric_dict["ssim"] += ssim_value

    # compute average over validation samples
    for metric_name in metric_dict:
        metric_dict[metric_name] /= num_sample

    val_images = torch.cat(val_images, dim=0)

    return metric_dict, val_images


@hydra.main(
    version_base=None,
    config_path="../configs",  # config file search path is relative to this script
    config_name="default",
)
def main(cfg: DictConfig) -> None:
    """The entry point of training code."""

    log_dir = Path(HydraConfig.get().runtime.output_dir)
    if "log_dir" in cfg.keys():

        # identify existing log directory
        log_dir = Path(cfg.log_dir)
        assert log_dir.exists(), f"Provided log directory {str(log_dir)} does not exist."

        # override the current config with the existing one
        config_dir = log_dir / ".hydra"
        assert config_dir.exists(), "Provided log directory does not contain config directory."
        cfg = OmegaConf.load(config_dir / "config.yaml")

    # initialize Tensorboard writer
    tb_log_dir = log_dir / "tensorboard"
    writer = init_tensorboard(tb_log_dir)

    # initialize PyTorch
    init_torch(cfg)
    # initialize CUDA device
    init_cuda(cfg)
    # initialize renderer, data
    renderer = init_renderer(cfg)
    # initialize dataset and loader
    train_dataset, train_loader, val_dataset, val_loader = init_dataset_and_loader(cfg)

    # initialize scene and network parameters
    default_scene, fine_scene = init_scene(cfg)
    # initialize optimizer and learning rate scheduler
    optimizer, scheduler = init_optimizer_and_scheduler(
        cfg,
        default_scene,
        fine_scene=fine_scene,
    )
    # initialize objective function
    loss_func = init_loss_func(cfg)
    # load if checkpoint exists
    start_epoch = load_ckpt(
        log_dir / "ckpt",
        default_scene,
        fine_scene,
        optimizer,
        scheduler,
    )
    for epoch in tqdm(range(start_epoch, cfg.train_params.optim.num_iter // len(train_dataset))):

        # train
        train_loss_dict = train_one_epoch(
            epoch,
            cfg,
            default_scene,
            renderer,
            train_dataset,
            train_loader,
            loss_func,
            optimizer,
            scheduler,
            fine_scene=fine_scene,
        )
        for loss_name, value in train_loss_dict.items():
            writer.add_scalar(f"train/{loss_name}", value, epoch)

        # validate
        if (epoch + 1) % cfg.train_params.validation.validate_every == 0:
            val_metric_dict, val_images = validate_one_epoch(
                cfg,
                default_scene,
                renderer,
                val_dataset,
                val_loader,
                fine_scene=fine_scene,
            )

            # log metrics
            for metric_name, value in val_metric_dict.items():
                writer.add_scalar(f"val/{metric_name}", value, epoch)

            # log images
            for index in range(val_images.shape[0]):
                writer.add_image(f"val/image_{index}", val_images[index], epoch)

        # save checkpoint
        if (epoch + 1) % cfg.train_params.log.epoch_btw_ckpt == 0:
            ckpt_dir = log_dir / "ckpt"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            save_ckpt(
                ckpt_dir,
                epoch,
                default_scene,
                fine_scene,
                optimizer,
                scheduler,
            )


if __name__ == "__main__":
    main()
