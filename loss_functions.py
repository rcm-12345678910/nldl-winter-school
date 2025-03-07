import pickle

import numpy as np
import torch
import torch.nn as nn
from utils.Tissue_compartment_modeling import (
    linear_interp1d,
    param_patlak_optimized,
    torch_CM_vB_wrap,
    twoTCMrev,
)
from utils.utils import get_valid_voxels


class CompoundLoss(nn.Module):
    def __init__(self, losses_with_weights, normalize_weights=False):
        """Compounds multiple loss functions with optional weight fading.

        Args:
            losses_with_weights (dict): A dictionary where keys are names of losses, and values
                are dictionaries with the following keys:
                    - "func": The loss function (nn.Module).
                    - "weight": The base weight for the loss (float, optional, default=1.0).
                    - "fade_range": A tuple (fade_start, fade_end) specifying the epoch range
                      for fading in the weight (optional). Weight is 0 before `fade_start`,
                      fades linearly from `fade_start` to `fade_end`, and is constant after `fade_end`.
            normalize_weights (bool): If True, the weights will be normalized so they sum to 1.0.
        """
        super().__init__()
        self.losses = {}
        for name, config in losses_with_weights.items():
            func = config.get("func")
            if not isinstance(func, nn.Module):
                raise ValueError(f"Loss function for '{name}' must be an nn.Module.")
            weight = config.get("weight", 1.0)
            fade_range = config.get("fade_range", None)
            if fade_range and (not isinstance(fade_range, tuple) or len(fade_range) != 2):
                raise ValueError(f"fade_range for '{name}' must be a tuple (fade_start, fade_end).")
            self.losses[name] = {"func": func, "weight": weight, "fade_range": fade_range}

        if normalize_weights:
            total_weight = sum(config["weight"] for config in self.losses.values())
            if total_weight == 0:
                raise ValueError("Total weight for loss normalization cannot be zero.")
            for config in self.losses.values():
                config["weight"] /= total_weight

        self.loss_modules = nn.ModuleDict(
            {name: config["func"] for name, config in self.losses.items()}
        )

    def get_dynamic_weight(self, weight, fade_range, epoch):
        """Compute dynamically adjusted weight for a loss term."""
        if fade_range is None:
            return weight
        fade_start, fade_end = fade_range
        if epoch < fade_start:
            return 0.0
        elif fade_start <= epoch <= fade_end:
            return weight * ((epoch - fade_start) / (fade_end - fade_start))
        else:
            return weight

    def forward(self, predictions, targets, epoch=None, **kwargs):
        """Compute the compound loss."""
        if epoch is None:
            raise ValueError("Epoch must be provided to compute dynamic weights.")

        total_loss = 0.0
        losses = {}
        for name, config in self.losses.items():
            func = config["func"]
            base_weight = config["weight"]
            fade_range = config["fade_range"]

            dynamic_weight = self.get_dynamic_weight(base_weight, fade_range, epoch)

            if dynamic_weight == 0.0:
                losses[name] = 0.0
                continue

            loss_value = func(predictions, targets, **kwargs)
            losses[name] = loss_value
            total_loss += loss_value * dynamic_weight

        losses["loss"] = total_loss
        return losses


class TCMLoss(nn.Module):
    def __init__(self, internal_lossfunc=nn.MSELoss()):
        super().__init__()

        self.reg_loss = internal_lossfunc

    def forward(self, predictions, targets, INPUT, TIME, VOI, **_):
        losses = []

        def wrap_model(model):
            def wrapper(x, param):
                return model(x, param[0], param[1], param[2], *param[3:])
            return wrapper

        compartment_model = wrap_model(torch_CM_vB_wrap(4))
        dtype = torch.float32

        for i, (DLIF, AIF, t_p) in enumerate(zip(predictions, targets, TIME)):
            t_p *= 60  # Convert to seconds

            # Interpolate the AIF and DLIF to the same time points
            t_int = torch.arange(
                t_p[0],
                t_p[-1],
                2.5,
                device=t_p.device,
                dtype=t_p.dtype,
            )
            dlif_int = linear_interp1d(t_int, t_p, DLIF)
            aif_int = linear_interp1d(t_int, t_p, AIF)

            # Stack Cp and time for input to tissue compartment model
            x_dlif = torch.stack([dlif_int, t_int / 60])
            x_aif = torch.stack([aif_int, t_int / 60])

            for _, activity in VOI.items():

                # Exctract the tissue activity for the current region, then interpolate
                tissue_activity = activity[i]
                # tissue_activity_int = linear_interp1d(t_int, t_p, tissue_activity)

                # Curve fit the tissue compartment model using numpy/scipy
                with torch.no_grad():
                    dlif_np = DLIF.detach().cpu().to(dtype).numpy()
                    aif_np = AIF.detach().cpu().to(dtype).numpy()
                    tp_np = t_p.detach().cpu().to(dtype).numpy()
                    tissue_activity_np = (
                        tissue_activity.detach().cpu().to(dtype).numpy()
                    )
                    assert not torch.isnan(DLIF).any(), f"DLIF has NaNs at i={i}"
                    assert not np.isnan(dlif_np).any(), f"DLIF_np has NaNs at i={i}"

                    # Fit curve
                    dlif_params_np, *_ = twoTCMrev(dlif_np, tp_np, tissue_activity_np)
                    aif_params_np, *_ = twoTCMrev(aif_np, tp_np, tissue_activity_np)

                    assert not np.isnan(dlif_params_np).any(), f"DLIF_params has NaNs at i={i}"
                # Convert to torch tensors
                dlif_params = torch.tensor(
                    dlif_params_np, dtype=DLIF.dtype, device=DLIF.device
                )
                aif_params = torch.tensor(
                    aif_params_np, dtype=AIF.dtype, device=AIF.device
                )

                # Compute the tissue compartment model curve fit with the external
                # params for the current region in the interpolated space
                DLIF_fit_int = compartment_model(x_dlif, dlif_params)
                AIF_fit_int = compartment_model(x_aif, aif_params)

                try:
                    assert not torch.isnan(DLIF_fit_int).any(), f"DLIF_fit_int has NaNs at i={i}"

                    # Interpolate the fit to the original time points
                    DLIF_fit = linear_interp1d(t_p, t_int, DLIF_fit_int)
                    AIF_fit = linear_interp1d(t_p, t_int, AIF_fit_int)

                    # Compute loss between predicted and reference reconstructions, and true TAC.
                    part_loss = self.reg_loss(DLIF_fit, AIF_fit)
                    # loss_fit = self.reg_loss(DLIF_fit, tissue_activity_int)
                    losses.append(part_loss)
                except Exception as e:
                    problematic_torch = {
                        "predictions": predictions.cpu().detach(),
                        "targets": targets.cpu().detach(),
                        "TIME": TIME.cpu().detach(),
                        "VOI": {k: v.cpu().detach() for k, v in VOI.items()},
                        "i": i,
                    }

                    with open("problematic_torch.pkl", "wb") as f:
                        pickle.dump(problematic_torch, f)
                    raise e

                # Clean up the numpy arrays
                del (
                    dlif_np,
                    aif_np,
                    tp_np,
                    tissue_activity_np,
                    dlif_params_np,
                    aif_params_np,
                )

        total_loss = torch.sum(torch.stack(losses))

        return total_loss


class WeightedMSELoss(nn.Module):
    def __init__(self, scale=1):
        super(WeightedMSELoss, self).__init__()
        self.scale = scale

    def forward(self, predictions, targets, **_):
        """
        Calculates the weighted mean squared error loss between predictions and targets.

        Args:
            predictions (torch.Tensor): The predicted values.
            targets (torch.Tensor): The target values.

        Returns:
            torch.Tensor: The total loss.
        """
        # Ensure the inputs are the correct shape
        if predictions.shape != targets.shape:
            raise ValueError("Inputs must have equal shapes")

        # Define the indices for splitting the vectors
        indices = [0, 25, 34, 42]
        weights = [0.4, 0.7, 1]

        # Calculate the weighted loss for each part
        losses = []
        for i in range(len(indices) - 1):
            start, end = indices[i], indices[i + 1]
            part_loss = torch.mean(
                weights[i] * (predictions[:, start:end] - targets[:, start:end]) ** 2
            )
            # print(f"Part {i+1} loss: {part_loss}")
            losses.append(part_loss)

        # Summing the individual losses to get the final loss
        total_loss = torch.sum(torch.stack(losses))

        # Clamp loss in case of numerical instability (when using float16)
        total_loss = torch.clamp(total_loss, 1e-5, 1e5)

        return total_loss
