# Functional implementation of EDMDPMSolverMultistepScheduler.
# 
# Copyright 2024 TSAIL Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# DISCLAIMER: This file is strongly influenced by https://github.com/LuChengTHU/dpm-solver and https://github.com/NVlabs/edm

import torch


def precondition_inputs(sample, sigma, sigma_data):
    """
    Preconditions inputs for the model.
    
    Args:
        sample: Input tensor to precondition
        sigma: Current noise level
        sigma_data: Data noise parameter
    
    Returns:
        Preconditioned sample tensor
    """
    c_in = 1 / ((sigma**2 + sigma_data**2) ** 0.5)
    scaled_sample = sample * c_in
    return scaled_sample


def precondition_outputs(sample, model_output, sigma, sigma_data, prediction_type):
    """
    Preconditions model outputs.
    
    Args:
        sample: Input sample tensor
        model_output: Raw model output tensor
        sigma: Current noise level
        sigma_data: Data noise parameter
        prediction_type: Type of prediction ('epsilon' or 'v_prediction')
    
    Returns:
        Preconditioned model output
    """
    c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)

    if prediction_type == "epsilon":
        c_out = sigma * sigma_data / (sigma**2 + sigma_data**2) ** 0.5
    elif prediction_type == "v_prediction":
        c_out = -sigma * sigma_data / (sigma**2 + sigma_data**2) ** 0.5
    else:
        raise ValueError(f"Prediction type {prediction_type} not supported")

    denoised = c_skip * sample + c_out * model_output
    return denoised


def dpm_solver_first_order_update(
    model_output,
    sample,
    sigma_t,
    sigma_s,
    noise=None,
    algorithm_type="dpmsolver++"
):
    """
    Performs first-order DPM-Solver update step.
    
    Args:
        model_output: Output from the model
        sample: Current sample
        sigma_t: Target noise level
        sigma_s: Source noise level 
        noise: Optional noise tensor for SDE variant
        algorithm_type: Either 'dpmsolver++' or 'sde-dpmsolver++'
    
    Returns:
        Updated sample tensor
    """
    alpha_t = torch.tensor(1.0)  # Pre-scaled inputs
    alpha_s = torch.tensor(1.0)  # Pre-scaled inputs
    
    lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
    lambda_s = torch.log(alpha_s) - torch.log(sigma_s)
    
    h = lambda_t - lambda_s
    
    if algorithm_type == "dpmsolver++":
        x_t = (sigma_t / sigma_s) * sample - (alpha_t * (torch.exp(-h) - 1.0)) * model_output
    elif algorithm_type == "sde-dpmsolver++":
        if noise is None:
            raise ValueError("Noise required for sde-dpmsolver++")
        x_t = (
            (sigma_t / sigma_s * torch.exp(-h)) * sample
            + (alpha_t * (1 - torch.exp(-2.0 * h))) * model_output
            + sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h)) * noise
        )
    else:
        raise ValueError(f"Unknown algorithm type: {algorithm_type}")
        
    return x_t


def multistep_dpm_solver_second_order_update(
    model_output_list,
    sample,
    sigma_t,
    sigma_s0,
    sigma_s1,
    noise=None,
    algorithm_type="dpmsolver++",
    solver_type="midpoint"
):
    """
    Performs second-order DPM-Solver update step.
    
    Args:
        model_output_list: List of last two model outputs [current, previous]
        sample: Current sample
        sigma_t: Target noise level
        sigma_s0: Current noise level
        sigma_s1: Previous noise level
        noise: Optional noise tensor for SDE variant
        algorithm_type: Either 'dpmsolver++' or 'sde-dpmsolver++'
        solver_type: Either 'midpoint' or 'heun'
    
    Returns:
        Updated sample tensor
    """
    alpha_t = torch.tensor(1.0)  # Pre-scaled inputs
    alpha_s0 = torch.tensor(1.0)
    alpha_s1 = torch.tensor(1.0)
    
    lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
    lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
    lambda_s1 = torch.log(alpha_s1) - torch.log(sigma_s1)
    
    h = lambda_t - lambda_s0
    h_0 = lambda_s0 - lambda_s1
    
    r0 = h_0 / h
    m0, m1 = model_output_list[-1], model_output_list[-2]
    
    D0 = m0
    D1 = (1.0 / r0) * (m0 - m1)
    
    if algorithm_type == "dpmsolver++":
        if solver_type == "midpoint":
            x_t = (
                (sigma_t / sigma_s0) * sample
                - (alpha_t * (torch.exp(-h) - 1.0)) * D0
                - 0.5 * (alpha_t * (torch.exp(-h) - 1.0)) * D1
            )
        elif solver_type == "heun":
            x_t = (
                (sigma_t / sigma_s0) * sample
                - (alpha_t * (torch.exp(-h) - 1.0)) * D0
                + (alpha_t * ((torch.exp(-h) - 1.0) / h + 1.0)) * D1
            )
        else:
            raise ValueError(f"Unknown solver type: {solver_type}")
            
    elif algorithm_type == "sde-dpmsolver++":
        if noise is None:
            raise ValueError("Noise required for sde-dpmsolver++")
            
        if solver_type == "midpoint":
            x_t = (
                (sigma_t / sigma_s0 * torch.exp(-h)) * sample
                + (alpha_t * (1 - torch.exp(-2.0 * h))) * D0
                + 0.5 * (alpha_t * (1 - torch.exp(-2.0 * h))) * D1
                + sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h)) * noise
            )
        elif solver_type == "heun":
            x_t = (
                (sigma_t / sigma_s0 * torch.exp(-h)) * sample
                + (alpha_t * (1 - torch.exp(-2.0 * h))) * D0
                + (alpha_t * ((1.0 - torch.exp(-2.0 * h)) / (-2.0 * h) + 1.0)) * D1
                + sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h)) * noise
            )
        else:
            raise ValueError(f"Unknown solver type: {solver_type}")
    else:
        raise ValueError(f"Unknown algorithm type: {algorithm_type}")
        
    return x_t
