"""
This file contains methods from https://github.com/theislab/feature-attribution-sc
the methods are adapted.
"""

import torch
def batch_to_dict_scanvi(batch):
    return dict(x=batch["X"], batch_index=batch["batch"])


def batch_jacobian(outpt, inpt):
    n_out = outpt.shape[-1]

    jacs = []

    ones = torch.ones(outpt.shape[0]).to(outpt.device)

    for i in range(n_out):
        retain_graph = i != n_out - 1
        jacs.append(torch.autograd.grad(outpt[..., i], inpt, retain_graph=retain_graph)[0])

    return torch.stack(jacs, dim=-1)



def integrated_jacobian(func, inpt_dict, backprop_inpt_key, prime_inpt=None, n_steps=10):
    x = inpt_dict[backprop_inpt_key]
    if prime_inpt is not None:
        x_prime = prime_inpt.to(x.device)
    else:
        x_prime = torch.zeros_like(x)

    x_diff = x - x_prime

    jacs = []

    new_inpt_dict = dict(inpt_dict)

    for i in range(n_steps):
        new_inpt_dict[backprop_inpt_key] = x_prime + x_diff * (i + 1) / n_steps
        # pass input_dict['x'] in this case
        inp = new_inpt_dict['x']
        out = func(inp)

        jacs.append(batch_jacobian(out, x))

    return sum(jacs) * (1 / n_steps) * x_diff[..., None].detach()


def run_integrated_jacobian_scanvi(module_func, dl_base, n_steps=10, apply_abs=False, sum_obs=False):
    integrated_jacs = []

    for batch in dl_base:
        inpt_dict = batch_to_dict_scanvi(batch)
        inpt_dict["x"].requires_grad = True
        if torch.cuda.is_available():
            inpt_dict["x"] = inpt_dict["x"].cuda()

        integr_jac_batch = integrated_jacobian(module_func, inpt_dict, "x", n_steps=n_steps)
        if apply_abs:
            integr_jac_batch = integr_jac_batch.abs()

        integrated_jacs.append(integr_jac_batch.cpu() if not sum_obs else integr_jac_batch.sum(0).cpu())

    if sum_obs:
        result = torch.stack(integrated_jacs).sum(0)
    else:
        result = torch.cat(integrated_jacs)
    return result
