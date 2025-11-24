from typing import List

import torch


def get_module_memory_consumption(module, device):
    module.to(device)
    memory_in_mb = torch.cuda.memory_allocated(device=device) / 1024 / 1024
    module.to(torch.device("cpu"))
    assert torch.cuda.memory_allocated(device=device) == 0.0
    return memory_in_mb


def get_model_memory_consumption_summary(model, device, verbose=False):
    assert torch.cuda.memory_allocated(device=device) == 0.0, "GPU is not empty, cannot measure memory consumption."
    memory_consumption = dict()
    for name, module in model.named_modules():
        if name == "":
            name = "overall"
        memory_in_mb = get_module_memory_consumption(module, device)
        if verbose:
            print(f"GPU Memory Requirement for {name}: {memory_in_mb} MiB")
        memory_consumption.update({name: memory_in_mb})
    return memory_consumption


def get_model_param_summary(model, verbose=False):
    params_dict = dict()
    overall_params = 0
    for name, params in model.named_parameters():
        num_params = params.numel()
        overall_params += num_params
        if verbose:
            print(f"GPU Memory Requirement for {name}: {params} MiB")
        params_dict.update({name: num_params})
    params_dict.update({"overall": overall_params})
    return params_dict


def show_model_param_summary(model, modules_of_interest: List[str], verbose=False):
    print("######## Parameter Summary: ########")
    params_dict = get_model_param_summary(model)
    summary_dict = dict()
    for module_name in modules_of_interest:
        summary_dict.update({module_name: 0.0})
    for name, val in params_dict.items():
        for module_name in modules_of_interest:
            if module_name == "norm":
                if module_name in name:
                    if verbose:
                        print(f"# of parameters for {name}: {val / 1000}k")
                    summary_dict[module_name] += val
            else:
                if module_name in name:
                    print(name)
                    if verbose:
                        print(f"# of parameters for {name}: {val / 1000}k")
                    summary_dict[module_name] += val
    print("############ Summary: ############")
    for module_name, val in summary_dict.items():
        print(f"Overall parameters for all {module_name}: {val / 1000}k")


def show_model_memory_consumption_summary(model, device, modules_of_interest: List[str], verbose=False):
    print("######## Memory Consumption Summary: ########")
    memory_consumption = get_model_memory_consumption_summary(model, device)
    summary_dict = dict()
    for module_name in modules_of_interest:
        summary_dict.update({module_name: 0.0})
    for name, val in memory_consumption.items():
        for module_name in modules_of_interest:
            if module_name in name.split(".")[-1]:
                if verbose:
                    print(f"GPU Memory Requirement for {name}: {val} MiB")
                summary_dict[module_name] += val
    print("############ Summary: ############")
    for module_name, val in summary_dict.items():
        print(f"Overall GPU Memory Requirement for all {module_name}: {val} MiB")
