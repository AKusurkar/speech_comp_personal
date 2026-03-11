def custom_params(self):
    
    adapter_params = []
    pretrained_params = []
    
    for name, param in self.named_parameters():
        if not param.requires_grad:
            continue 
            
        if "adapter" in name:
            adapter_params.append(param)
        else:
            pretrained_params.append(param)

    self._optimizer_param_groups = [
        {"params": adapter_params}, 
        {"params": pretrained_params, "lr": 1e-5, "weight_decay": 0} 
    ]