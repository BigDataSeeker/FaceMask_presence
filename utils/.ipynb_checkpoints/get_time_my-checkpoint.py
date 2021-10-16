import torch
import numpy as np
import time

def get_time_my(name_model,model, size_1, size_2, name_device='cpu'):

    device = torch.device(name_device)
    model = model.to(device)
    model.eval()

    dummy_input = torch.randn(1, 3, size_1, size_2, dtype=torch.float).to(device)
    # INIT LOGGERS
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 1000
    timings = np.zeros((repetitions,1))

    #GPU-WARM-UP
    for _ in range(10):
        _= model(dummy_input)
        
    torch.cuda.synchronize()
    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            dummy_input = torch.randn(1, 3, size_1, size_2).type(torch.cuda.FloatTensor).to(device)
            starter.record()
            _ = model(dummy_input)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time

    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    print(name_model, mean_syn, f' {size_1}_{size_2}')
    return mean_syn