from torch import nn
import pickle

from mmcv.runner import get_dist_info


class Debug:
    def __init__(self,
                 **debug_args):
        self.iter=0
        for k, v in debug_args.items():
            if k == "debug_modules": continue
            setattr(self, k, v)

def with_debug(obj):
    return hasattr(obj, "debug") and obj.debug.debug is True

def do_debug_process(obj, repeating=False, interval=None):
    if not with_debug(obj) or (isinstance(obj, nn.Module) and not obj.training):
        return False
    debug_mod = obj.debug
    debug_iter=False
    if hasattr(debug_mod, "debug_print_iters"):
        debug_iter = debug_iter or debug_mod.iter in debug_mod.debug_print_iters
    if repeating and (hasattr(debug_mod, "debug_print_interval") or interval is not None):
        if interval is None: 
            interval = debug_mod.debug_print_interval
        debug_iter = debug_iter or debug_mod.iter % interval == 0
    return debug_iter


def save_and_exit(obj, filename="mdha_forward"):
    rank, world_size = get_dist_info()
    if rank == 0:
        with open(f"./experiments/{filename}.pkl", "wb") as f:
            pickle.dump(obj, f)
        raise Exception("save and exit")
