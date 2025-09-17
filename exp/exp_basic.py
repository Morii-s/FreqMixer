import os
import torch
from models import  FreqMixer
from torch.nn.parallel import DistributedDataParallel as DDP


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'FreqMixer': FreqMixer,

        }
        self.device = self._acquire_device()
        self.model = self._build_model()

        # Count trainable parameters
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        param_count = sum([np.prod(p.size()) for p in model_parameters])
        
        # Calculate memory usage in bytes and convert to megabytes
        memory_usage_bytes = param_count * 4    # 4 bytes per float32 parameter
        memory_usage_MB = memory_usage_bytes / (1024 ** 2) 
        print(f"Total trainable parameters: {param_count}")
        print(f"Memory usage for trainable parameters: {memory_usage_MB:.2f} MB")
        
        # Measure static memory footprint
        print(f"Static memory footprint (allocated): {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")
        print(f"Static memory footprint (reserved): {torch.cuda.memory_reserved() / (1024 ** 2):.2f} MB")

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
