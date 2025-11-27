import os
import numpy as np
import matplotlib.pyplot as plt

from models.networks import *
from misc.metric_tool import ConfuseMatrixMeter
from misc.logger_tool import Logger
from utils import de_norm
import utils


# Decide which device we want to run on
# torch.cuda.current_device()

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import torch
from thop import profile

import time
import torch
from thop import profile

def compute_model_complexity(
    model,
    input_shape=(1, 3, 512, 512),   # (N, C, H, W)
    device='cuda:0',
    num_inputs=2,                   # 变更检测网络一般是2张图像输入，如果是单输入模型设为 1
    warmup_runs=10,
    timed_runs=50,
    print_log=True,
):
    """
    计算模型的参数量、FLOPs 和 per-image 推理延迟。

    Args:
        model: PyTorch 模型
        input_shape: 输入张量形状 (N, C, H, W)
        device: 'cuda:0' 或 'cpu'
        num_inputs: 模型 forward 所需的输入个数（CD 模型通常是 2）
        warmup_runs: 预热次数（不计时，用来让 cuDNN 等稳定）
        timed_runs: 计时次数（取平均）
        print_log: 是否打印结果

    Returns:
        result: dict, 包含
            - params_m: 参数量 (Million)
            - flops_g: FLOPs (Giga)
            - latency_ms_per_image: 每张图像的平均推理时间 (ms)
            - latency_ms_per_batch: 每个 batch 的推理时间 (ms)
    """

    device = torch.device(device)
    model = model.to(device)
    model.eval()

    # 构造 dummy 输入
    dummy_input = torch.randn(*input_shape, device=device)
    # 对于变更检测，模型一般 forward(x1, x2)，所以这里复制 num_inputs 份
    inputs = tuple(dummy_input for _ in range(num_inputs))

    # -----------------------------
    # 1) 计算 FLOPs 和 Params（thop）
    # -----------------------------
    with torch.no_grad():
        flops, params = profile(model, inputs=inputs, verbose=False)

    params_m = params / 1e6
    flops_g = flops / 1e9

    # -----------------------------
    # 2) 计算 per-image latency
    # -----------------------------
    # 为了公平，打开 benchmark（如果你想固定可重复可关掉）
    torch.backends.cudnn.benchmark = True

    # 预热若干次，不计时
    with torch.no_grad():
        if device.type == 'cuda':
            torch.cuda.synchronize()
        for _ in range(warmup_runs):
            _ = model(*inputs)
        if device.type == 'cuda':
            torch.cuda.synchronize()

        # 正式计时
        start = time.time()
        for _ in range(timed_runs):
            _ = model(*inputs)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        end = time.time()

    avg_batch_time = (end - start) / timed_runs  # 每个 batch 的平均时间（秒）

    batch_size = input_shape[0]
    latency_ms_per_batch = avg_batch_time * 1000.0
    latency_ms_per_image = latency_ms_per_batch / batch_size

    # 对于变更检测任务：每个 sample 是一对图像
    # 如果你想「每对图像」算一次，也可以在写论文时说明：
    # “latency is measured per input pair (two images)”
    # 这里给出的 per-image 是按 batch_size 平均的。

    result = {
        'params_m': params_m,
        'flops_g': flops_g,
        'latency_ms_per_image': latency_ms_per_image,
        'latency_ms_per_batch': latency_ms_per_batch,
    }

    if print_log:
        print("==== Model Complexity ====")
        print(f"Input shape      : {input_shape}")
        print(f"Params           : {params_m:.3f} M")
        print(f"FLOPs            : {flops_g:.3f} G")
        print(f"Latency / batch  : {latency_ms_per_batch:.3f} ms")
        print(f"Latency / image  : {latency_ms_per_image:.3f} ms "
              f"(batch_size={batch_size})")
        print("==========================")


class CDEvaluator():

    def __init__(self, args, dataloader):

        self.dataloader = dataloader

        self.n_class = args.n_class
        # define G
        self.net_G = define_G(args=args, gpu_ids=args.gpu_ids)
        self.device = torch.device("cuda:%s" % args.gpu_ids[0] if torch.cuda.is_available() and len(args.gpu_ids)>0
                                   else "cpu")
        print(self.device)

        # define some other vars to record the training states
        self.running_metric = ConfuseMatrixMeter(n_class=self.n_class)

        # define logger file
        logger_path = os.path.join(args.checkpoint_dir, 'log_test.txt')
        self.logger = Logger(logger_path)
        self.logger.write_dict_str(args.__dict__)


        #  training log
        self.epoch_acc = 0
        self.best_val_acc = 0.0
        self.best_epoch_id = 0

        self.steps_per_epoch = len(dataloader)

        self.G_pred = None
        self.pred_vis = None
        self.batch = None
        self.is_training = False
        self.batch_id = 0
        self.epoch_id = 0
        self.checkpoint_dir = args.checkpoint_dir
        self.vis_dir = args.vis_dir

        # check and create model dir
        if os.path.exists(self.checkpoint_dir) is False:
            os.mkdir(self.checkpoint_dir)
        if os.path.exists(self.vis_dir) is False:
            os.mkdir(self.vis_dir)


    def _load_checkpoint(self, checkpoint_name='best_ckpt.pt'):

        if os.path.exists(os.path.join(self.checkpoint_dir, checkpoint_name)):
            self.logger.write('loading last checkpoint...\n')
            # load the entire checkpoint
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, checkpoint_name), map_location=self.device)

            self.net_G.load_state_dict(checkpoint['model_G_state_dict'])

            self.net_G.to(self.device)

            # update some other states
            self.best_val_acc = checkpoint['best_val_acc']
            self.best_epoch_id = checkpoint['best_epoch_id']

            self.logger.write('Eval Historical_best_acc = %.4f (at epoch %d)\n' %
                  (self.best_val_acc, self.best_epoch_id))
            self.logger.write('\n')

        else:
            raise FileNotFoundError('no such checkpoint %s' % checkpoint_name)


    def _visualize_pred(self):
        pred = torch.argmax(self.G_pred, dim=1, keepdim=True)
        pred_vis = pred * 255
        return pred_vis


    def _update_metric(self):
        """
        update metric
        """
        target = self.batch['L'].to(self.device).detach()
        G_pred = self.G_pred.detach()
        G_pred = torch.argmax(G_pred, dim=1)

        current_score = self.running_metric.update_cm(pr=G_pred.cpu().numpy(), gt=target.cpu().numpy())
        return current_score

    def _collect_running_batch_states(self):

        running_acc = self._update_metric()

        m = len(self.dataloader)

        if np.mod(self.batch_id, 100) == 1:
            message = 'Is_training: %s. [%d,%d],  running_mf1: %.5f\n' %\
                      (self.is_training, self.batch_id, m, running_acc)
            self.logger.write(message)

        if np.mod(self.batch_id, 100) == 3:
            vis_input = utils.make_numpy_grid(de_norm(self.batch['A']))
            vis_input2 = utils.make_numpy_grid(de_norm(self.batch['B']))

            vis_pred = utils.make_numpy_grid(self._visualize_pred())

            vis_gt = utils.make_numpy_grid(self.batch['L'])
            vis = np.concatenate([vis_input, vis_input2, vis_pred, vis_gt], axis=0)
            vis = np.clip(vis, a_min=0.0, a_max=1.0)
            file_name = os.path.join(
                self.vis_dir, 'eval_' + str(self.batch_id)+'.jpg')
            plt.imsave(file_name, vis)


    def _collect_epoch_states(self):

        scores_dict = self.running_metric.get_scores()

        np.save(os.path.join(self.checkpoint_dir, 'scores_dict.npy'), scores_dict)

        self.epoch_acc = scores_dict['mf1']

        with open(os.path.join(self.checkpoint_dir, '%s.txt' % (self.epoch_acc)),
                  mode='a') as file:
            pass

        message = ''
        for k, v in scores_dict.items():
            message += '%s: %.5f ' % (k, v)
        self.logger.write('%s\n' % message)  # save the message

        self.logger.write('\n')

    def _clear_cache(self):
        self.running_metric.clear()

    def _forward_pass(self, batch):
        self.batch = batch
        img_in1 = batch['A'].to(self.device)
        img_in2 = batch['B'].to(self.device)
        compute_model_complexity(self.net_G)
        self.G_pred, K = self.net_G(img_in1, img_in2)

    def eval_models(self,checkpoint_name='ResNet18.pt'):

        self._load_checkpoint(checkpoint_name)
        ################## Eval ##################
        ##########################################
        self.logger.write('Begin evaluation...\n')
        self._clear_cache()
        self.is_training = False
        self.net_G.eval()

        # Iterate over data.
        for self.batch_id, batch in enumerate(self.dataloader, 0):
            with torch.no_grad():
                self._forward_pass(batch)
            self._collect_running_batch_states()
        self._collect_epoch_states()
