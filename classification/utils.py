import torch
import torch.utils as torch_utils

BATCH_SIZE = 32

def create_dataloader(data, labeling):
    dataset = torch_utils.data.TensorDataset(data, labeling)
    dataloader = torch_utils.data.DataLoader(dataset, batch_size=BATCH_SIZE)
    return dataloader

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    # output = torch.mean(output, -1)
    # target = torch.mean(target, -1)
    res_arr = []
    last_dim_size = output.size(-1)
    result = []
    maxk = max(topk)
    for i in range(last_dim_size):
        curr_output = output[:, :, i:i + 1]
        curr_output = curr_output.squeeze(-1)

        curr_target = target[:, i:i + 1]
        curr_target = curr_target.squeeze(-1)

        with torch.no_grad():

            batch_size = curr_target.size(0)
            _, pred = curr_output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(curr_target.view(1, -1).expand_as(pred))
            res = []
            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            result.append(res)

    result_summary = []
    for i in range(2):
        result_summary.append([])
    for i in range(last_dim_size):
        for j, elem in enumerate(result[i]):
            result_summary[j].append(elem)
    for i in range(2):
        result_summary[i] = torch.FloatTensor(result_summary[i])
        result_summary[i] = [torch.mean(result_summary[i], -1)]
    return result_summary

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)