import os
import numpy as np
import math
import sys
from typing import Iterable, Optional
import torch
from mixup import Mixup
from timm.utils import accuracy, ModelEma
import utils
from scipy.special import softmax
import pickle

def train_class_batch(model, samples, target, criterion, args=None):
    outputs = model(samples)
    loss = criterion(outputs, target)
    return loss, outputs

def train_gaze_batch(model, samples, target, criterion_detailed, args=None):
    """专门用于gaze回归任务的训练批次"""
    # 检查输入数据是否有异常值
    if torch.isnan(samples).any():
        print("Warning: NaN detected in input samples!")
        samples = torch.nan_to_num(samples, nan=0.0)
    
    if torch.isinf(samples).any():
        print("Warning: Inf detected in input samples!")
        samples = torch.nan_to_num(samples, posinf=1.0, neginf=-1.0)
    
    # 限制输入值范围
    samples = torch.clamp(samples, min=-10.0, max=10.0)
    
    outputs = model(samples)
    
    # 检查模型输出
    if torch.isnan(outputs).any():
        print("Warning: NaN detected in model outputs!")
        outputs = torch.nan_to_num(outputs, nan=0.0)
    
    if torch.isinf(outputs).any():
        print("Warning: Inf detected in model outputs!")
        outputs = torch.nan_to_num(outputs, posinf=1.0, neginf=-1.0)
    
    # 使用详细损失函数
    if criterion_detailed is not None:
        print("shape of outputs:", outputs.shape)
        print("shape of target:", target.shape)
        total_loss, mse_loss, angular_loss = criterion_detailed(outputs, target)
        return total_loss, outputs, mse_loss, angular_loss
    else:
        # 简单的MSE损失
        loss = torch.nn.functional.mse_loss(outputs, target)
        return loss, outputs, loss, torch.tensor(0.0)

def train_l2cs_batch(model, samples, target, criterion, args=None):
    # 检查输入数据是否有异常值
    if torch.isnan(samples).any():
        print("Warning: NaN detected in input samples!")
        samples = torch.nan_to_num(samples, nan=0.0)
    
    if torch.isinf(samples).any():
        print("Warning: Inf detected in input samples!")
        samples = torch.nan_to_num(samples, posinf=1.0, neginf=-1.0)
    
    # 限制输入值范围
    samples = torch.clamp(samples, min=-10.0, max=10.0)
    
    outputs = model(samples)
    
    # 检查模型输出
    if torch.isnan(outputs).any():
        print("Warning: NaN detected in model outputs!")
        print("Sample max:", samples.max().item())
        print("Sample min:", samples.min().item())
        print("Sample mean:", samples.mean().item())
        outputs = torch.nan_to_num(outputs, nan=0.0)
    
    if torch.isinf(outputs).any():
        print("Warning: Inf detected in model outputs!")
        outputs = torch.nan_to_num(outputs, posinf=1.0, neginf=-1.0)
    
    # print("outputs shape:", outputs.shape)
    # print("targets shape:", target.shape)
    
    loss = criterion(outputs, target)
    
    angular_error = compute_angular_error(outputs, target)
    
    print('Angular Error is', angular_error.item())
    
    # 检查损失值
    if torch.isnan(loss):
        print("Warning: NaN loss detected!")
        loss = torch.tensor(1.0, device=loss.device, requires_grad=True)
    
    # print("loss:", loss.item())
    return loss, outputs


def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
                    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None, args=None, criterion_detailed=None):
    model.train(True)
    
    # print("before entering train_one_epoch, model.micro_steps:", getattr(model, 'micro_steps', 0))
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    
    # print("before zero_grad, model.micro_steps:", getattr(model, 'micro_steps', 0))

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    for data_iter_step, (samples, targets, _, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]
                    
        # print("len samples:", len(samples))
        # print("shape of samples:", samples[0].shape)
        # print("shape of samples:", samples[1].shape)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        # 确保标签数据类型正确
        if args and args.data_set == 'Gaze360':
            targets = targets.float()  # 回归任务使用float
        else:
            targets = targets.long()   # 分类任务使用long
        
        print("samples shape:", samples.shape)
        print("targets shape:", targets.shape)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if loss_scaler is None:
            # 对于 Gaze360 数据集，不要转换为 half 类型
            if args and args.data_set == 'Gaze360':
                samples = samples.float()  # 保持 float 类型
                if criterion_detailed is not None:
                    loss, output, mse_loss, angular_loss = train_gaze_batch(
                        model, samples, targets, criterion_detailed, args)
                    # 记录详细损失信息
                    print(f"Total Loss: {loss.item():.6f}, MSE: {mse_loss.item():.6f}, Angular: {angular_loss.item():.4f}°")
                else:
                    loss, output = train_class_batch(
                        model, samples, targets, criterion, args)
            else:
                samples = samples.half()
                # samples = samples.float()
                
                loss, output = train_class_batch(
                    model, samples, targets, criterion, args)
        else:
            with torch.cuda.amp.autocast():
                if args and args.data_set == 'Gaze360' and criterion_detailed is not None:
                    loss, output, mse_loss, angular_loss = train_gaze_batch(
                        model, samples, targets, criterion_detailed, args)
                else:
                    loss, output = train_class_batch(
                        model, samples, targets, criterion, args)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()

            if (data_iter_step + 1) % update_freq == 0:
                # model.zero_grad()
                # Deepspeed will call step() & model.zero_grad() automatic
                if model_ema is not None:
                    model_ema.update(model)
            grad_norm = None
            loss_scale_value = get_loss_scale_for_deepspeed(model)
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        # 根据数据集类型计算准确率
        if mixup_fn is None:
            if args and args.data_set == 'Gaze360':
                # 回归任务：计算角度误差
                with torch.no_grad():
                    # 计算每个样本的角度误差
                    angle_errors = torch.sqrt(torch.sum((output - targets) ** 2, dim=1))
                    class_acc = torch.mean(angle_errors)  # 平均角度误差作为"准确率"
            else:
                # 分类任务：计算分类准确率
                class_acc = (output.max(-1)[-1] == targets).float().mean()
        else:
            class_acc = None
        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



def spherical2cartesial(x):
    
    output = torch.zeros(x.size(0),3)
    output[:,2] = -torch.cos(x[:,1])*torch.cos(x[:,0])
    output[:,0] = torch.cos(x[:,1])*torch.sin(x[:,0])
    output[:,1] = torch.sin(x[:,1])

    return output
def compute_angular_error(input,target):

    input = spherical2cartesial(input)
    target = spherical2cartesial(target)

    input = input.view(-1,3,1)
    target = target.view(-1,1,3)
    output_dot = torch.bmm(target,input)
    output_dot = output_dot.view(-1)
    output_dot = torch.acos(output_dot)
    output_dot = output_dot.data
    output_dot = 180*torch.mean(output_dot)/math.pi
    return output_dot


@torch.no_grad()
def validation_one_epoch(data_loader, model, device, args=None):
    # 根据数据集类型选择不同的损失函数
    if args and args.data_set == 'Gaze360':
        criterion = torch.nn.MSELoss()  # 回归任务使用MSE损失
    else:
        criterion = torch.nn.CrossEntropyLoss()  # 分类任务使用交叉熵损失

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Val:'

    # switch to evaluation mode
    model.eval()

    outputs, targets = [], []

    for batch in metric_logger.log_every(data_loader, 10, header):
        videos = batch[0]
        target = batch[1]
        videos = videos.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(videos)
            loss = criterion(output, target)

        # 根据数据集类型计算不同的准确率
        if args and args.data_set == 'Gaze360':
            # 回归任务：计算角度误差
            angle_error = compute_angular_error(output, target)
            acc1 = torch.mean(angle_error)  # 平均角度误差
            acc5 = torch.mean(angle_error)  # 对于回归任务，acc5 = acc1
        else:
            # 分类任务：计算top-1和top-5准确率
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
        
        output, target = output.cpu().detach().numpy(), target.cpu().detach().numpy()
        outputs.append(output)
        targets.append(target)

        batch_size = videos.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    # 计算整体指标
    if args and args.data_set == 'Gaze360':
        # 回归任务的指标计算
        preds, labels = np.concatenate(outputs), np.concatenate(targets)
        # angle_errors = np.sqrt(np.sum((preds - labels) ** 2, axis=1))
        angle_errors = compute_angular_error(torch.tensor(preds), torch.tensor(labels)).numpy()
        mean_angle_error = np.mean(angle_errors)
        
        metric_logger.meters['mean_angle_error'].update(mean_angle_error, n=len(preds))
        
        print('* Mean Angle Error {mae:.4f} loss {losses.global_avg:.3f}'
              .format(mae=mean_angle_error, losses=metric_logger.loss))
    else:
        # 分类任务的指标计算
        preds, labels = np.concatenate(outputs), np.concatenate(targets)
        preds = np.argmax(preds, axis=1)
        from sklearn.metrics import confusion_matrix, f1_score
        conf_mat = confusion_matrix(y_pred=preds, y_true=labels)
        class_acc = conf_mat.diagonal() / conf_mat.sum(axis=1)
        uar = np.mean(class_acc)
        war = conf_mat.trace() / conf_mat.sum()
        weighted_f1 = f1_score(y_pred=preds, y_true=labels, average='weighted')
        micro_f1 = f1_score(y_pred=preds, y_true=labels, average='micro')
        macro_f1 = f1_score(y_pred=preds, y_true=labels, average='macro')
        metric_logger.meters['uar'].update(uar, n=len(preds))
        metric_logger.meters['war'].update(war, n=len(preds))
        metric_logger.meters['weighted_f1'].update(weighted_f1, n=len(preds))
        metric_logger.meters['micro_f1'].update(micro_f1, n=len(preds))
        metric_logger.meters['macro_f1'].update(macro_f1, n=len(preds))

        print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
              .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

        print('* WAR {war.global_avg:.4f} UAR {uar.global_avg:.4f} weighted_f1 {weighted_f1.global_avg:.4f} micro_f1 {micro_f1.global_avg:.4f} macro_f1 {macro_f1.global_avg:.4f}'
              .format(war=metric_logger.war, uar=metric_logger.uar, weighted_f1=metric_logger.weighted_f1, micro_f1=metric_logger.micro_f1, macro_f1=metric_logger.macro_f1))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def final_test(data_loader, model, device, file, save_feature=False, args=None):
    # criterion = torch.nn.CrossEntropyLoss()
    if args and args.data_set == 'Gaze360':
        criterion = torch.nn.MSELoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()
    

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    final_result = []

    # me: for saving feature in the last layer
    saved_features = {}

    for batch in metric_logger.log_every(data_loader, 10, header):
        videos = batch[0]
        target = batch[1]
        ids = batch[2]
        chunk_nb = batch[3]
        split_nb = batch[4]
        videos = videos.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            # me: for saving feature in the last layer
            if save_feature:
                output, saved_feature = model(videos, save_feature=save_feature)
            else:
                output = model(videos)
            loss = criterion(output, target)

        for i in range(output.size(0)):
            if args and args.data_set == 'Gaze360':
                # 回归任务：保存连续值
                string = "{} {} {} {} {}\n".format(ids[i], 
                                                  str(output.data[i].cpu().numpy().tolist()), 
                                                  str(target[i].cpu().numpy().tolist()),
                                                  str(chunk_nb[i].numpy()), 
                                                  str(split_nb[i].numpy()))
            else:
                # 分类任务：保存类别标签
                string = "{} {} {} {} {}\n".format(ids[i], 
                                                  str(output.data[i].cpu().numpy().tolist()), 
                                                  str(int(target[i].cpu().numpy())),
                                                  str(chunk_nb[i].numpy()), 
                                                  str(split_nb[i].numpy()))
            final_result.append(string)

            # me: for saving feature in the last layer
            if save_feature:
                if ids[i] not in saved_features:
                    saved_features[ids[i]] = {'chunk_id': [], 'split_id': [],
                                              'label': int(target[i].cpu().numpy()),
                                              'feature': [], 'logit': []}
                saved_features[ids[i]]['chunk_id'].append(int(chunk_nb[i].cpu().numpy()))
                saved_features[ids[i]]['split_id'].append(int(split_nb[i].cpu().numpy()))
                saved_features[ids[i]]['feature'].append(saved_feature.data[i].cpu().numpy().tolist())
                saved_features[ids[i]]['logit'].append(output.data[i].cpu().numpy().tolist())

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = videos.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    if not os.path.exists(file):
        os.mknod(file)
    with open(file, 'w') as f:
        f.write("{}, {}\n".format(acc1, acc5))
        for line in final_result:
            f.write(line)

    # me: for saving feature in the last layer
    if save_feature:
        feature_file = file.replace(file[-4:], '_feature.pkl')
        pickle.dump(saved_features, open(feature_file, 'wb'))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def merge(eval_path, num_tasks, args, best=False):
    dict_feats = {}
    dict_label = {}
    dict_pos = {}
    print("Reading individual output files")

    # me: for saving feature in the last layer
    overall_saved_features = {}

    for x in range(num_tasks):
        file = os.path.join(eval_path, str(x) + '.txt') if not best else os.path.join(eval_path, str(x) + '_best.txt')
        lines = open(file, 'r').readlines()[1:]
        for line in lines:
            line = line.strip()
            name = line.split('[')[0]
            
            if args and args.data_set == 'Gaze360':
                # 回归任务的解析
                parts = line.split(']')
                label_str = parts[1].split(' ')[1]
                # 解析标签（可能是列表形式）
                if label_str.startswith('[') and label_str.endswith(']'):
                    label = np.fromstring(label_str[1:-1], dtype=np.float32, sep=',')
                else:
                    label = float(label_str)
                chunk_nb = parts[1].split(' ')[2]
                split_nb = parts[1].split(' ')[3]
                data = np.fromstring(parts[0].split('[')[1], dtype=np.float32, sep=',')
                # 不需要 softmax，保持原始回归值
            else:
                # 分类任务的解析
                label = line.split(']')[1].split(' ')[1]
                chunk_nb = line.split(']')[1].split(' ')[2]
                split_nb = line.split(']')[1].split(' ')[3]
                data = np.fromstring(line.split('[')[1].split(']')[0], dtype=np.float, sep=',')
                data = softmax(data)
                
            if not name in dict_feats:
                dict_feats[name] = []
                dict_label[name] = 0
                dict_pos[name] = []
            if chunk_nb + split_nb in dict_pos[name]:
                continue
            dict_feats[name].append(data)
            dict_pos[name].append(chunk_nb + split_nb)
            dict_label[name] = label

        # me: for saving feature in the last layer
        if args.save_feature:
            feature_file = file.replace(file[-4:], '_feature.pkl')
            saved_features = pickle.load(open(feature_file, 'rb'))
            for sample_id in saved_features.keys():
                if sample_id not in overall_saved_features:
                    overall_saved_features[sample_id] = {
                        'chunk_split_id': [], # the only identifier for each view
                        'label': saved_features[sample_id]['label'],
                        'feature': [], 'prob': []}
                chunk_ids = saved_features[sample_id]['chunk_id']
                split_ids = saved_features[sample_id]['split_id']
                for idx, (chunk_id, split_id) in enumerate(zip(chunk_ids, split_ids)):
                    chunk_split_id = f"{chunk_id}_{split_id}"
                    # avoid repetition
                    if chunk_split_id not in overall_saved_features[sample_id]['chunk_split_id']:
                        overall_saved_features[sample_id]['chunk_split_id'].append(chunk_split_id)
                        overall_saved_features[sample_id]['feature'].append(saved_features[sample_id]['feature'][idx])
                        # NOTE: do softmax, logit -> prob
                        overall_saved_features[sample_id]['prob'].append(softmax(saved_features[sample_id]['logit'][idx]))


    print("Computing final results")

    input_lst = []
    print(len(dict_feats))
    # me: more metrics and save preds
    pred_dict = {'id': [], 'label': [], 'pred': []}
    for i, item in enumerate(dict_feats):
        input_lst.append([i, item, dict_feats[item], dict_label[item]])
        pred = int(np.argmax(np.mean(dict_feats[item], axis=0)))
        label = int(dict_label[item])
        pred_dict['pred'].append(pred)
        pred_dict['label'].append(label)
        pred_dict['id'].append(item.strip())
    # from multiprocessing import Pool
    # p = Pool(4)
    # ans = p.map(compute_video, input_lst)
    # me: disable multi-process because it often gets stuck
    ans = [compute_video(lst) for lst in input_lst]
    top1 = [x[1] for x in ans]
    top5 = [x[2] for x in ans]
    pred = [x[0] for x in ans]
    label = [x[3] for x in ans]
    final_top1 ,final_top5 = np.mean(top1), np.mean(top5)

    # me: for saving feature in the last layer
    if args.save_feature:
        # get avg feature and pred
        for sample_id in overall_saved_features.keys():
            overall_saved_features[sample_id]['feature'] = np.mean(overall_saved_features[sample_id]['feature'], axis=0)
            overall_saved_features[sample_id]['pred'] = int(np.argmax(np.mean(overall_saved_features[sample_id]['prob'], axis=0)))
        feature_file = os.path.join(eval_path, 'overall_feature.pkl') if not best else os.path.join(eval_path, 'overall_feature_best.pkl')
        pickle.dump(overall_saved_features, open(feature_file, 'wb'))

    return final_top1*100 ,final_top5*100, pred_dict

def compute_video(lst):
    i, video_id, data, label = lst
    feat = [x for x in data]
    feat = np.mean(feat, axis=0)
    pred = np.argmax(feat)
    top1 = (int(pred) == int(label)) * 1.0
    top5 = (int(label) in np.argsort(-feat)[:5]) * 1.0
    return [pred, top1, top5, int(label)]
