import os
import time
import random
import shutil
import logging

import click
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn

from .utils.misc import accuracy, disb, AverageMeter
from .utils.eval_metrics import evaluate
from .models import grucls as models
from .dataset import TimeSeriesDataset, get_aiops, jitter, permutation, scaling

log_format = "%(asctime)s [%(levelname)s] - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger(__name__)


def save_checkpoint(state, is_best, checkpoint, filename="checkpoint.pth.tar"):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, "model_best.pth.tar"))


def load_best(model, checkpoint):
    checkpoint = torch.load(os.path.join(checkpoint, "model_best.pth.tar"))
    model.load_state_dict(checkpoint["state_dict"])


class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, mask):
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = -torch.mean(torch.sum(F.log_softmax(outputs_u, dim=1) * targets_u, dim=1) * mask)

        return Lx, Lu


def annotate_unlabeled_data(loader, model, use_cuda):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()

    end = time.time()
    p_bar = tqdm(range(len(loader)))

    logits_scores = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            data_time.update(time.time() - end)

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)

            outputs = model(inputs)
            logits_scores.append(outputs.cpu())

            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            p_bar.set_description(
                "Annotating unlabeled data... ({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | top1: {top1: .4f} | top5: {top5: .4f}".format(
                    batch=batch_idx + 1,
                    size=len(loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                )
            )
            p_bar.update()
        p_bar.close()

    return torch.cat(logits_scores, dim=0)


def get_split(x_train, y_train, x_unlab_test, y_unlab_test, target_disb, num_classes, alpha_t, pseudo_label_list=None):
    if pseudo_label_list is not None:
        x_picked = []
        y_picked = []

        min_index = torch.argmin(target_disb).item()
        for c in range(num_classes):
            num_picked = int(
                len(pseudo_label_list[c]) * np.math.pow(target_disb[min_index] / target_disb[c], 1 / alpha_t)
            )
            idx_picked = pseudo_label_list[c][:num_picked]
            x_picked.append(x_unlab_test[idx_picked])
            y_picked.append(np.ones_like(y_unlab_test[idx_picked]) * c)
            logger.info("class {} is added {} pseudo labels".format(c, num_picked))
        x_picked.append(x_train)
        y_picked.append(y_train)
        x_train = np.concatenate(x_picked, axis=0)
        y_train = np.concatenate(y_picked, axis=0)
    else:
        logger.info("not update")
    logger.info("{} train set images in total".format(len(x_train)))
    return x_train, y_train


def train(
    labeled_train_iter,
    unlabeled_train_iter,
    labeled_trainloader,
    unlabeled_trainloader,
    model,
    optimizer,
    criterion,
    epoch,
    target_disb,
    emp_distb_u,
    current_delta,
    val_iteration,
    num_class,
    use_cuda,
    tau,
):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()
    mask_prob = AverageMeter()
    total_c = AverageMeter()
    used_c = AverageMeter()
    end = time.time()

    p_bar = tqdm(range(val_iteration))

    model.train()
    for batch_idx in range(val_iteration):
        try:
            inputs_x, targets_x = next(labeled_train_iter)
        except:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, targets_x = next(labeled_train_iter)

        try:
            (inputs_u, inputs_u2, inputs_u3), gt_targets_u = next(unlabeled_train_iter)
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            (inputs_u, inputs_u2, inputs_u3), gt_targets_u = next(unlabeled_train_iter)

        data_time.update(time.time() - end)
        batch_size = inputs_x.size(0)

        # Transform label to one-hot
        targets_x = torch.zeros(batch_size, num_class).scatter_(1, targets_x.view(-1, 1), 1)
        if use_cuda:
            inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)
            inputs_u, inputs_u2, inputs_u3 = inputs_u.cuda(), inputs_u2.cuda(), inputs_u3.cuda()

        # Generate the pseudo labels
        with torch.no_grad():
            # Generate the pseudo labels by aggregation and sharpening
            outputs_u = model(inputs_u)
            p = torch.softmax(outputs_u, dim=1)

            # Tracking the empirical distribution on the unlabeled samples
            real_batch_idx = batch_idx + epoch * val_iteration
            if real_batch_idx == 0:
                emp_distb_u = p.mean(0, keepdim=True)
            elif real_batch_idx // 128 == 0:
                emp_distb_u = torch.cat([emp_distb_u, p.mean(0, keepdim=True)], 0)
            else:
                emp_distb_u = emp_distb_u[-127:]
                emp_distb_u = torch.cat([emp_distb_u, p.mean(0, keepdim=True)], 0)

            # Distribution alignment with adjustment parameter scaling
            t_scaled_target_disb = target_disb**current_delta
            t_scaled_target_disb /= t_scaled_target_disb.sum()
            pa = p * (t_scaled_target_disb.cuda() + 1e-6) / (emp_distb_u.mean(0).cuda() + 1e-6)
            p = pa / pa.sum(dim=1, keepdim=True)
            targets_u = p

            max_p, p_hat = torch.max(targets_u, dim=1)
            select_mask = max_p.ge(tau).float()

            total_acc = p_hat.cpu().eq(gt_targets_u).float().view(-1)
            if select_mask.sum() != 0:
                used_c.update(total_acc[select_mask.cpu() != 0].mean(0).item(), select_mask.sum())
            mask_prob.update(select_mask.mean().item())
            total_c.update(total_acc.mean(0).item())

            p_hat = torch.zeros(batch_size, num_class).cuda().scatter_(1, p_hat.view(-1, 1), 1)
            select_mask = torch.cat([select_mask, select_mask], 0)

        all_inputs = torch.cat([inputs_x, inputs_u2, inputs_u3], dim=0)
        all_targets = torch.cat([targets_x, p_hat, p_hat], dim=0)

        all_outputs = model(all_inputs)
        logits_x = all_outputs[:batch_size]
        logits_u = all_outputs[batch_size:]

        Lx, Lu = criterion(logits_x, all_targets[:batch_size], logits_u, all_targets[batch_size:], select_mask)
        loss = Lx + Lu

        losses.update(loss.item(), inputs_x.size(0))
        losses_x.update(Lx.item(), inputs_x.size(0))
        losses_u.update(Lu.item(), inputs_x.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        p_bar.set_description(
            "Train Epoch: ({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | "
            "Loss: {loss:.4f} | Loss_x: {loss_x:.4f} | Loss_u: {loss_u:.4f} | Mask: {mask:.4f}| "
            "Use_acc: {used_acc:.4f}".format(
                batch=batch_idx + 1,
                size=val_iteration,
                data=data_time.avg,
                bt=batch_time.avg,
                loss=losses.avg,
                loss_x=losses_x.avg,
                loss_u=losses_u.avg,
                mask=mask_prob.avg,
                used_acc=used_c.avg,
            )
        )
        p_bar.update()
    p_bar.close()

    return (losses.avg, losses_x.avg, losses_u.avg, mask_prob.avg, total_c.avg, used_c.avg, emp_distb_u)


def validate(valloader, model, criterion, use_cuda, mode="Valid", need_evaluate=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.eval()

    end = time.time()
    valloader = tqdm(valloader)

    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            data_time.update(time.time() - end)
            y_true.extend(targets.tolist())

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            losses.update(loss.item(), inputs.size(0))

            pred_label = outputs.max(1)[1]
            y_pred.extend(pred_label.tolist())

            batch_time.update(time.time() - end)
            end = time.time()

            valloader.set_description(
                mode
                + " ({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Loss: {loss:.4f}".format(
                    batch=batch_idx + 1,
                    size=len(valloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                )
            )
        if need_evaluate:
            evaluate(y_true, y_pred)
        valloader.close()

    return losses.avg


@click.command()
# Optimization options
@click.option("--epochs", type=int, default=100, help="number of total epochs to run")
@click.option("--num-workers", type=int, default=0, help="number of workers")
@click.option("--batch-size", type=int, default=32, help="train batchsize")
@click.option("--lr", "--learning-rate", type=float, default=1e-4, help="initial learning rate")
# Checkpoints
@click.option("--resume", type=str, default="", help="path to latest checkpoint (default: none)")
@click.option("--out", type=str, default="results", help="directory to output the result")
# Miscs
@click.option("--seed", type=int, default=0, help="seed")
@click.option("--file-path", type=str, default="data/rca_multimodal_window_size_11.pkl", help="data path")
# Device options
@click.option("--gpu", type=str, default="0", help="id(s) for CUDA_VISIBLE_DEVICES")
# Method options
@click.option("--entity-type", type=str, default="pod", help="pod or node")
@click.option("--ratio", type=float, default=0.3, help="relative size between labeled and all data")
# Hyperparameters for STRCA
@click.option("--tau", type=float, default=0.95, help="threshold for pseudo-labels")
@click.option("--num-generation", type=int, default=5, help="number of generations")
@click.option("--delta-min", type=float, default=0.5, help="delta used in the final generation")
@click.option("--alpha-t", type=float, default=3, help="1/alpha")
def main(
    epochs,
    num_workers,
    batch_size,
    lr,
    resume,
    out,
    seed,
    file_path,
    gpu,
    entity_type,
    ratio,
    tau,
    num_generation,
    delta_min,
    alpha_t,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    use_cuda = torch.cuda.is_available()

    if seed is None:
        seed = random.randint(1, 10000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    if use_cuda:
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    os.makedirs(out, exist_ok=True)
    best_loss = None
    if entity_type == "pod":
        num_class = 10
        input_size = 19
    elif entity_type == "node":
        num_class = 7
        input_size = 24
    else:
        raise NotImplementedError

    logger.info(f"Getting class-imbalanced {entity_type}")

    pseudo_label_list = None
    if resume:
        tmp = resume.split("/")
        start_gen = int(tmp[-1][-9])
    else:
        start_gen = 0
    for gen_idx in range(start_gen, num_generation):
        cur = gen_idx / (num_generation - 1)
        current_delta = (1.0 - cur) * 1.0 + cur * delta_min

        lb_data, lb_target, ulb_data_test, ulb_target_test, eval_data, eval_target, test_data, test_target = get_aiops(
            file_path, entity_type, ratio, seed
        )

        ulb_data = np.concatenate([lb_data, ulb_data_test], axis=0)
        ulb_target = np.concatenate([lb_target, ulb_target_test], axis=0)
        target_disb = disb(lb_target, num_class)
        lb_data, lb_target = get_split(
            lb_data,
            lb_target,
            ulb_data_test,
            ulb_target_test,
            target_disb,
            num_class,
            alpha_t=alpha_t,
            pseudo_label_list=pseudo_label_list,
        )

        train_labeled_set = TimeSeriesDataset(lb_data, lb_target)
        train_unlabeled_set = TimeSeriesDataset(
            ulb_data, ulb_target, transform=lambda ts: (scaling(ts), jitter(permutation(ts)), jitter(permutation(ts)))
        )
        unlabeled_anno_set = TimeSeriesDataset(ulb_data_test, ulb_target_test)
        eval_set = TimeSeriesDataset(eval_data, eval_target)
        test_set = TimeSeriesDataset(test_data, test_target)

        labeled_trainloader = data.DataLoader(
            train_labeled_set, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers
        )
        unlabeled_trainloader = data.DataLoader(
            train_unlabeled_set, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers
        )
        unlabeled_anno_loader = data.DataLoader(
            unlabeled_anno_set, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers
        )
        eval_loader = data.DataLoader(eval_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        val_iteration = len(labeled_trainloader)

        logger.info("Creating Model")

        def create_model():
            model = models.build_grucls(input_size=input_size, hidden_size=128, num_layers=3, num_classes=num_class)
            model.cuda()
            return model

        model = create_model()

        logger.info("Total params: %.2fM" % (sum(p.numel() for p in create_model().parameters()) / 1000000.0))

        train_criterion = SemiLoss()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

        total_steps = val_iteration * epochs
        start_epoch = 0

        if resume:
            logger.info("Resuming from checkpoint..")
            assert os.path.isfile(resume), "Error: no checkpoint directory found!"
            out = os.path.dirname(resume)
            checkpoint = torch.load(resume)
            start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])

        logger.info("===== Start training =====")
        logger.info(f"  Task = {entity_type}")
        logger.info(f"  Num Epochs = {epochs}")
        logger.info(f"  Total train batch size = {batch_size}")
        logger.info(f"  Total optimization steps = {total_steps}")

        emp_distb_u = torch.ones(num_class) / num_class
        labeled_train_iter = iter(labeled_trainloader)
        unlabeled_train_iter = iter(unlabeled_trainloader)
        for epoch in range(start_epoch, epochs):
            logger.info("Epoch: [%d | %d] LR: %f" % (epoch + 1, epochs, optimizer.param_groups[0]["lr"]))

            *train_info, emp_distb_u = train(
                labeled_train_iter,
                unlabeled_train_iter,
                labeled_trainloader,
                unlabeled_trainloader,
                model,
                optimizer,
                train_criterion,
                epoch,
                target_disb,
                emp_distb_u,
                current_delta,
                val_iteration,
                num_class,
                use_cuda,
                tau,
            )

            valid_loss = validate(eval_loader, model, criterion, use_cuda, need_evaluate=False)
            is_best = False
            if best_loss is None or valid_loss < best_loss:
                is_best = True
                best_loss = valid_loss
                best_generation = gen_idx
                best_epoch = epoch

            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                is_best,
                out,
                filename=f"checkpoint_gen{gen_idx}.pth.tar",
            )

        with torch.no_grad():
            load_best(model, out)
            logits_scores = annotate_unlabeled_data(unlabeled_anno_loader, model, use_cuda)

            y_pred = torch.argmax(logits_scores, dim=1)
            y_score = torch.max(logits_scores, dim=1)[0]

            pseudo_label_list = (
                []
            )  # list of np.arr, each of which contains idx of data that has label class_idx (sorted)
            for class_idx in range(num_class):
                idx_gather = torch.nonzero(y_pred == class_idx).view(-1)
                score_gather = y_score[idx_gather]
                _, order = score_gather.sort(descending=True)
                idx_gather = idx_gather[order]
                pseudo_label_list.append(idx_gather.numpy())

        logger.info(f"Load Best Checkpoint at generation {best_generation} epoch {best_epoch}")
        load_best(model, out)
        validate(test_loader, model, criterion, use_cuda, "Test", need_evaluate=True)


if __name__ == "__main__":
    main()
