"""
Tester

Author: pedro sidra (pedro.freitas@inf.ufrgs.br)
Please cite our work if the code is helpful to you.
"""

from pathlib import Path
import wandb
import json
from uuid import uuid4
import os
import time
import numpy as np
from collections import OrderedDict
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.data

from pointcept.engines.defaults import create_ddp_model
import pointcept.utils.comm as comm
from pointcept.datasets import build_dataset, collate_fn
from pointcept.models import build_model
from pointcept.utils.logger import get_root_logger
from pointcept.utils.registry import Registry
from pointcept.utils.misc import (
    AverageMeter,
    intersection_and_union,
    intersection_and_union_gpu,
    make_dirs,
)
from pointcept.engines.test import TESTERS

try:
    import pointops
except:
    pointops = None


from sculpting.utils import dict_to_cuda


class TesterBase:
    def __init__(self, cfg, model=None, test_loader=None, verbose=False) -> None:
        torch.multiprocessing.set_sharing_strategy("file_system")
        self.logger = get_root_logger(
            log_file=os.path.join(cfg.save_path, "test.log"),
            file_mode="a" if cfg.resume else "w",
        )
        self.logger.info("=> Loading config ...")
        self.cfg = cfg
        self.verbose = verbose
        if self.verbose and model is None:
            # if model is not none, trigger tester with trainer, no need to print config
            self.logger.info(f"Save path: {cfg.save_path}")
            self.logger.info(f"Config:\n{cfg.pretty_text}")
        if model is None:
            self.logger.info("=> Building model ...")
            self.model = self.build_model()
        else:
            self.model = model
        if test_loader is None:
            self.logger.info("=> Building test dataset & dataloader ...")
            self.test_loader = self.build_test_loader()
        else:
            self.test_loader = test_loader

    def build_model(self):
        model = build_model(self.cfg.model)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"Num params: {n_parameters}")
        model = create_ddp_model(
            model.cuda(),
            broadcast_buffers=False,
            find_unused_parameters=self.cfg.find_unused_parameters,
        )
        if os.path.isfile(self.cfg.weight):
            self.logger.info(f"Loading weight at: {self.cfg.weight}")
            checkpoint = torch.load(self.cfg.weight, weights_only=False)
            weight = OrderedDict()
            for key, value in checkpoint["state_dict"].items():
                if key.startswith("module."):
                    if comm.get_world_size() == 1:
                        key = key[7:]  # module.xxx.xxx -> xxx.xxx
                else:
                    if comm.get_world_size() > 1:
                        key = "module." + key  # xxx.xxx -> module.xxx.xxx
                weight[key] = value
            model.load_state_dict(weight, strict=True)
            self.logger.info(
                "=> Loaded weight '{}' (epoch {})".format(
                    self.cfg.weight, checkpoint["epoch"]
                )
            )
        else:
            raise RuntimeError("=> No checkpoint found at '{}'".format(self.cfg.weight))
        return model

    def build_test_loader(self):
        test_dataset = build_dataset(self.cfg.data.test)
        if comm.get_world_size() > 1:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        else:
            test_sampler = None
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.cfg.batch_size_test_per_gpu,
            shuffle=False,
            num_workers=self.cfg.batch_size_test_per_gpu,
            pin_memory=True,
            sampler=test_sampler,
            collate_fn=self.__class__.collate_fn,
        )
        return test_loader

    def test(self):
        raise NotImplementedError

    @staticmethod
    def collate_fn(batch):
        raise collate_fn(batch)


@TESTERS.register_module()
class SemSegFragmentTester(TesterBase):
    def __init__(self, cfg, model=None, test_loader=None, verbose=False):

        if comm.is_main_process():
            exp_path = Path(cfg.save_path)
            exp_path.mkdir(exist_ok=True, parents=True)
            wandb.init(
                name=exp_path.name,
                project=exp_path.parent.name,
                config=cfg._cfg_dict,
                sync_tensorboard=True,
                save_code=True
            )

        weight_not_exist = (cfg.weight) and (not Path(cfg.weight).is_file())
        if weight_not_exist:
            download = wandb.Api().artifact(cfg.weight).download()
            if(Path(download) / "model_last.pth").exists():
                cfg.weight = Path(download) / "model_last.pth"
            else:
                cfg.weight = next(Path(download).glob("*.pth"))

        super().__init__(cfg, model, test_loader, verbose)

    def test(self):
        # Create meters, path vars, model, etc.
        self.setup()
        comm.synchronize()

        # Run fragment-based inference on the full dataset
        # (per worker)
        self.record = {}
        self.fragment_infer()

        # Sync all workers and gather results
        self.logger.info("Syncing ...")
        comm.synchronize()
        record_sync = comm.gather(self.record, dst=0)

        # Main worker logs the results
        if comm.is_main_process():
            self.log_final_metrics(record_sync)

    def setup(self):
        # Wandb setup
        # Default setup
        assert self.test_loader.batch_size == 1
        self.logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")

        self.batch_time = AverageMeter()
        self.intersection_meter = AverageMeter()
        self.union_meter = AverageMeter()
        self.target_meter = AverageMeter()
        self.model.eval()

        self.save_path = os.path.join(self.cfg.save_path, "result")
        make_dirs(self.save_path)

        # create submit folder only on main process
        if (
            self.cfg.data.test.type == "ScanNetDataset"
            or self.cfg.data.test.type == "ScanNet200Dataset"
            or self.cfg.data.test.type == "ScanNetPPDataset"
        ) and comm.is_main_process():
            make_dirs(os.path.join(self.save_path, "submit"))
        elif (
            self.cfg.data.test.type == "SemanticKITTIDataset" and comm.is_main_process()
        ):
            make_dirs(os.path.join(self.save_path, "submit"))
        elif self.cfg.data.test.type == "NuScenesDataset" and comm.is_main_process():
            import json

            make_dirs(os.path.join(self.save_path, "submit", "lidarseg", "test"))
            make_dirs(os.path.join(self.save_path, "submit", "test"))
            submission = dict(
                meta=dict(
                    use_camera=False,
                    use_lidar=True,
                    use_radar=False,
                    use_map=False,
                    use_external=False,
                )
            )
            with open(
                os.path.join(self.save_path, "submit", "test", "submission.json"), "w"
            ) as f:
                json.dump(submission, f, indent=4)

    def to_cuda(self, input_dict):
        for key in input_dict.keys():
            if isinstance(input_dict[key], torch.Tensor):
                input_dict[key] = input_dict[key].cuda(non_blocking=True)

    def try_load_preds(self, pred_save_path, idx):
        if os.path.isfile(pred_save_path):
            self.logger.info(
                "{}/{}: {}, loaded pred and label.".format(
                    idx + 1, len(self.test_loader), Path(pred_save_path).stem
                )
            )
            return np.load(pred_save_path)
        else:
            return None

    def fragment_list_inference(self, total_size, fragment_list, idx, data_name):

        pred = torch.zeros((total_size, self.cfg.data.num_classes)).cuda()
        for i in range(len(fragment_list)):
            fragment_batch_size = 1
            s_i, e_i = i * fragment_batch_size, min(
                (i + 1) * fragment_batch_size, len(fragment_list)
            )

            input_dict = collate_fn(fragment_list[s_i:e_i])
            dict_to_cuda(input_dict)

            idx_part = input_dict["index"]
            with torch.no_grad():
                pred_part = self.model(input_dict)["seg_logits"]  # (n, k)
                pred_part = F.softmax(pred_part, -1)
                if self.cfg.empty_cache:
                    torch.cuda.empty_cache()
                bs = 0
                for be in input_dict["offset"]:
                    pred[idx_part[bs:be], :] += pred_part[bs:be]
                    bs = be

            self.logger.info(
                "Test: {}/{}-{data_name}, Batch: {batch_idx}/{batch_num}".format(
                    idx + 1,
                    len(self.test_loader),
                    data_name=data_name,
                    batch_idx=i,
                    batch_num=len(fragment_list),
                )
            )

        return pred

    def pred_postprocess(self, data_dict, pred, segment):
        if self.cfg.data.test.type == "ScanNetPPDataset":
            pred = pred.topk(3, dim=1)[1].data.cpu().numpy()
        else:
            pred = pred.max(1)[1].data.cpu().numpy()

        if "origin_segment" in data_dict.keys():
            assert "inverse" in data_dict.keys()
            pred = pred[data_dict["inverse"]]
            segment = data_dict["origin_segment"]

        return pred, segment

    def make_submissions(self, pred, data_name):
        if (
            self.cfg.data.test.type == "ScanNetDataset"
            or self.cfg.data.test.type == "ScanNet200Dataset"
        ):
            np.savetxt(
                os.path.join(self.save_path, "submit", "{}.txt".format(data_name)),
                self.test_loader.dataset.class2id[pred].reshape([-1, 1]),
                fmt="%d",
            )
        elif self.cfg.data.test.type == "ScanNetPPDataset":
            np.savetxt(
                os.path.join(self.save_path, "submit", "{}.txt".format(data_name)),
                pred.astype(np.int32),
                delimiter=",",
                fmt="%d",
            )
            pred = pred[:, 0]  # for mIoU, TODO: support top3 mIoU
        elif self.cfg.data.test.type == "SemanticKITTIDataset":
            # 00_000000 -> 00, 000000
            sequence_name, frame_name = data_name.split("_")
            os.makedirs(
                os.path.join(
                    self.save_path,
                    "submit",
                    "sequences",
                    sequence_name,
                    "predictions",
                ),
                exist_ok=True,
            )
            submit = pred.astype(np.uint32)
            submit = np.vectorize(
                self.test_loader.dataset.learning_map_inv.__getitem__
            )(submit).astype(np.uint32)
            submit.tofile(
                os.path.join(
                    self.save_path,
                    "submit",
                    "sequences",
                    sequence_name,
                    "predictions",
                    f"{frame_name}.label",
                )
            )
        elif self.cfg.data.test.type == "NuScenesDataset":
            np.array(pred + 1).astype(np.uint8).tofile(
                os.path.join(
                    self.save_path,
                    "submit",
                    "lidarseg",
                    "test",
                    "{}_lidarseg.bin".format(data_name),
                )
            )

    def update_metrics_batch(self, pred, segment, data_name, idx, start):
        intersection, union, target = intersection_and_union(
            pred, segment, self.cfg.data.num_classes, self.cfg.data.ignore_index
        )

        self.intersection_meter.update(intersection)
        self.union_meter.update(union)
        self.target_meter.update(target)
        self.record[data_name] = dict(
            intersection=intersection, union=union, target=target
        )

        mask = union != 0
        iou_class = intersection / (union + 1e-10)
        iou = np.mean(iou_class[mask])
        acc = sum(intersection) / (sum(target) + 1e-10)

        m_iou = np.mean(self.intersection_meter.sum / (self.union_meter.sum + 1e-10))
        m_acc = np.mean(self.intersection_meter.sum / (self.target_meter.sum + 1e-10))

        self.batch_time.update(time.time() - start)
        self.logger.info(
            "Test: {} [{}/{}]-{} "
            "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
            "Accuracy {acc:.4f} ({m_acc:.4f}) "
            "mIoU {iou:.4f} ({m_iou:.4f})".format(
                data_name,
                idx + 1,
                len(self.test_loader),
                segment.size,
                batch_time=self.batch_time,
                acc=acc,
                m_acc=m_acc,
                iou=iou,
                m_iou=m_iou,
            )
        )

    def fragment_infer(self):
        for idx, data_dict in enumerate(self.test_loader):
            start = time.time()

            data_dict = data_dict[0]  # current assume batch size is 1
            fragment_list = data_dict.pop("fragment_list")
            segment = data_dict.pop("segment")
            data_name = data_dict.pop("name")

            pred_save_path = os.path.join(
                self.save_path, "{}_pred.npy".format(data_name)
            )
            loaded_preds = self.try_load_preds(pred_save_path, idx)

            if loaded_preds is not None:
                pred = loaded_preds
                if "origin_segment" in data_dict.keys():
                    segment = data_dict["origin_segment"]
            else:
                pred = self.fragment_list_inference(
                    total_size=segment.size,
                    fragment_list=fragment_list,
                    idx=idx,
                    data_name=data_name,
                )

                pred, segment = self.pred_postprocess(data_dict, pred, segment)

                np.save(pred_save_path, pred)

            self.make_submissions(pred, data_name)
            self.update_metrics_batch(pred, segment, data_name, idx, start)

    def log_final_metrics(self, record_sync):
        record = {}
        for _ in range(len(record_sync)):
            r = record_sync.pop()
            record.update(r)
            del r
        intersection = np.sum(
            [meters["intersection"] for _, meters in record.items()], axis=0
        )
        union = np.sum([meters["union"] for _, meters in record.items()], axis=0)
        target = np.sum([meters["target"] for _, meters in record.items()], axis=0)

        if self.cfg.data.test.type == "S3DISDataset":
            torch.save(
                dict(intersection=intersection, union=union, target=target),
                os.path.join(self.save_path, f"{self.test_loader.dataset.split}.pth"),
            )

        iou_class = intersection / (union + 1e-10)
        accuracy_class = intersection / (target + 1e-10)
        mIoU = np.mean(iou_class)
        mAcc = np.mean(accuracy_class)
        allAcc = sum(intersection) / (sum(target) + 1e-10)

        self.logger.info(
            "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}".format(
                mIoU, mAcc, allAcc
            )
        )
        wandb.summary["miou"] = mIoU
        wandb.summary["macc"] = mAcc
        wandb.summary["allacc"] = allAcc

        for i in range(self.cfg.data.num_classes):
            wandb.summary[f"{self.cfg.data.names[i]}_iou"] = iou_class[i]
            wandb.summary[f"{self.cfg.data.names[i]}_acc"] = accuracy_class[i]
            self.logger.info(
                "Class_{idx} - {name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
                    idx=i,
                    name=self.cfg.data.names[i],
                    iou=iou_class[i],
                    accuracy=accuracy_class[i],
                )
            )
        self.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    @staticmethod
    def collate_fn(batch):
        return batch
