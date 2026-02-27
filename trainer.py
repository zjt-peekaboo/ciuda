# trainer.py
"""
E3P Trainer: source pre-training + incremental target adaptation.
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from tqdm import tqdm
from loguru import logger
from typing import Dict, List, Optional, Tuple

from config import E3PConfig
from model import E3PModel
from energy_reliability import ReliabilityEstimator
from prototype import PrototypeLibrary
from buffer import ContributionBuffer
from dataset import (
    Office31Dataset, TaskSplitter, MultiViewDataset,
    get_transforms,
)


class SourceTrainer:
    """Train source model on labeled source domain (standard supervised)."""

    def __init__(self, config: E3PConfig):
        self.config = config
        self.device = torch.device(config.device)

    def train(self, model: E3PModel, train_dataset: Office31Dataset) -> E3PModel:
        logger.info("=" * 60)
        logger.info("Source domain training started")
        logger.info("=" * 60)

        model.unfreeze_all()
        model.to(self.device)

        loader = DataLoader(
            train_dataset,
            batch_size=self.config.source_batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            drop_last=True,
            pin_memory=True,
        )

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.config.source_lr,
            momentum=0.9,
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.source_epochs
        )
        criterion = nn.CrossEntropyLoss()

        best_acc = 0.0
        for epoch in range(self.config.source_epochs):
            model.train()
            total_loss = 0.0
            correct = 0
            total = 0

            pbar = tqdm(loader, desc=f"Source Epoch {epoch+1}/{self.config.source_epochs}")
            for images, labels, _ in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                logits, _ = model(images, use_adapter=False)
                loss = criterion(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * images.size(0)
                correct += (logits.argmax(1) == labels).sum().item()
                total += images.size(0)
                pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100*correct/total:.1f}%")

            scheduler.step()
            epoch_acc = 100 * correct / total
            epoch_loss = total_loss / total
            logger.info(
                f"Source Epoch {epoch+1}: loss={epoch_loss:.4f}, acc={epoch_acc:.1f}%"
            )

            if epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), self.config.source_model_path)

        logger.info(f"Source training done. Best acc: {best_acc:.1f}%")
        model.load_state_dict(torch.load(self.config.source_model_path, map_location=self.device))
        return model


class TargetTrainer:
    """Incremental target adaptation using E3P."""

    def __init__(self, config: E3PConfig, splitter: TaskSplitter):
        self.config = config
        self.device = torch.device(config.device)
        self.splitter = splitter

        self.reliability_estimator = ReliabilityEstimator(
            temperature=config.energy_temperature
        )
        self.prototype_lib = PrototypeLibrary(
            feature_dim=256,  # bottleneck dim
            ema_beta=config.proto_ema_beta,
            temperature=config.proto_temperature,
            reliability_threshold=config.reliability_threshold_gamma,
            device=config.device,
        )
        self.buffer = ContributionBuffer(
            total_capacity=config.buffer_total_capacity,
            sweet_spot_m=config.buffer_sweet_spot_m,
            feature_dim=256,
        )

        # Track metrics
        self.task_accuracies: Dict[int, Dict[int, float]] = {}

    def adapt(
        self,
        source_model: E3PModel,
        target_base_dataset: Office31Dataset,
        eval_dataset: Office31Dataset,
    ):
        """
        Run incremental adaptation across all tasks.
        Args:
            source_model: trained source model
            target_base_dataset: full target domain dataset (we filter per task)
            eval_dataset: full target domain for evaluation
        """
        logger.info("=" * 60)
        logger.info("Target incremental adaptation started")
        logger.info("=" * 60)

        # Initialize target model from source
        target_model = copy.deepcopy(source_model)
        target_model.freeze_backbone()
        target_model.to(self.device)

        # Keep a copy of previous task model for distillation
        prev_model = None

        num_tasks = len(self.config.task_sizes)

        for task_id in range(num_tasks):
            logger.info("-" * 50)
            logger.info(
                f"Task {task_id+1}/{num_tasks}: "
                f"{self.config.task_sizes[task_id]} new classes, "
                f"total seen: {sum(self.config.task_sizes[:task_id+1])}"
            )
            logger.info("-" * 50)

            # Get current task classes (original indices)
            task_classes_orig = self.splitter.task_classes[task_id]
            all_classes_orig = self.splitter.get_classes_up_to_task(task_id)

            # Filter target dataset for current task classes
            task_dataset = Office31Dataset(
                root=self.config.data_root,
                domain=self.config.target_domain,
                transform=get_transforms(self.config.img_size, "train"),
                class_indices=task_classes_orig,
                class_mapping=self.splitter.class_mapping,
            )

            # Multi-view dataset for reliability estimation
            mv_dataset = MultiViewDataset(
                Office31Dataset(
                    root=self.config.data_root,
                    domain=self.config.target_domain,
                    transform=None,  # MultiViewDataset handles transforms
                    class_indices=task_classes_orig,
                    class_mapping=self.splitter.class_mapping,
                ),
                img_size=self.config.img_size,
            )

            # --- Phase 1: Estimate reliability & discover classes ---
            logger.info("Phase 1: Reliability estimation & class discovery")
            (
                all_reliabilities,
                all_pseudo_labels,
                all_features,
                all_images,
            ) = self._estimate_reliability_all(target_model, mv_dataset)

            # Discover classes in this task
            discovered = self._discover_classes(
                all_reliabilities, all_pseudo_labels, task_id
            )
            logger.info(f"Discovered classes (mapped): {discovered}")

            # --- Phase 2: Update prototypes ---
            logger.info("Phase 2: Prototype update")
            self.prototype_lib.update(
                all_features.to(self.device),
                all_pseudo_labels.to(self.device),
                all_reliabilities.to(self.device),
            )
            self.prototype_lib.log_status()

            # --- Phase 3: Correct pseudo-labels via prototype propagation ---
            logger.info("Phase 3: Pseudo-label correction")
            model_probs = F.softmax(
                self._get_all_logits(target_model, mv_dataset), dim=1
            )
            corrected_labels, corrected_probs = self.prototype_lib.correct_pseudo_labels(
                all_features.to(self.device),
                model_probs.to(self.device),
                all_reliabilities.to(self.device),
                self.config.num_classes,
            )
            corrected_labels = corrected_labels.cpu()

            # --- Phase 4: Update buffer ---
            logger.info("Phase 4: Buffer update")
            self._update_buffer(
                discovered, all_images, all_features,
                corrected_labels, all_reliabilities,
            )
            self.buffer.log_status()

            # --- Phase 5: Train target model ---
            logger.info("Phase 5: Target model training")
            prev_model_copy = copy.deepcopy(target_model) if prev_model is None else prev_model
            target_model = self._train_task(
                target_model,
                prev_model_copy,
                all_images,
                all_features,
                corrected_labels,
                all_reliabilities,
                task_id,
            )

            # Save previous model for next task distillation
            prev_model = copy.deepcopy(target_model)

            # --- Evaluate on all seen classes ---
            all_seen_mapped = self.splitter.get_all_mapped_classes_up_to(task_id)
            acc = self._evaluate(target_model, eval_dataset, all_seen_mapped, task_id)
            self.task_accuracies[task_id] = {"all_seen_acc": acc}

            logger.info(f"Task {task_id+1} done. Accuracy on all seen classes: {acc:.2f}%")

        # Final summary
        self._print_summary()
        return target_model

    @torch.no_grad()
    def _estimate_reliability_all(
        self,
        model: E3PModel,
        mv_dataset: MultiViewDataset,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Estimate reliability for all samples in the multi-view dataset."""
        loader = DataLoader(
            mv_dataset,
            batch_size=self.config.target_batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

        all_rel, all_pl, all_feat, all_img = [], [], [], []

        for batch in tqdm(loader, desc="Reliability estimation"):
            views = {
                "weak": batch["weak"],
                "strong": batch["strong"],
                "test": batch["test"],
            }
            rel, pl, feat = self.reliability_estimator.compute_reliability(
                model, views, self.device
            )
            all_rel.append(rel.cpu())
            all_pl.append(pl.cpu())
            all_feat.append(feat.cpu())
            all_img.append(batch["test"])  # store test-view images

        return (
            torch.cat(all_rel),
            torch.cat(all_pl),
            torch.cat(all_feat),
            torch.cat(all_img),
        )

    @torch.no_grad()
    def _get_all_logits(
        self, model: E3PModel, mv_dataset: MultiViewDataset
    ) -> torch.Tensor:
        """Get model logits for all samples (test view)."""
        loader = DataLoader(
            mv_dataset,
            batch_size=self.config.target_batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )
        model.eval()
        all_logits = []
        for batch in loader:
            logits, _ = model(batch["test"].to(self.device))
            all_logits.append(logits.cpu())
        return torch.cat(all_logits)

    def _discover_classes(
        self,
        reliability: torch.Tensor,
        pseudo_labels: torch.Tensor,
        task_id: int,
    ) -> List[int]:
        """
        Discover which classes are present in current task data.
        Uses weighted voting with reliability scores.
        """
        vote = torch.zeros(self.config.num_classes)
        for c in range(self.config.num_classes):
            mask = pseudo_labels == c
            if mask.sum() > 0:
                vote[c] = (reliability[mask]).sum().item()

        threshold = self.config.class_detect_alpha
        discovered = [c for c in range(self.config.num_classes) if vote[c] >= threshold]

        # Also include classes from previous tasks (already in buffer)
        for c in self.buffer.seen_classes:
            if c not in discovered:
                discovered.append(c)
        discovered.sort()
        return discovered

    def _update_buffer(
        self,
        discovered_classes: List[int],
        images: torch.Tensor,
        features: torch.Tensor,
        pseudo_labels: torch.Tensor,
        reliability: torch.Tensor,
    ):
        """Update buffer with contribution-driven selection."""
        for c in discovered_classes:
            mask = pseudo_labels == c
            if mask.sum() == 0:
                continue

            c_images = images[mask]
            c_features = features[mask]
            c_labels = pseudo_labels[mask]
            c_reliability = reliability[mask]

            # Get prototype for this class
            if self.prototype_lib.has_class(c):
                proto = self.prototype_lib.prototypes[c]
            else:
                proto = F.normalize(c_features.mean(dim=0), dim=0).to(self.device)

            self.buffer.update_class(
                class_id=c,
                images=c_images,
                features=c_features,
                labels=c_labels,
                reliability=c_reliability,
                prototype=proto.cpu(),
            )

    def _train_task(
        self,
        model: E3PModel,
        prev_model: E3PModel,
        task_images: torch.Tensor,
        task_features: torch.Tensor,
        task_labels: torch.Tensor,
        task_reliability: torch.Tensor,
        task_id: int,
    ) -> E3PModel:
        """
        Train target model on current task data + buffer replay.
        Loss = lambda1 * L_pl + lambda2 * L_proto + lambda3 * L_dist
        """
        model.train()
        # Freeze backbone, only adapter + classifier trainable
        model.freeze_backbone()

        prev_model.eval()
        prev_model.to(self.device)
        for p in prev_model.parameters():
            p.requires_grad = False

        # Build training data: current task + buffer
        buf_images, buf_labels = self.buffer.get_all_samples()

        # Create combined dataset
        train_imgs = [task_images]
        train_lbls = [task_labels]
        train_rels = [task_reliability]

        if buf_images.numel() > 0:
            buf_rel = torch.ones(buf_labels.size(0)) * 0.8  # buffer samples are pre-filtered
            train_imgs.append(buf_images)
            train_lbls.append(buf_labels)
            train_rels.append(buf_rel)

        all_imgs = torch.cat(train_imgs, dim=0)
        all_lbls = torch.cat(train_lbls, dim=0)
        all_rels = torch.cat(train_rels, dim=0)

        dataset = TensorDataset(all_imgs, all_lbls, all_rels)
        loader = DataLoader(
            dataset,
            batch_size=self.config.target_batch_size,
            shuffle=True,
            num_workers=0,  # TensorDataset in memory
            drop_last=len(dataset) > self.config.target_batch_size,
            pin_memory=True,
        )

        optimizer = torch.optim.SGD(
            model.get_trainable_params(),
            lr=self.config.target_lr,
            momentum=0.9,
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.target_epochs_per_task
        )

        proto_matrix, proto_class_ids = self.prototype_lib.get_prototype_matrix()
        proto_matrix = proto_matrix.to(self.device)

        for epoch in range(self.config.target_epochs_per_task):
            epoch_loss = 0.0
            epoch_pl = 0.0
            epoch_proto = 0.0
            epoch_dist = 0.0
            n_batches = 0

            pbar = tqdm(
                loader,
                desc=f"  Task {task_id+1} Epoch {epoch+1}/{self.config.target_epochs_per_task}",
            )
            for imgs, lbls, rels in pbar:
                imgs = imgs.to(self.device)
                lbls = lbls.to(self.device)
                rels = rels.to(self.device)

                logits, feats = model(imgs)

                # --- L_pl: reliability-weighted pseudo-label CE ---
                log_probs = F.log_softmax(logits, dim=1)
                ce_per_sample = F.nll_loss(log_probs, lbls, reduction="none")
                loss_pl = (rels * ce_per_sample).mean()

                # --- L_proto: prototype contrastive loss ---
                loss_proto = torch.tensor(0.0, device=self.device)
                if proto_matrix.size(0) > 1:
                    feat_norm = F.normalize(feats, dim=1)
                    sim = torch.mm(feat_norm, proto_matrix.t()) / self.config.proto_temperature

                    # For each sample, find its prototype index
                    proto_targets = []
                    for lbl in lbls:
                        lbl_item = lbl.item()
                        if lbl_item in proto_class_ids:
                            proto_targets.append(proto_class_ids.index(lbl_item))
                        else:
                            # Assign to nearest prototype
                            proto_targets.append(sim[0].argmax().item())

                    proto_targets = torch.tensor(proto_targets, device=self.device, dtype=torch.long)
                    loss_proto = F.cross_entropy(sim, proto_targets)

                # --- L_dist: self-distillation from previous model ---
                loss_dist = torch.tensor(0.0, device=self.device)
                if task_id > 0:
                    with torch.no_grad():
                        prev_logits, _ = prev_model(imgs)
                        prev_probs = F.softmax(prev_logits / 2.0, dim=1)
                    curr_log_probs = F.log_softmax(logits / 2.0, dim=1)
                    loss_dist = F.kl_div(
                        curr_log_probs, prev_probs, reduction="batchmean"
                    ) * (2.0 ** 2)

                # --- Total loss ---
                loss = (
                    self.config.lambda_pl * loss_pl
                    + self.config.lambda_proto * loss_proto
                    + self.config.lambda_dist * loss_dist
                )

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.get_trainable_params(), max_norm=5.0)
                optimizer.step()

                epoch_loss += loss.item()
                epoch_pl += loss_pl.item()
                epoch_proto += loss_proto.item()
                epoch_dist += loss_dist.item()
                n_batches += 1

                pbar.set_postfix(
                    L=f"{loss.item():.3f}",
                    pl=f"{loss_pl.item():.3f}",
                    pr=f"{loss_proto.item():.3f}",
                    di=f"{loss_dist.item():.3f}",
                )

            scheduler.step()

            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(
                    f"  Epoch {epoch+1}: loss={epoch_loss/n_batches:.4f} "
                    f"(pl={epoch_pl/n_batches:.4f}, proto={epoch_proto/n_batches:.4f}, "
                    f"dist={epoch_dist/n_batches:.4f})"
                )

        # After training, re-estimate and update prototypes with improved model
        model.eval()
        with torch.no_grad():
            # Process in batches to avoid OOM
            batch_size = self.config.target_batch_size
            all_feats_new = []
            for i in range(0, task_images.size(0), batch_size):
                batch_imgs = task_images[i:i+batch_size].to(self.device)
                _, batch_feats = model(batch_imgs)
                all_feats_new.append(batch_feats.cpu())
            feats_new = torch.cat(all_feats_new, dim=0).to(self.device)
        self.prototype_lib.update(
            feats_new,
            task_labels.to(self.device),
            task_reliability.to(self.device),
        )

        return model

    @torch.no_grad()
    def _evaluate(
        self,
        model: E3PModel,
        eval_dataset: Office31Dataset,
        seen_classes: List[int],
        task_id: int,
    ) -> float:
        """Evaluate on all seen classes so far."""
        model.eval()
        loader = DataLoader(
            eval_dataset,
            batch_size=self.config.target_batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

        correct = 0
        total = 0
        per_class_correct = {}
        per_class_total = {}

        for images, labels, _ in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Only evaluate on seen classes
            mask = torch.zeros(labels.size(0), dtype=torch.bool, device=self.device)
            for c in seen_classes:
                mask |= (labels == c)

            if mask.sum() == 0:
                continue

            images = images[mask]
            labels = labels[mask]

            logits, _ = model(images)
            # Mask logits for unseen classes (set to -inf)
            unseen_mask = torch.ones(self.config.num_classes, dtype=torch.bool, device=self.device)
            for c in seen_classes:
                unseen_mask[c] = False
            logits[:, unseen_mask] = float("-inf")

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            for c in seen_classes:
                c_mask = labels == c
                if c_mask.sum() > 0:
                    per_class_correct[c] = per_class_correct.get(c, 0) + (preds[c_mask] == labels[c_mask]).sum().item()
                    per_class_total[c] = per_class_total.get(c, 0) + c_mask.sum().item()

        overall_acc = 100.0 * correct / max(total, 1)

        # Per-class accuracy
        class_accs = []
        for c in sorted(per_class_total.keys()):
            ca = 100.0 * per_class_correct.get(c, 0) / max(per_class_total[c], 1)
            class_accs.append(ca)

        mean_class_acc = sum(class_accs) / max(len(class_accs), 1)

        logger.info(
            f"  Eval Task {task_id+1}: overall={overall_acc:.2f}%, "
            f"mean-class={mean_class_acc:.2f}%, "
            f"classes evaluated: {len(per_class_total)}/{len(seen_classes)}"
        )

        return mean_class_acc

    def _print_summary(self):
        logger.info("=" * 60)
        logger.info("E3P-SFCIDA Adaptation Summary")
        logger.info("=" * 60)
        for tid, metrics in sorted(self.task_accuracies.items()):
            logger.info(
                f"  After Task {tid+1}: "
                f"Acc on all seen = {metrics['all_seen_acc']:.2f}%"
            )

        # Forgetting measure
        if len(self.task_accuracies) > 1:
            accs = [v["all_seen_acc"] for v in self.task_accuracies.values()]
            peak = max(accs)
            final = accs[-1]
            forgetting = peak - final
            logger.info(f"  Peak accuracy: {peak:.2f}%, Final: {final:.2f}%, Forgetting: {forgetting:.2f}%")
        logger.info("=" * 60)
