#!/usr/bin/env python3
"""Fine-tune SentenceTransformer retrieval model with MultipleNegativesRankingLoss."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Sequence, Set, Tuple, Any

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from sentence_transformers import InputExample, SentenceTransformer, SentencesDataset
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.util import batch_to_device


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "retrieval"

JsonDict = Dict[str, object]


@dataclass
class TrainConfig:
    corpus_path: Path
    triplets_path: Path
    pairs_path: Path
    output_dir: Path
    base_model: str
    epochs: int
    batch_size: int
    accumulation: int
    learning_rate: float
    warmup_ratio: float
    max_grad_norm: float
    seed: int
    dev_ratio: float
    max_hard_negatives: int
    eval_batch_size: int
    top_k: int
    # New options
    device: str
    fp16: bool
    grad_checkpointing: bool
    max_seq_length: int
    num_workers: int
    early_stopping_patience: int
    save_best_only: bool
    resume_from: Path | None
    pairs_only: bool


@dataclass
class DevQuery:
    query_id: str
    query_text: str
    positive_ids: Set[str]


def read_jsonl(path: Path) -> Iterator[JsonDict]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no} in {path}: {exc}") from exc


def compose_doc_text(record: JsonDict) -> str:
    title = str(record.get("title") or "").strip()
    body = str(record.get("text") or "").strip()
    if title and body:
        return f"{title}\n{body}"
    return title or body


def _get_tokenizer(model: SentenceTransformer):
    """Lấy tokenizer bên trong SentenceTransformer nếu có (dùng cho segment)."""
    try:
        first = model[0]
    except Exception:
        first = None
    if first is not None:
        tok = getattr(first, "tokenizer", None)
        if tok is not None:
            return tok
    return getattr(model, "tokenizer", None)


def segment_text_for_training(
    text: str,
    tokenizer: Any,
    max_tokens: int,
    overlap: int,
) -> List[str]:
    """Cắt Điều thành các đoạn biểu diễn phục vụ huấn luyện (multi-vector per Điều).

    - Nếu Điều ngắn (≤ max_tokens) → 1 đoạn như cũ.
    - Nếu dài → sliding window theo token với overlap.
    - Luôn trả về danh sách text các đoạn (không thay đổi label doc-level)."""
    text = (text or "").strip()
    if not text:
        return []

    if tokenizer is None:
        # Fallback ước lượng: ~4 ký tự / token
        approx_tokens = max_tokens * 4
        if len(text) <= approx_tokens:
            return [text]
        segments: List[str] = []
        stride = max(1, approx_tokens - overlap * 4)
        start = 0
        while start < len(text):
            end = min(len(text), start + approx_tokens)
            seg = text[start:end].strip()
            if seg:
                segments.append(seg)
            if end >= len(text):
                break
            start += stride
        return segments

    try:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
    except Exception:
        return [text]

    total_tokens = len(token_ids)
    if total_tokens <= max_tokens:
        return [text]

    segments: List[str] = []
    stride = max(1, max_tokens - overlap)
    start = 0
    while start < total_tokens:
        end = min(total_tokens, start + max_tokens)
        window_ids = token_ids[start:end]
        if not window_ids:
            break
        try:
            seg_text = tokenizer.decode(window_ids, skip_special_tokens=True).strip()
        except Exception:
            seg_text = text
        if seg_text:
            segments.append(seg_text)
        if end >= total_tokens:
            break
        start += stride

    return segments or [text]


def load_pairs(pairs_path: Path) -> Tuple[Dict[str, str], Dict[str, Set[str]]]:
    query_texts: Dict[str, str] = {}
    positives: Dict[str, Set[str]] = {}

    for record in tqdm(read_jsonl(pairs_path), desc="Read pairs", unit="row"):
        query_id = str(record.get("query_id") or "").strip()
        query_text = str(record.get("query_text") or "").strip()
        corpus_id = str(record.get("corpus_id") or "").strip()
        if not query_id or not query_text or not corpus_id:
            continue
        query_texts.setdefault(query_id, query_text)
        positives.setdefault(query_id, set()).add(corpus_id)

    if not query_texts:
        raise ValueError(f"Pairs file {pairs_path} is empty or invalid")

    return query_texts, positives


def choose_dev_queries(query_ids: Sequence[str], ratio: float, seed: int) -> Set[str]:
    if len(query_ids) < 2:
        raise ValueError("Need at least two queries to split train/dev")
    ratio = max(0.0, min(ratio, 0.5))
    rng = random.Random(seed)
    dev_size = max(1, int(round(len(query_ids) * ratio)))
    if dev_size >= len(query_ids):
        dev_size = len(query_ids) - 1
    return set(rng.sample(list(query_ids), dev_size))


def build_dev_queries(
    query_texts: Dict[str, str],
    positives: Dict[str, Set[str]],
    dev_ids: Set[str],
) -> List[DevQuery]:
    dev_queries: List[DevQuery] = []
    for query_id in dev_ids:
        query_text = query_texts.get(query_id)
        pos_ids = positives.get(query_id)
        if not query_text or not pos_ids:
            continue
        dev_queries.append(DevQuery(query_id=query_id, query_text=query_text, positive_ids=set(pos_ids)))
    if not dev_queries:
        raise ValueError("Dev split is empty after filtering")
    return dev_queries


def load_corpus_texts(corpus_path: Path) -> Tuple[List[str], List[str]]:
    doc_ids: List[str] = []
    doc_texts: List[str] = []
    for record in tqdm(read_jsonl(corpus_path), desc="Read corpus", unit="row"):
        doc_id = str(record.get("_id") or "").strip()
        text = compose_doc_text(record)
        if not doc_id or not text:
            continue
        doc_ids.append(doc_id)
        doc_texts.append(text)
    if not doc_ids:
        raise ValueError(f"Corpus file {corpus_path} has no valid documents")
    return doc_ids, doc_texts


def load_triplet_examples(
    triplets_path: Path,
    train_query_ids: Set[str],
    max_hard_negatives: int,
    model: SentenceTransformer,
    max_seq_length: int,
    segment_overlap: int = 64,
    pairs_only: bool = False,
) -> Tuple[List[InputExample], int]:
    """Sinh training examples với multi-segment per Điều.

    - Mỗi triplet → nhiều InputExample nếu positive/negative dài.
    - Query giữ nguyên; positive/negative được cắt theo token (max_seq_length, overlap).
    - Vẫn dùng MultipleNegativesRankingLoss (in-batch negatives)."""
    examples: List[InputExample] = []
    unique_queries: Set[str] = set()

    tokenizer = _get_tokenizer(model)
    overlap = max(0, min(segment_overlap, max_seq_length // 2))

    for record in tqdm(read_jsonl(triplets_path), desc="Read triplets", unit="row"):
        query_id = str(record.get("query_id") or "").strip()
        if query_id not in train_query_ids:
            continue
        query_text = str(record.get("query_text") or "").strip()
        positive_text = str(record.get("positive_text") or "").strip()
        negative_texts = record.get("negative_texts") or []
        if not query_text or not positive_text:
            continue

        # Segment positive Điều
        pos_segments = segment_text_for_training(positive_text, tokenizer, max_seq_length, overlap)
        if not pos_segments:
            continue

        # Segment negatives nếu dùng hard negatives
        neg_segments_all: List[str] = []
        if not pairs_only and isinstance(negative_texts, (list, tuple)) and max_hard_negatives > 0:
            for neg_text in negative_texts[:max_hard_negatives]:
                neg_str = str(neg_text or "").strip()
                if not neg_str:
                    continue
                segs = segment_text_for_training(neg_str, tokenizer, max_seq_length, overlap)
                neg_segments_all.extend(segs)

        # Multi-instance positives: mỗi segment của Điều dương → một InputExample
        for pos_seg in pos_segments:
            sentences: List[str] = [query_text, pos_seg]
            if neg_segments_all:
                sentences.extend(neg_segments_all)
            if len(sentences) < 2:
                continue
            examples.append(InputExample(texts=sentences))
            unique_queries.add(query_id)

    if not examples:
        raise ValueError("No training examples were constructed from triplets")
    return examples, len(unique_queries)


def create_dataloader(model: SentenceTransformer, examples: List[InputExample], batch_size: int, num_workers: int = 0) -> DataLoader:
    dataset = SentencesDataset(examples, model=model)
    # Windows an toàn: num_workers mặc định 0; cho phép tăng qua config
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, drop_last=False, num_workers=max(0, int(num_workers)))
    dataloader.collate_fn = model.smart_batching_collate
    return dataloader


def evaluate_model(
    model: SentenceTransformer,
    dev_queries: List[DevQuery],
    doc_ids: List[str],
    doc_texts: List[str],
    top_k: int,
    batch_size: int,
) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        doc_embeddings = model.encode(
            doc_texts,
            convert_to_tensor=True,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        query_texts = [item.query_text for item in dev_queries]
        query_embeddings = model.encode(
            query_texts,
            convert_to_tensor=True,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        scores = torch.matmul(query_embeddings, doc_embeddings.t())
        top_indices = torch.topk(scores, k=min(top_k, scores.shape[1]), dim=1).indices.cpu().numpy()

    hits = 0
    reciprocal_ranks: List[float] = []
    for row_idx, indices in enumerate(top_indices):
        positives = dev_queries[row_idx].positive_ids
        hit_rank = None
        for rank, doc_index in enumerate(indices, start=1):
            doc_id = doc_ids[int(doc_index)]
            if doc_id in positives:
                hits += 1
                hit_rank = rank
                break
        if hit_rank is not None:
            reciprocal_ranks.append(1.0 / hit_rank)
    recall = hits / len(dev_queries)
    mrr = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0
    return {"recall@10": recall, "mrr@10": mrr, "evaluated_queries": float(len(dev_queries))}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_loss_header(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["global_step", "epoch", "epoch_step", "loss"])


def train(config: TrainConfig) -> None:
    set_seed(config.seed)

    query_texts, positives = load_pairs(config.pairs_path)
    all_query_ids = sorted(query_texts.keys())
    dev_ids = choose_dev_queries(all_query_ids, config.dev_ratio, config.seed)
    train_query_ids = set(all_query_ids) - dev_ids
    dev_queries = build_dev_queries(query_texts, positives, dev_ids)

    # Resume nếu có
    if config.resume_from:
        model = SentenceTransformer(str(config.resume_from))
    else:
        model = SentenceTransformer(config.base_model)

    # Áp max_seq_length đồng bộ với inference pipeline và segment policy
    try:
        if hasattr(model, "max_seq_length"):
            model.max_seq_length = max(8, int(config.max_seq_length))
        try:
            first = model[0]
            if hasattr(first, "max_seq_length"):
                setattr(first, "max_seq_length", max(8, int(config.max_seq_length)))
        except Exception:
            pass
        # Đồng bộ với ENV để build_index / service dùng chung L
        os.environ["LEGALADVISOR_ENCODER_MAX_SEQ_LENGTH"] = str(int(config.max_seq_length))
    except Exception:
        pass

    # Chuẩn bị training examples sau khi đã có model/tokenizer
    examples, unique_train_queries = load_triplet_examples(
        config.triplets_path,
        train_query_ids,
        config.max_hard_negatives,
        model=model,
        max_seq_length=int(config.max_seq_length),
        pairs_only=config.pairs_only,
    )
    doc_ids, doc_texts = load_corpus_texts(config.corpus_path)

    # Thiết bị
    if config.device == "cpu":
        device = torch.device("cpu")
    elif config.device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:  # auto
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Bật gradient checkpointing cho backbone để tiết kiệm RAM nếu cần
    if config.grad_checkpointing:
        try:
            first = model[0]
            if hasattr(first, "auto_model") and hasattr(first.auto_model, "gradient_checkpointing_enable"):
                first.auto_model.gradient_checkpointing_enable()
        except Exception:
            pass

    train_dataloader = create_dataloader(model, examples, config.batch_size, num_workers=config.num_workers)
    loss_fn = MultipleNegativesRankingLoss(model)

    total_update_steps = math.ceil(len(train_dataloader) / config.accumulation) * config.epochs
    warmup_steps = int(total_update_steps * config.warmup_ratio)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_update_steps)

    loss_path = RESULTS_DIR / "train_loss.csv"
    write_loss_header(loss_path)
    loss_handle = loss_path.open("a", encoding="utf-8", newline="")
    loss_writer = csv.writer(loss_handle)

    global_step = 0
    metrics_history: List[Dict[str, float]] = []
    best_metric = -1.0
    epochs_no_improve = 0

    # AMP scaler
    scaler = torch.amp.GradScaler(enabled=(config.fp16 and device.type == "cuda"))

    def move_features_to_device(data):
        if isinstance(data, dict):
            return batch_to_device(data, device)
        if isinstance(data, list):
            return [batch_to_device(item, device) for item in data]
        return data

    try:
        for epoch in range(config.epochs):
            model.train()
            optimizer.zero_grad()
            accumulated = 0
            epoch_step = 0
            running_loss = 0.0

            progress = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config.epochs}")
            for batch in progress:
                if isinstance(batch, dict):
                    features = batch.get("sentence_features")
                    labels = batch.get("labels")
                else:
                    features, labels = batch
                features = move_features_to_device(features)
                if isinstance(labels, torch.Tensor):
                    labels = labels.to(device)
                if scaler.is_enabled():
                    with torch.cuda.amp.autocast():
                        loss = loss_fn(features, labels)
                        loss = loss / config.accumulation
                    scaler.scale(loss).backward()
                else:
                    loss = loss_fn(features, labels)
                    loss = loss / config.accumulation
                    loss.backward()

                accumulated += 1
                running_loss += loss.item()

                if accumulated % config.accumulation == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    if scaler.is_enabled():
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    actual_loss = running_loss
                    running_loss = 0.0
                    accumulated = 0

                    global_step += 1
                    epoch_step += 1
                    loss_writer.writerow([global_step, epoch + 1, epoch_step, f"{actual_loss:.6f}"])
                    loss_handle.flush()

            if accumulated != 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                actual_loss = running_loss
                running_loss = 0.0
                accumulated = 0

                global_step += 1
                epoch_step += 1
                loss_writer.writerow([global_step, epoch + 1, epoch_step, f"{actual_loss:.6f}"])
                loss_handle.flush()

            metrics = evaluate_model(
                model,
                dev_queries,
                doc_ids,
                doc_texts,
                config.top_k,
                config.eval_batch_size,
            )
            metrics["epoch"] = float(epoch + 1)
            metrics_history.append(metrics)
            recall_pct = metrics["recall@10"] * 100.0
            mrr_pct = metrics["mrr@10"] * 100.0
            print(f"Epoch {epoch+1}: recall@10={recall_pct:.2f}% mrr@10={mrr_pct:.2f}%")

            # Lưu checkpoint theo epoch
            epoch_dir = config.output_dir / f"checkpoint-epoch{epoch+1}"
            epoch_dir.mkdir(parents=True, exist_ok=True)
            model.save(str(epoch_dir))

            # Theo dõi best theo recall@10, có thể điều chỉnh theo mrr@10
            current = float(metrics.get("recall@10", 0.0))
            if current > best_metric + 1e-6:
                best_metric = current
                epochs_no_improve = 0
                best_dir = config.output_dir / "best"
                best_dir.mkdir(parents=True, exist_ok=True)
                model.save(str(best_dir))
            else:
                epochs_no_improve += 1

            if config.early_stopping_patience > 0 and epochs_no_improve >= config.early_stopping_patience:
                print(f"Early stopping after {epoch+1} epochs (no improvement {epochs_no_improve} >= {config.early_stopping_patience})")
                break

    finally:
        loss_handle.close()

    config.output_dir.mkdir(parents=True, exist_ok=True)
    # Nếu chỉ muốn lưu best, copy từ best; mặc định lưu full model cuối
    if config.save_best_only and (config.output_dir / "best").exists():
        # best đã lưu trong output_dir/best, không cần ghi đè
        pass
    else:
        model.save(str(config.output_dir))

    metrics_path = RESULTS_DIR / "dev_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics_history, handle, indent=2)

    model_info = {
        "base_model": config.base_model,
        "fine_tuned": True,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "train_triplets_count": len(examples),
        "train_query_count": unique_train_queries,
        "dev_query_count": len(dev_queries),
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "accumulation_steps": config.accumulation,
        "learning_rate": config.learning_rate,
        "warmup_ratio": config.warmup_ratio,
        "max_hard_negatives": config.max_hard_negatives,
        "final_metrics": metrics_history[-1] if metrics_history else {},
        "triplets_path": str(config.triplets_path),
        "pairs_path": str(config.pairs_path),
        "corpus_path": str(config.corpus_path),
    }
    with (config.output_dir / "model_info.json").open("w", encoding="utf-8") as handle:
        json.dump(model_info, handle, indent=2)


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train retrieval embedding model")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("data/processed/zalo-legal/corpus_cleaned.jsonl"),
    )
    parser.add_argument(
        "--triplets",
        type=Path,
        default=Path("data/processed/zalo-legal/triplets_train.jsonl"),
    )
    parser.add_argument(
        "--pairs",
        type=Path,
        default=Path("data/processed/zalo-legal/train_pairs_enriched.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/retrieval/zalo_v1"),
        help="Output directory for the fine-tuned model",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--max-hard-negatives", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=10)
    # New flags
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Thiết bị train: auto/cpu/cuda")
    parser.add_argument("--fp16", action="store_true", help="Dùng mixed precision (FP16) nếu có CUDA")
    parser.add_argument("--grad-checkpointing", action="store_true", help="Bật gradient checkpointing cho backbone")
    parser.add_argument("--max-seq-len", type=int, default=256, help="Giới hạn tokenizer/model max_seq_length")
    parser.add_argument("--num-workers", type=int, default=0, help="num_workers cho DataLoader (Windows nên để 0)")
    parser.add_argument("--early-stopping-patience", type=int, default=0, help="Dừng sớm nếu không cải thiện (số epoch)")
    parser.add_argument("--save-best-only", action="store_true", help="Chỉ giữ checkpoint tốt nhất (best)")
    parser.add_argument("--resume-from", type=Path, default=None, help="Tiếp tục huấn luyện từ checkpoint")
    parser.add_argument("--pairs-only", action="store_true", help="Bỏ qua negatives trong triplets, dùng in-batch negatives")

    args = parser.parse_args()

    config = TrainConfig(
        corpus_path=resolve_path(args.corpus),
        triplets_path=resolve_path(args.triplets),
        pairs_path=resolve_path(args.pairs),
        output_dir=resolve_path(args.output_dir),
        base_model=args.base_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        accumulation=max(1, args.accumulation),
        learning_rate=args.lr,
        warmup_ratio=max(0.0, min(args.warmup_ratio, 0.5)),
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        dev_ratio=max(0.01, min(args.dev_ratio, 0.3)),
        max_hard_negatives=max(0, args.max_hard_negatives),
        eval_batch_size=max(8, args.eval_batch_size),
        top_k=max(1, args.top_k),
        device=args.device,
        fp16=bool(args.fp16),
        grad_checkpointing=bool(args.grad_checkpointing),
        max_seq_length=max(8, int(args.max_seq_len)),
        num_workers=max(0, int(args.num_workers)),
        early_stopping_patience=max(0, int(args.early_stopping_patience)),
        save_best_only=bool(args.save_best_only),
        resume_from=resolve_path(args.resume_from) if args.resume_from else None,
        pairs_only=bool(args.pairs_only),
    )
    return config


def main() -> None:
    config = parse_args()
    train(config)


if __name__ == "__main__":
    main()
