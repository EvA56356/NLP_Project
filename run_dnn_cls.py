import os
import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from tools.common import seed_everything, plot_img_acc_loss, plot_img_auc
from models import TextCNN, TextBiLSTM
from processors.text_classify import convert_examples_to_features
from processors.text_classify import WordsProcessor as processors
from processors.text_classify import collate_fn
from tools.my_argparse import get_argparse
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


def train_evaluate_test(args, train_dataset, dev_dataset, test_dataset, model):
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset)
    collate_function = collate_fn
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                  collate_fn=collate_function)

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    args.warmup_steps = int(t_total * 0.1)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)

    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num Epochs = {args.num_train_epochs}")
    print(f"  Batch size = {args.per_gpu_train_batch_size}")
    print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    print(f"  Total optimization steps = {t_total}")

    global_step = 0
    steps_trained_in_current_epoch = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    seed_everything(args.seed)
    dev_best_acc = 0
    train_loss_list, train_acc_list = [], []
    dev_loss_list, dev_acc_list, dev_auc_list = [], [], []
    tmp_train_loss_list, tmp_train_acc_list = [], []
    for epoch in range(int(args.num_train_epochs)):
        progress = tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}", total=len(train_dataloader))
        for step, batch in enumerate(train_dataloader):
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0], "input_mask": batch[1],
                      "target": batch[2]}
            outputs = model(**inputs)
            loss, logits = outputs
            loss.backward()
            progress.set_postfix(loss=loss.item())
            tr_loss += loss.item()
            predict_all = torch.max(logits, 1)[1].cpu().numpy()
            train_acc = accuracy_score(batch[2].cpu(), predict_all)
            tmp_train_acc_list.append(train_acc)
            tmp_train_loss_list.append(loss.item())
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step() 
                model.zero_grad()
                global_step += 1
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    print(" ")
                    dev_acc, dev_loss, dev_auc = evaluate(args, model, dev_dataset)
                    train_loss = np.mean(np.array(tmp_train_loss_list))
                    train_acc = np.mean(np.array(tmp_train_acc_list))
                    train_loss_list.append(train_loss)
                    train_acc_list.append(train_acc)
                    dev_loss_list.append(dev_loss)
                    dev_acc_list.append(dev_acc)
                    dev_auc_list.append(dev_auc)
                    msg = ('Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  '
                            'Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Val AUC: {5:>6.2f}')
                    print(msg.format(global_step, loss.item(), train_acc, dev_loss, dev_acc, dev_auc))
                    tmp_train_loss_list, tmp_train_acc_list = [], []
                    if dev_acc > dev_best_acc:
                        dev_best_acc = dev_acc
                        save_path = os.path.join(args.output_dir, f"{args.model_type}.ckpt")
                        torch.save(model.state_dict(), save_path)
                        print(f"Saving model to {save_path}")
                progress.update(1)
        print("\n")
        if 'cuda' in str(args.device):
            torch.cuda.empty_cache()
    plot_img_acc_loss(train_loss_list, dev_loss_list, "Loss", args.model_type)
    plot_img_acc_loss(train_acc_list, dev_acc_list, "Accuracy", args.model_type)
    plot_img_auc(dev_auc_list, args.model_type)
    test(args, model, test_dataset)


def evaluate(args, model, eval_dataset, flag=False):
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset)
    collate_function = collate_fn
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                 collate_fn=collate_function)

    print("***** Running evaluation *****")
    print(f"  Num examples = {len(eval_dataset)}")
    print(f"  Batch size = {args.eval_batch_size}")
    eval_loss = 0.0
    nb_eval_steps = 0
    progress = tqdm(eval_dataloader, desc="Evaluating", total=len(eval_dataloader))
    if isinstance(model, nn.DataParallel):
        model = model.module
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    all_probs = []

    for step, batch in enumerate(eval_dataloader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0], "input_mask": batch[1],
                      "target": batch[2]}
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()

            batch_preds = torch.max(logits, 1)[1].cpu()
            labels_all = np.append(labels_all, batch[2].cpu())
            predict_all = np.append(predict_all, batch_preds)

            all_probs.append(probs)
        eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        progress.set_postfix(eval_loss=eval_loss / (nb_eval_steps + 1))
        progress.update(1)

    all_probs = np.concatenate(all_probs, axis=0)
    try:
        auc = roc_auc_score(labels_all, all_probs[:, 1], multi_class='ovr', average='macro')
    except ValueError:
        auc = 0.0
        print("AUC computation failed. It may be due to class imbalance or insufficient samples.")


    dev_acc = accuracy_score(labels_all, predict_all)
    dev_loss = eval_loss / nb_eval_steps
    if flag:
        report = classification_report(labels_all, predict_all, target_names=args.label_list, digits=4)
        confusion = confusion_matrix(labels_all, predict_all)
        return dev_acc, dev_loss, auc, report, confusion
    return dev_acc, dev_loss, auc


def test(args, model, dev_dataset):
    save_path = os.path.join(args.output_dir, f"{args.model_type}.ckpt")
    model.load_state_dict(torch.load(save_path))
    model.eval()
    test_acc, test_loss, test_auc, test_report, test_confusion = evaluate(args, model, dev_dataset, flag=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}, Test AUC: {2:>6.2f}'
    print(msg.format(test_loss, test_acc, test_auc))
    print("Precision, Recall and F1-Score")
    print(test_report)


def load_and_cache_examples(args, processor, data_type='train'):
    if data_type == 'train':
        max_length = args.train_max_seq_length
    else:
        max_length = args.eval_max_seq_length
    cached_features_file = os.path.join(args.data_dir, 'cached_-{}_{}'.format(
        data_type,
        args.model_type))
    if os.path.exists(cached_features_file):
        print(f"Loading features from cached file {cached_features_file}")
        features = torch.load(cached_features_file)
    else:
        print(f"Creating features from dataset file at {args.data_dir}")
        label_list = processor.get_labels()
        if data_type == 'train':
            examples = processor.get_train_examples(args.data_dir)
        elif data_type == 'dev':
            examples = processor.get_dev_examples(args.data_dir)
        else:
            examples = processor.get_test_examples(args.data_dir)
        features = convert_examples_to_features(
            examples=examples, label2id=args.label2id,
            max_seq_length=max_length, vocab_dict=processor.vocab_dict
        )
        print(f"Saving features into cached file {cached_features_file}")
        torch.save(features, cached_features_file)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_idx for f in features], dtype=torch.long)
    all_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_lens, all_label_ids)
    return dataset


def main():
    args = get_argparse().parse_args() 
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    args.output_dir = args.output_dir + '/{}'.format(args.model_type)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    time_ = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = 1
    args.device = device
    print(f"Device: {device}, n_gpu: {args.n_gpu}")
    seed_everything(args.seed)
    processor = processors(args.data_dir, word_type=args.word_type)
    args.label_list, args.label2id, args.id2label = processor.get_labels()
    num_labels = len(args.label2id)
    vocab_size = len(processor.vocab_dict)

    args.model_type = args.model_type.lower()

    train_dataset = load_and_cache_examples(args, processor, data_type='train')
    eval_dataset = load_and_cache_examples(args, processor, data_type='eval')
    test_dataset = load_and_cache_examples(args, processor, data_type='test')

    if args.model_type == 'cnn':
        weight = torch.Tensor([0.88, 0.12]).to(args.device)
        model = TextCNN(vocab_size=vocab_size, embedding_size=256, hidden_size=256, num_filters=128
                        , num_classes=num_labels, weight=weight)
    elif args.model_type == "lstm":
        model = TextBiLSTM(vocab_size=vocab_size, embedding_size=256,
                           hidden_size=256, num_classes=num_labels)
    else:
        print("model type error...")
        return

    model.to(args.device)
    if args.do_train:
        train_evaluate_test(args, train_dataset, eval_dataset, test_dataset, model)
        

if __name__ == "__main__":
    main()
