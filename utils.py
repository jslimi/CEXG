
from tqdm import trange
from typing import Union, List
import numpy
from torch.nn.utils.rnn import pad_sequence
import torch
import random
import os
from collections import defaultdict

def model_info(tomita: int):
    """
    Returns the recurrent net information
    tomita: the tomita lang class
    """
    # reading input outputs samples
    x_sampling, y_sampling = read_data(f"data/datasets/tomita_lang/tomita{tomita}/test/tom{tomita}_sents.txt",

                                       f"data/datasets/tomita_lang/tomita{tomita}/test/tom{tomita}_labels.txt")
    # setting the model and tokenization function
    tokenizer = Tokenizer()
    sampling_tokens = pad_sequence([torch.tensor(tokenizer.tokenize(sent)) for sent in x_sampling], batch_first=True)
    sampling_labels = pad_sequence([torch.tensor(sent) for sent in y_sampling], batch_first=True)
    sampling_mask = (sampling_tokens != 0)
    trained_model = Tagger(tokenizer.n_tokens, 10, 100)
    filename = f"data/models/tomita_rnn/tom{tomita}.th"
    trained_model.load_state_dict(torch.load(filename, map_location=torch.device('cpu')))
    with torch.no_grad():
        sampling_results = trained_model(sampling_tokens, sampling_labels, sampling_mask)
    # storing the predictions, hidden states and vectors embeddings
    sampling_preds = sampling_results["predictions"]
    hidden_states = sampling_results["states"]
    embeddings = sampling_results["embeddings"]
    return x_sampling, y_sampling, sampling_tokens, sampling_labels, sampling_preds, hidden_states, embeddings


class Tokenizer:
    def __init__(self, bos=True, eos=False):
        self.next_idx = 0
        self.token_to_index = {}
        self.index_to_token = {}
        self.to_index("<pad>")
        self.to_index("<unk>")
        self.to_index("<bos>")
        self.to_index("<eos>")

        # Whether to add <bos>/<eos> tags during tokenization.
        self.eos = eos
        self.bos = bos

    def to_index(self, token, add=True):
        if token not in self.token_to_index:
            if add:
                self.token_to_index[token] = self.next_idx
                self.index_to_token[self.next_idx] = token
                self.next_idx += 1
            else:
                return self.token_to_index["<unk>"]
        return self.token_to_index[token]

    def tokenize(self, sentence, add=True):
        tokens = []
        if self.bos:
            tokens.append(self.to_index("<bos>"))
        tokens.extend(self.to_index(token, add=add) for token in sentence)
        if self.eos:
            tokens.append(self.to_index("<eos>"))
        return tokens

    @property
    def n_tokens(self):
        return self.next_idx

    @property
    def vocab(self):
        return [tok for tok in self.token_to_index if not tok.startswith("<")]


#Copied from the source code for the library AllenNLP.
#https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py#L628-L765

def sequence_cross_entropy_with_logits(
    logits: torch.FloatTensor,
    targets: torch.LongTensor,
    weights: Union[torch.FloatTensor, torch.BoolTensor],
    average: str = "batch",
    label_smoothing: float = None,
    gamma: float = None,
    alpha: Union[float, List[float], torch.FloatTensor] = None,
) -> torch.FloatTensor:
    """
    Computes the cross entropy loss of a sequence, weighted with respect to
    some user provided weights. Note that the weighting here is not the same as
    in the `torch.nn.CrossEntropyLoss()` criterion, which is weighting
    classes; here we are weighting the loss contribution from particular elements
    in the sequence. This allows loss computations for models which use padding.
    # Parameters
    logits : `torch.FloatTensor`, required.
        A `torch.FloatTensor` of size (batch_size, sequence_length, num_classes)
        which contains the unnormalized probability for each class.
    targets : `torch.LongTensor`, required.
        A `torch.LongTensor` of size (batch, sequence_length) which contains the
        index of the true class for each corresponding step.
    weights : `Union[torch.FloatTensor, torch.BoolTensor]`, required.
        A `torch.FloatTensor` of size (batch, sequence_length)
    average: `str`, optional (default = `"batch"`)
        If "batch", average the loss across the batches. If "token", average
        the loss across each item in the input. If `None`, return a vector
        of losses per batch element.
    label_smoothing : `float`, optional (default = `None`)
        Whether or not to apply label smoothing to the cross-entropy loss.
        For example, with a label smoothing value of 0.2, a 4 class classification
        target would look like `[0.05, 0.05, 0.85, 0.05]` if the 3rd class was
        the correct label.
    gamma : `float`, optional (default = `None`)
        Focal loss[*] focusing parameter `gamma` to reduces the relative loss for
        well-classified examples and put more focus on hard. The greater value
        `gamma` is, the more focus on hard examples.
    alpha : `Union[float, List[float]]`, optional (default = `None`)
        Focal loss[*] weighting factor `alpha` to balance between classes. Can be
        used independently with `gamma`. If a single `float` is provided, it
        is assumed binary case using `alpha` and `1 - alpha` for positive and
        negative respectively. If a list of `float` is provided, with the same
        length as the number of classes, the weights will match the classes.
        [*] T. Lin, P. Goyal, R. Girshick, K. He and P. Dollár, "Focal Loss for
        Dense Object Detection," 2017 IEEE International Conference on Computer
        Vision (ICCV), Venice, 2017, pp. 2999-3007.
    # Returns
    `torch.FloatTensor`
        A torch.FloatTensor representing the cross entropy loss.
        If `average=="batch"` or `average=="token"`, the returned loss is a scalar.
        If `average is None`, the returned loss is a vector of shape (batch_size,).
    """
    if average not in {None, "token", "batch"}:
        raise ValueError("Got average f{average}, expected one of None, 'token', or 'batch'")

    # make sure weights are float
    weights = weights.to(logits.dtype)
    # sum all dim except batch
    non_batch_dims = tuple(range(1, len(weights.shape)))
    # shape : (batch_size,)
    weights_batch_sum = weights.sum(dim=non_batch_dims)
    # shape : (batch * sequence_length, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # shape : (batch * sequence_length, num_classes)
    log_probs_flat = torch.nn.functional.log_softmax(logits_flat, dim=-1)
    # shape : (batch * max_len, 1)
    targets_flat = targets.view(-1, 1).long()
    # focal loss coefficient
    if gamma:
        # shape : (batch * sequence_length, num_classes)
        probs_flat = log_probs_flat.exp()
        # shape : (batch * sequence_length,)
        probs_flat = torch.gather(probs_flat, dim=1, index=targets_flat)
        # shape : (batch * sequence_length,)
        focal_factor = (1.0 - probs_flat) ** gamma
        # shape : (batch, sequence_length)
        focal_factor = focal_factor.view(*targets.size())
        weights = weights * focal_factor

    if alpha is not None:
        # shape : () / (num_classes,)
        if isinstance(alpha, (float, int)):

            # shape : (2,)
            alpha_factor = torch.tensor(
                [1.0 - float(alpha), float(alpha)], dtype=weights.dtype, device=weights.device
            )

        elif isinstance(alpha, (list, numpy.ndarray, torch.Tensor)):

            # shape : (c,)
            alpha_factor = torch.tensor(alpha, dtype=weights.dtype, device=weights.device)

            if not alpha_factor.size():
                # shape : (1,)
                alpha_factor = alpha_factor.view(1)
                # shape : (2,)
                alpha_factor = torch.cat([1 - alpha_factor, alpha_factor])
        else:
            raise TypeError(
                ("alpha must be float, list of float, or torch.FloatTensor, {} provided.").format(
                    type(alpha)
                )
            )
        # shape : (batch, max_len)
        alpha_factor = torch.gather(alpha_factor, dim=0, index=targets_flat.view(-1)).view(
            *targets.size()
        )
        weights = weights * alpha_factor

    if label_smoothing is not None and label_smoothing > 0.0:
        num_classes = logits.size(-1)
        smoothing_value = label_smoothing / num_classes
        # Fill all the correct indices with 1 - smoothing value.
        one_hot_targets = torch.zeros_like(log_probs_flat).scatter_(
            -1, targets_flat, 1.0 - label_smoothing
        )
        smoothed_targets = one_hot_targets + smoothing_value
        negative_log_likelihood_flat = -log_probs_flat * smoothed_targets
        negative_log_likelihood_flat = negative_log_likelihood_flat.sum(-1, keepdim=True)
    else:
        # Contribution to the negative log likelihood only comes from the exact indices
        # of the targets, as the target distributions are one-hot. Here we use torch.gather
        # to extract the indices of the num_classes dimension which contribute to the loss.
        # shape : (batch * sequence_length, 1)
        negative_log_likelihood_flat = -torch.gather(log_probs_flat, dim=1, index=targets_flat)
    # shape : (batch, sequence_length)
    negative_log_likelihood = negative_log_likelihood_flat.view(*targets.size())
    # shape : (batch, sequence_length)
    negative_log_likelihood = negative_log_likelihood * weights

    if average == "batch":
        # shape : (batch_size,)
        per_batch_loss = negative_log_likelihood.sum(non_batch_dims) / (
            weights_batch_sum + tiny_value_of_dtype(negative_log_likelihood.dtype)
        )
        num_non_empty_sequences = (weights_batch_sum > 0).sum() + tiny_value_of_dtype(
            negative_log_likelihood.dtype
        )
        return per_batch_loss.sum() / num_non_empty_sequences
    elif average == "token":
        return negative_log_likelihood.sum() / (
            weights_batch_sum.sum() + tiny_value_of_dtype(negative_log_likelihood.dtype)
        )
    else:
        # shape : (batch_size,)
        per_batch_loss = negative_log_likelihood.sum(non_batch_dims) / (
            weights_batch_sum + tiny_value_of_dtype(negative_log_likelihood.dtype)
        )
        return per_batch_loss

def tiny_value_of_dtype(dtype: torch.dtype):
    """
    Returns a moderately tiny value for a given PyTorch data type that is used to avoid numerical
    issues such as division by zero.
    This is different from `info_value_of_dtype(dtype).tiny` because it causes some NaN bugs.
    Only supports floating point dtypes.
    """
    if not dtype.is_floating_point:
        raise TypeError("Only supports floating point dtypes.")
    if dtype == torch.float or dtype == torch.double:
        return 1e-13
    elif dtype == torch.half:
        return 1e-4
    else:
        raise TypeError("Does not support dtype " + str(dtype))

class Tagger(torch.nn.Module):
    """
    Simple Recurrent Net (SRN)
    """
    def __init__(self, n_tokens, embed_dim, rnn_dim, n_labels=2):
        super().__init__()
        self.embed = torch.nn.Embedding(n_tokens, embed_dim)
        self.rnn = torch.nn.RNN(embed_dim, rnn_dim, batch_first=True)
        self.output = torch.nn.Linear(rnn_dim, n_labels)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, token_ids, labels, mask):
        embeddings = self.embed(token_ids)
        states, _ = self.rnn(embeddings)
        logits = self.output(states)
        loss = sequence_cross_entropy_with_logits(logits, labels, mask)
        predictions = logits.argmax(dim=-1)
        acc = ((predictions == labels) * mask).sum().float() / mask.sum()
        return {
            "states": states,
            "predictions": predictions,
            "embeddings": embeddings,
            "accuracy": acc,
            "loss": loss,
        }

def train_rnn(lang= "tomita 1.0"):
    """
    Train SRN on formal language recognition
    """
    use_gpu = torch.cuda.is_available()
    random.seed(2)
    torch.random.manual_seed(2)
    save_name = lang
    model_dir = os.path.join("models", save_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    x_train, y_train = lang1(100000, 10)
    x_test, y_test = lang1(10000, 10)
    # setting the model and tokenization function
    tokenizer = Tokenizer()
    test_tokens = pad_sequence([torch.tensor(tokenizer.tokenize(sent)) for sent in x_test], batch_first=True)
    test_labels = pad_sequence([torch.tensor(sent) for sent in y_test], batch_first=True)
    test_mask = (test_tokens != 0)
    train_tokens = pad_sequence([torch.tensor(tokenizer.tokenize(sent)) for sent in x_train], batch_first=True)
    train_labels = pad_sequence([torch.tensor(sent) for sent in y_train], batch_first=True)
    train_mask = (train_tokens != 0)


    model = Tagger(tokenizer.n_tokens, 10, 100)
    if use_gpu:
        print(f"Using CUDA device {0}")
        model.cuda(0)
    optim = torch.optim.AdamW(model.parameters())

    best_acc = 0.
    best_epoch = -1

    saved_epochs = []
    metrics = defaultdict(list)

    for epoch in range(100):
        print(f"Starting epoch {epoch}...")
        perm = torch.randperm(len(train_tokens))
        train_tokens = train_tokens[perm, :]
        train_labels = train_labels[perm, :]

        for batch_idx in trange(0, len(train_tokens) - 16, 16):
            optim.zero_grad()
            batch_tokens = train_tokens[batch_idx:batch_idx + 16]
            batch_labels = train_labels[batch_idx:batch_idx + 16]
            batch_mask = train_mask[batch_idx:batch_idx + 16]
            if use_gpu:
                batch_tokens = batch_tokens.cuda(0)
                batch_labels = batch_labels.cuda(0)
                batch_mask = batch_mask.cuda(0)
            output_dict = model(batch_tokens, batch_labels, batch_mask)
            loss = output_dict["loss"]
            loss.backward()
            optim.step()

        print("=== Train metrics ===")
        print("\tacc =", output_dict["accuracy"].item())
        print("\tloss =", output_dict["loss"].item())

        with torch.no_grad():
            if use_gpu:
                test_tokens = test_tokens.cuda(0)
                test_labels = test_labels.cuda(0)
                test_mask = test_mask.cuda(0)
            test_output_dict = model(test_tokens, test_labels, test_mask)
            metrics["loss"] = test_output_dict["loss"].item()
            metrics["accuracy"] = test_output_dict["accuracy"].item()
            print("=== test metrics ===")
            print("\tacc =", metrics["accuracy"])
            print("\tloss =", metrics["loss"])
            acc = metrics["accuracy"]

        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch

        if acc >= best_acc:
            best_path = os.path.join(model_dir, "best.th")
            torch.save(model.state_dict(), best_path)
            epoch_path = os.path.join(model_dir, f"epoch{epoch}.th")
            torch.save(model.state_dict(), epoch_path)
            saved_epochs.append(epoch)
            print(f"Saved checkpoint!")

        if epoch - best_epoch > 2:
            print("Stopped early!")
            break
