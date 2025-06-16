import torch
import math

def fixed_length_decode_ctc(logp: torch.Tensor, blank_idx: int):
    # argmax per time step → (T, B)
    maxes = logp.argmax(dim=-1).cpu().tolist()  # list of length T, each sublist length B
    T = len(maxes)
    B = len(maxes[0])

    outs = []
    for b in range(B):
        collapsed = []
        prev = None
        for t in range(T):
            c = maxes[t][b]
            if c != prev and c != blank_idx:
                collapsed.append(c)
            prev = c
        # truncate to length 8
        if len(collapsed) < 8:
            collapsed += [blank_idx] * (8 - len(collapsed))
        else:
            collapsed = collapsed[:8]
        outs.append(collapsed)
    return outs  # [[idx0, …, idx7], …]

def beam_search_ctc(logp: torch.Tensor, beam_width: int, blank_idx: int, chars: list):
    T, B, C = logp.shape
    logp_cpu = logp.detach().cpu()

    all_beams = []
    for b in range(B):
        beam = [(0.0, [])]
        for t in range(T):
            new_beam = {}
            step_lp = logp_cpu[t, b]
            topk_vals, topk_idxs = torch.topk(step_lp, k=min(C, beam_width))
            for (prefix_score, prefix_seq) in beam:
                for k in range(len(topk_idxs)):
                    idx_c = int(topk_idxs[k].item())
                    new_score = prefix_score + float(topk_vals[k].item())
                    new_seq = prefix_seq + [idx_c]
                    key = tuple(new_seq)
                    if key not in new_beam or new_score > new_beam[key]:
                        new_beam[key] = new_score
            # keep top beam_width
            beam = sorted(new_beam.items(), key=lambda x: x[1], reverse=True)[:beam_width]
            beam = [(s, list(seq)) for (seq, s) in beam]
        beam_sorted = sorted(beam, key=lambda x: x[0], reverse=True)
        seqs_only = [seq for (s, seq) in beam_sorted]
        all_beams.append(seqs_only[:beam_width])
    return all_beams  # [ [ seq1, seq2, … ], … ]

def compute_char_plate_accuracy(preds: list, targets: list):
    #preds: List[B] of lists (length 8) of character indices

    assert len(preds) == len(targets)
    total_chars = 0
    correct_chars = 0
    total_plates = 0
    correct_plates = 0

    for p_seq, g_seq in zip(preds, targets):
        total_chars += len(g_seq)
        for i in range(len(g_seq)):
            if p_seq[i] == g_seq[i]:
                correct_chars += 1
        total_plates += 1
        if p_seq == g_seq:
            correct_plates += 1

    char_acc = correct_chars / total_chars if total_chars > 0 else 0.0
    plate_acc = correct_plates / total_plates if total_plates > 0 else 0.0
    return char_acc, plate_acc
