def compute_ED_scores(preds, golds):
    scores = {}
    scores["trigger_id"] = compute_ED_trigger_id_score(preds, golds)
    scores["trigger_cls"] = compute_ED_trigger_cls_score(preds, golds)
    return scores


def compute_EAE_scores(preds, golds):
    scores = {}
    scores["argument_id"] = compute_EAE_argument_id_score(preds, golds)
    scores["argument_cls"] = compute_EAE_argument_cls_score(preds, golds)
    return scores


def compute_ED_trigger_id_score(preds, golds):
    gold_tri_id, pred_tri_id = [], []
    for gold in golds:
        gold_tri_id_ = [(gold["doc_id"], gold["wnd_id"], t[0], t[1]) for t in gold["triggers"]]
        gold_tri_id.extend(gold_tri_id_)

    for pred in preds:
        pred_tri_id_ = [(pred["doc_id"], pred["wnd_id"], t[0], t[1]) for t in pred["triggers"]]
        pred_tri_id.extend(pred_tri_id_)
        
    gold_tri_id = set(gold_tri_id)
    pred_tri_id = set(pred_tri_id)
    tri_id_f1 = compute_f1(len(pred_tri_id), len(gold_tri_id), len(gold_tri_id & pred_tri_id))

    scores = {
        "pred_num": len(pred_tri_id), 
        "gold_num": len(gold_tri_id), 
        "match_num": len(gold_tri_id & pred_tri_id), 
        "precision": tri_id_f1[0], 
        "recall": tri_id_f1[1], 
        "f1": tri_id_f1[2], 
    }
    return scores


def compute_ED_trigger_cls_score(preds, golds):
    gold_tri_cls, pred_tri_cls = [], []
    for gold in golds:
        gold_tri_cls_ = [(gold["doc_id"], gold["wnd_id"], t[0], t[1], t[2]) for t in gold["triggers"]]
        gold_tri_cls.extend(gold_tri_cls_)

    for pred in preds:
        pred_tri_cls_ = [(pred["doc_id"], pred["wnd_id"], t[0], t[1], t[2]) for t in pred["triggers"]]
        pred_tri_cls.extend(pred_tri_cls_)

    gold_tri_cls = set(gold_tri_cls)
    pred_tri_cls = set(pred_tri_cls)
    tri_cls_f1 = compute_f1(len(pred_tri_cls), len(gold_tri_cls), len(gold_tri_cls & pred_tri_cls))

    scores = {
        "pred_num": len(pred_tri_cls), 
        "gold_num": len(gold_tri_cls), 
        "match_num": len(gold_tri_cls & pred_tri_cls), 
        "precision": tri_cls_f1[0], 
        "recall": tri_cls_f1[1], 
        "f1": tri_cls_f1[2], 
    }
    return scores


def compute_EAE_argument_id_score(preds, golds):
    gold_arg_id, pred_arg_id = [], []

    for gold in golds:
        gold_arg_id_ = [(gold["doc_id"], gold["wnd_id"], gold["trigger"][2], r[0], r[1]) for r in gold["arguments"]]
        gold_arg_id.extend(gold_arg_id_)
    for pred in preds:
        pred_arg_id_ = [(pred["doc_id"], pred["wnd_id"], pred["trigger"][2], r[0], r[1]) for r in pred["arguments"]]
        pred_arg_id.extend(pred_arg_id_)
        
    gold_arg_id = set(gold_arg_id)
    pred_arg_id = set(pred_arg_id)
    arg_id_f1 = compute_f1(len(pred_arg_id), len(gold_arg_id), len(gold_arg_id & pred_arg_id))

    scores = {
        "pred_num": len(pred_arg_id), 
        "gold_num": len(gold_arg_id), 
        "match_num": len(gold_arg_id & pred_arg_id), 
        "precision": arg_id_f1[0], 
        "recall": arg_id_f1[1], 
        "f1": arg_id_f1[2], 
    }
    return scores


def compute_EAE_argument_cls_score(preds, golds):
    gold_arg_cls, pred_arg_cls = [], []
    
    for gold in golds:
        gold_arg_cls_ = [(gold["doc_id"], gold["wnd_id"], gold["trigger"][2], r[0], r[1], r[2]) for r in gold["arguments"]]
        gold_arg_cls.extend(gold_arg_cls_)
    for pred in preds:
        pred_arg_cls_ = [(pred["doc_id"], pred["wnd_id"], pred["trigger"][2], r[0], r[1], r[2]) for r in pred["arguments"]]
        pred_arg_cls.extend(pred_arg_cls_)
        
    gold_arg_cls = set(gold_arg_cls)
    pred_arg_cls = set(pred_arg_cls)
    arg_cls_f1 = compute_f1(len(pred_arg_cls), len(gold_arg_cls), len(gold_arg_cls & pred_arg_cls))

    scores = {
        "pred_num": len(pred_arg_cls), 
        "gold_num": len(gold_arg_cls), 
        "match_num": len(gold_arg_cls & pred_arg_cls), 
        "precision": arg_cls_f1[0], 
        "recall": arg_cls_f1[1], 
        "f1": arg_cls_f1[2], 
    }
    return scores


def compute_f1(predicted, gold, matched):
    def safe_div(num, denom):
        return num / denom if denom > 0 else 0.0

    precision = safe_div(matched, predicted)
    recall = safe_div(matched, gold)
    f1 = safe_div(2 * precision * recall, precision + recall)

    return precision*100.0, recall*100.0, f1*100.0


def print_scores(scores):
    print("------------------------------------------------------------------------------")
    if "trigger_id" in scores:
        print('Tri-I            - P: {:6.2f} ({:5d}/{:5d}), R: {:6.2f} ({:5d}/{:5d}), F: {:6.2f}'.format(
            scores["trigger_id"]["precision"], scores["trigger_id"]["match_num"], 
            scores["trigger_id"]["pred_num"], scores["trigger_id"]["recall"], 
            scores["trigger_id"]["match_num"], scores["trigger_id"]["gold_num"], 
            scores["trigger_id"]["f1"]))
    if "trigger_cls" in scores:
        print('Tri-C            - P: {:6.2f} ({:5d}/{:5d}), R: {:6.2f} ({:5d}/{:5d}), F: {:6.2f}'.format(
            scores["trigger_cls"]["precision"], scores["trigger_cls"]["match_num"], 
            scores["trigger_cls"]["pred_num"], scores["trigger_cls"]["recall"], 
            scores["trigger_cls"]["match_num"], scores["trigger_cls"]["gold_num"], 
            scores["trigger_cls"]["f1"]))
    if "argument_id" in scores:
        print('Arg-I            - P: {:6.2f} ({:5d}/{:5d}), R: {:6.2f} ({:5d}/{:5d}), F: {:6.2f}'.format(
            scores["argument_id"]["precision"], scores["argument_id"]["match_num"], 
            scores["argument_id"]["pred_num"], scores["argument_id"]["recall"], 
            scores["argument_id"]["match_num"], scores["argument_id"]["gold_num"], 
            scores["argument_id"]["f1"]))
    if "argument_cls" in scores:
        print('Arg-C            - P: {:6.2f} ({:5d}/{:5d}), R: {:6.2f} ({:5d}/{:5d}), F: {:6.2f}'.format(
            scores["argument_cls"]["precision"], scores["argument_cls"]["match_num"], 
            scores["argument_cls"]["pred_num"], scores["argument_cls"]["recall"], 
            scores["argument_cls"]["match_num"], scores["argument_cls"]["gold_num"], 
            scores["argument_cls"]["f1"]))
    print("------------------------------------------------------------------------------")