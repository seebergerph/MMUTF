def compute_ED_scores(preds, golds):
    scores = {}
    scores["trigger_cls"] = compute_ED_trigger_cls_score(preds, golds)
    return scores


def compute_EAE_scores(preds, golds):
    scores = {}
    scores["argument_cls"] = compute_EAE_argument_cls_score(preds, golds)
    return scores


def compute_ED_trigger_cls_score(preds, golds):
    gold_tri_cls, pred_tri_cls = [], []
    for gold in golds:
        gold_tri_cls_ = [(gold["doc_id"], gold["wnd_id"], gold["trigger"])]
        gold_tri_cls.extend(gold_tri_cls_)

    for pred in preds:
        pred_tri_cls_ = [(pred["doc_id"], pred["wnd_id"], pred["trigger"])]
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


def compute_EAE_argument_cls_score(preds, golds):
    pred_arg_cls, gold_arg_cls, match_arg_cls = 0, 0, 0

    pred_map = {}
    for pred in preds:
        map_id = (pred["doc_id"], pred["wnd_id"])
        pred_map[map_id] = pred

    preds_tmp = []
    for gold in golds:
        map_id = (gold["doc_id"], gold["wnd_id"])
        pred = pred_map.pop(map_id, {
            "doc_id": gold["doc_id"], "wnd_id": gold["wnd_id"], "trigger": "O", "arguments": []
        })
        preds_tmp.append(pred)

    for pred in pred_map.values():
        if pred["trigger"] == "O":
            continue
        for pred_arg in pred["arguments"]:
            if pred_arg[-1] != "O":
                pred_arg_cls +=1

    preds = preds_tmp

    for pred, gold in zip(preds, golds):
        assert pred["doc_id"] == gold["doc_id"] and pred["wnd_id"] == gold["wnd_id"]

        for gold_arg in gold["arguments"]:
            gold_arg_cls += 1

        for pred_arg in pred["arguments"]:
            pred_arg_cls += 1
            if pred["trigger"] == gold["trigger"]:
                for gold_arg in gold["arguments"]:
                    if (pred_arg[-1] == gold_arg[-1] and 
                        bbox_intersection_over_union(pred_arg[:-1], gold_arg[:-1])):
                        match_arg_cls += 1

    arg_cls_f1 = compute_f1(pred_arg_cls, gold_arg_cls, match_arg_cls)

    scores = {
        "pred_num": pred_arg_cls, 
        "gold_num": gold_arg_cls, 
        "match_num": match_arg_cls,
        "pred_match_num": match_arg_cls,
        "precision": arg_cls_f1[0], 
        "recall": arg_cls_f1[1], 
        "f1": arg_cls_f1[2], 
    }
    return scores


def bbox_intersection_over_union(bbox1, bbox2, iou_threshold=0.5):
    x1, y1 = max(bbox1[0], bbox2[0]), max(bbox1[1], bbox2[1])
    x2, y2 = min(bbox1[2], bbox2[2]), min(bbox1[3], bbox2[3])

    interArea = max(x2 - x1, 0) * max(y2 - y1, 0)
    bbox1Area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2Area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    unionArea = max(bbox1Area + bbox2Area - interArea, 1e-4)
    iou = interArea / float(unionArea)

    if iou >= iou_threshold: 
        return True
    return False


def compute_f1(predicted, gold, matched):
    def safe_div(num, denom):
        return num / denom if denom > 0 else 0.0
    
    precision = safe_div(matched, predicted)
    recall = safe_div(matched, gold)
    f1 = safe_div(2 * precision * recall, precision + recall)

    return precision * 100.0, recall * 100.0, f1 * 100.0


def compute_f1_eae(predicted, gold, matched, pred_matched):
    def safe_div(num, denom):
        return num / denom if denom > 0 else 0.0
    
    precision = safe_div(pred_matched, predicted)
    recall = safe_div(matched, gold)
    f1 = safe_div(2 * precision * recall, precision + recall)

    return precision * 100.0, recall * 100.0, f1 * 100.0


def print_scores(scores):
    print("------------------------------------------------------------------------------")
    if "trigger_cls" in scores:
        print('Tri-C            - P: {:6.2f} ({:5d}/{:5d}), R: {:6.2f} ({:5d}/{:5d}), F: {:6.2f}'.format(
            scores["trigger_cls"]["precision"], scores["trigger_cls"]["match_num"], 
            scores["trigger_cls"]["pred_num"], scores["trigger_cls"]["recall"], 
            scores["trigger_cls"]["match_num"], scores["trigger_cls"]["gold_num"], 
            scores["trigger_cls"]["f1"]))
    if "argument_cls" in scores:
        print('Arg-C            - P: {:6.2f} ({:5d}/{:5d}), R: {:6.2f} ({:5d}/{:5d}), F: {:6.2f}'.format(
            scores["argument_cls"]["precision"], scores["argument_cls"]["pred_match_num"], 
            scores["argument_cls"]["pred_num"], scores["argument_cls"]["recall"], 
            scores["argument_cls"]["match_num"], scores["argument_cls"]["gold_num"], 
            scores["argument_cls"]["f1"]))
    print("------------------------------------------------------------------------------")