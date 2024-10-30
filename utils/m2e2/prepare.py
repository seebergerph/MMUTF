import os
import json
import argparse
import pandas as pd


def print_text_stats(instances):
    docs = set()
    num_sentences = len(instances)
    num_entities = 0
    num_events = 0
    num_roles = 0
    entity_types = set()
    event_types = set()
    role_types = set()

    for instance in instances:
        docs.add(instance["doc_id"])
        for entity in instance["entities"]:
            num_entities +=1
            entity_types.add(entity["entity_type"])
        for event in instance["events"]:
            num_events += 1
            event_types.add(event["event_type"])
            for arg in event["arguments"]:
                num_roles += 1
                role_types.add(arg["role"])

    print("Docs: {}".format(len(docs)))
    print("Sentences: {}".format(num_sentences))
    print("Entity Types: {}".format(len(entity_types)))
    print("Entity Mentions: {}".format(num_entities))
    print("Event Types: {}".format(len(event_types)))
    print("Event Mentions: {}".format(num_events))
    print("Argument Roles: {}".format(len(role_types)))
    print("Argument Mentions: {}".format(num_roles))


def print_image_stats(instances):
    docs = set()
    num_images = len(instances)
    num_events = 0
    num_roles = 0
    event_types = set()
    role_types = set()

    for instance in instances:
        docs.add(instance["doc_id"])
        if "event_type" in instance["event"]:
            event = instance["event"]
            num_events += 1
            event_types.add(event["event_type"])
            for arg in event["arguments"]:
                num_roles += 1
                role_types.add(arg["role"])

    print("Docs: {}".format(len(docs)))
    print("Images: {}".format(num_images))
    print("Event Types: {}".format(len(event_types)))
    print("Event Mentions: {}".format(num_events))
    print("Argument Roles: {}".format(len(role_types)))
    print("Argument Mentions: {}".format(num_roles))


def prepare_text_samples(samples):
        instances = []    
        count_w_event = 0
        count_wo_event = 0

        # Iterate documents
        for _, sent_item in enumerate(samples):
            sid = sent_item["sentence_id"]
            did = "_".join(sid.split("_")[:-1])
            tokens = sent_item["words"]

            # Iterate entities per sentence
            entities = []
            for i, entity_item in enumerate(sent_item.get("golden-entity-mentions", [])):
                entity = {}
                entity["entity_id"] = f"{sid}-entity-{i+1}"
                entity["entity_type"] = entity_item["entity-type"]
                entity["text"] = entity_item["text"]
                entity["start"] = entity_item["start"]
                entity["end"] = entity_item["end"]
                entities.append(entity)
                
            # Iterate events per sentence
            events = []
            for i, event_item in enumerate(sent_item.get("golden-event-mentions", [])):
                event = {}
                event["event_id"] = f"{sid}-event-{i+1}"
                event["event_type"] = event_item["event_type"]
                event["text"] = event_item["trigger"]["text"]
                event["start"] = event_item["trigger"]["start"]
                event["end"] = event_item["trigger"]["end"]

                event_arguments = []
                # Iterate arguments per event
                for _, arg_item in enumerate(event_item.get("arguments", [])):
                    argument = dict()
                    argument["role"] = arg_item["role"]
                    argument["text"] = arg_item["text"] 
                    argument["start"] = arg_item["start"]
                    argument["end"] = arg_item["end"]
                    event_arguments.append(argument)

                event["arguments"] = event_arguments
                events.append(event)

            images = sent_item.get("image", [])

            instances.append(
                {
                    "doc_id": did,
                    "wnd_id": sid,  
                    "tokens": tokens, 
                    "entities": entities,
                    "events": events,
                    "images": images
                }
            )

            # Count samples
            if len(events) == 0:
                count_wo_event += 1
            else:
                count_w_event += 1
            
        print("{} sentences (w/ event) collected.".format(count_w_event))
        print("{} sentences (w/o event) collected.".format(count_wo_event))
        print()
        return instances


def prepare_image_samples(samples):
    instances = []    

    # Iterate documents
    for image, img_item in samples.items():
        did = "_".join(image.split("_")[:-1])
        iid = image

        event = {}
        event["event_id"] = f"{iid}-event"
        event["event_type"] = img_item["event_type"]

        event_arguments = []
        # Iterate arguments per event
        for role, arg_items in img_item.get("role", {}).items():
            for arg_item in arg_items:
                argument = dict()
                argument["role"] = role
                argument["bbox"] = arg_item[1:]
                event_arguments.append(argument)

        event["arguments"] = event_arguments

        instances.append(
            {
                "doc_id": did,
                "img_id": iid,
                "file": image + ".jpg",
                "event": event
            }
        )

    print("{} images (w/ event) collected.".format(len(instances)))
    print()
    return instances


def prepare_image_wo_event_samples(image_dir, instances_w_event):
    instances_wo_event = []

    image_ids = set([sample["img_id"] for sample in instances_w_event])

    # Iterate documents
    for image in os.listdir(image_dir):
        image = ".".join(image.split(".")[:-1])
        did = "_".join(image.split("_")[:-1])
        iid = image

        if not iid in image_ids:
            instances_wo_event.append(
                {
                    "doc_id": did,
                    "img_id": iid,
                    "file": image + ".jpg",
                    "event": {}
                }
            )

    print("{} images (w/o event) collected.".format(len(instances_wo_event)))
    print()
    return instances_wo_event


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    text_out_file = os.path.join(args.output_dir, args.text_out_file)
    image_out_file = os.path.join(args.output_dir, args.image_out_file)

    with open(args.text_only_file, "r") as fs: text_only_docs = json.load(fs)
    with open(args.text_multi_file, "r") as fs: text_multi_docs = json.load(fs)
    with open(args.image_only_file, "r") as fs: image_only_docs = json.load(fs)
    with open(args.image_multi_file, "r") as fs: image_multi_docs = json.load(fs)

    print("##### Text only")
    text_only_docs = prepare_text_samples(text_only_docs)
    print_text_stats(text_only_docs); print()

    print("##### Text multimedia")
    text_multi_docs = prepare_text_samples(text_multi_docs)
    print_text_stats(text_multi_docs); print()

    text_all_docs = pd.DataFrame(text_only_docs + text_multi_docs)
    text_all_docs.to_json(text_out_file, orient="records", lines=True)

    print("##### Image only")
    image_only_docs = prepare_image_samples(image_only_docs)
    print_image_stats(image_only_docs); print()

    print("##### Image multimedia")
    image_multi_docs = prepare_image_samples(image_multi_docs)
    print_image_stats(image_multi_docs); print()

    print("##### Image w/o event")
    image_wo_event_docs = prepare_image_wo_event_samples(args.image_dir, image_only_docs + image_multi_docs)

    image_all_docs = pd.DataFrame(image_only_docs + image_multi_docs + image_wo_event_docs)
    image_all_docs.to_json(image_out_file, orient="records", lines=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_only_file", type=str, required=True)
    parser.add_argument("--text_multi_file", type=str, required=True)
    parser.add_argument("--image_only_file", type=str, required=True)
    parser.add_argument("--image_multi_file", type=str, required=True)
    parser.add_argument("--text_out_file", type=str, required=True)
    parser.add_argument("--image_out_file", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    main(args)