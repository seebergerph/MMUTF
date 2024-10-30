import os
import json
import copy
import pickle
import random
import collections
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


ed_text_instance_fields = ["doc_id", "wnd_id", "tokens", "token_num", "trigger", "entities"]
EDTextInstance = collections.namedtuple(
    "EDTextInstance", field_names=ed_text_instance_fields, defaults=[None] * len(ed_text_instance_fields)
)

ed_text_batch_fields = ["doc_ids", "wnd_ids", "tokens", "token_nums", "triggers", "entities"]
EDTextBatch = collections.namedtuple(
    "EDTextBatch", field_names=ed_text_batch_fields, defaults=[None] * len(ed_text_batch_fields)
)

eae_text_instance_fields = ["doc_id", "wnd_id", "tokens", "token_num", "trigger", "entities", "arguments", "prompt_tokens", "prompt_slots"]
EAETextInstance = collections.namedtuple(
    "EAETextInstance", field_names=eae_text_instance_fields, defaults=[None] * len(eae_text_instance_fields)
)

eae_text_batch_fields = ["doc_ids", "wnd_ids", "tokens", "token_nums", "triggers", "entities", "arguments", "prompt_tokens", "prompt_slots"]
EAETextBatch = collections.namedtuple(
    "EAETextBatch", field_names=eae_text_batch_fields, defaults=[None] * len(eae_text_batch_fields)
)

ed_image_instance_fields = ["doc_id", "wnd_id", "image", "trigger", "objects"]
EDImageInstance = collections.namedtuple(
    "EDImageInstance", field_names=ed_image_instance_fields, defaults=[None] * len(ed_image_instance_fields)
)

ed_image_batch_fields = ["doc_ids", "wnd_ids", "images", "triggers", "objects"]
EDImageBatch = collections.namedtuple(
    "EDImageBatch", field_names=ed_image_batch_fields, defaults=[None] * len(ed_image_batch_fields)
)

eae_image_instance_fields = ["doc_id", "wnd_id", "image", "trigger", "arguments", "objects", "prompt_tokens", "prompt_slots"]
EAEImageInstance = collections.namedtuple(
    "EAEImageInstance", field_names=eae_image_instance_fields, defaults=[None] * len(eae_image_instance_fields)
)

eae_image_batch_fields = ["doc_ids", "wnd_ids", "images", "triggers", "arguments", "objects", "prompt_tokens", "prompt_slots"]
EAEImageBatch = collections.namedtuple(
    "EAEImageBatch", field_names=eae_image_batch_fields, defaults=[None] * len(eae_image_batch_fields)
)



class ACEDatasetED(Dataset):
    def __init__(self, path, filter_type_set=None):
        self.filter_type_set = filter_type_set
        self.dataset = self._load_data(path)
        self.entity_type_set = self._entity_type_set()
        self.event_type_set = self._event_type_set()


    def _entity_type_set(self):
        entity_type_set = set()
        for instance in self.dataset:
            for entity in instance.entities:
                entity_type_set.add(entity[2])
        return entity_type_set
    

    def _event_type_set(self):
        event_type_set = set()
        for instance in self.dataset:
            for event in instance.trigger:
                event_type_set.add(event[2])
        return event_type_set
    

    def _load_data(self, path):
        with open(path, "r", encoding="utf-8") as fp:
            lines = fp.readlines()
        samples = [json.loads(line) for line in lines]

        dataset = []
        for sample in samples:
            entities = [(
                entity["start"], entity["end"], entity["entity_type"], entity["text"]
            ) for entity in sample["entities"]]

            triggers = [(
                event["start"], event["end"], event["event_type"], event["text"]
            ) for event in sample["events"]]

            if self.filter_type_set:
                triggers = [item for item in triggers
                            if item[2] in self.filter_type_set["event_type_set"]]

            instance = EDTextInstance(
                doc_id=sample["doc_id"],
                wnd_id=sample["wnd_id"],
                tokens=sample["tokens"],
                token_num=len(sample["tokens"]),
                trigger=triggers,
                entities=entities
            )
            dataset.append(instance)
        print("{} -> loaded {} instances from {}".format(self.__class__.__name__, len(dataset), path))
        return dataset


    def collate_fn(self, batch):
        doc_ids = [inst.doc_id for inst in batch]
        wnd_ids = [inst.wnd_id for inst in batch]
        tokens = [inst.tokens for inst in batch]
        token_nums = [inst.token_num for inst in batch]
        triggers = [inst.trigger for inst in batch]
        entities = [inst.entities for inst in batch]

        return EDTextBatch(
            doc_ids=doc_ids, wnd_ids=wnd_ids, tokens=tokens, token_nums=token_nums, 
            triggers=triggers, entities=entities
        )


    def __len__(self): 
        return len(self.dataset)
    

    def __getitem__(self, index): 
        instance = self.dataset[index]
        return instance


class ACEDatasetEAE(Dataset):
    def __init__(self, path, filter_type_set=None, prompts_path=None):
        self.filter_type_set = filter_type_set
        self.prompts = self._load_prompts(prompts_path)
        self.dataset = self._load_data(path)
        self.entity_type_set = self._entity_type_set()
        self.event_type_set = self._event_type_set()
        self.argument_type_set = self._argument_type_set()


    def _entity_type_set(self):
        entity_type_set = set()
        for instance in self.dataset:
            for entity in instance.entities:
                entity_type_set.add(entity[2])
        return entity_type_set
    

    def _event_type_set(self):
        event_type_set = set()
        for instance in self.dataset:
            event_type_set.add(instance.trigger[2])
        return event_type_set
    

    def _argument_type_set(self):
        argument_type_set = set()
        for instance in self.dataset:
            for argument in instance.arguments:
                argument_type_set.add(argument[2])
        return argument_type_set
    

    def _load_prompts(self, path):
        if path:
            with open(path, "r") as fp:
                prompts = json.load(fp)
            return prompts
        return {}
    

    def _load_data(self, path):
        with open(path, "r", encoding="utf-8") as fp:
            lines = fp.readlines()
        samples = [json.loads(line) for line in lines]

        dataset = []
        for sample in samples:
            entities = [(
                entity["start"], entity["end"], entity["entity_type"], entity["text"]
            ) for entity in sample["entities"]]
                
            for event in sample["events"]:
                trigger = (event["start"], event["end"], event["event_type"], event["text"])
                if self.filter_type_set:
                    if not event["event_type"] in self.filter_type_set["event_type_set"]: 
                        continue

                arguments = [(
                    arg["start"], arg["end"], arg["role"], arg["text"]
                ) for arg in event["arguments"]]

                if self.filter_type_set:
                    arguments = [item for item in arguments
                                if item[2] in self.filter_type_set["argument_type_set"]]

                if len(self.prompts) > 0:
                    assert event["event_type"] in self.prompts
                prompt = self.prompts.get(event["event_type"], {"eae_prompt": None, "eae_slots": None})
                
                instance = EAETextInstance(
                    doc_id=sample["doc_id"],
                    wnd_id=sample["wnd_id"],
                    tokens=sample["tokens"],
                    token_num=len(sample["tokens"]),
                    trigger=trigger,
                    entities=entities,
                    arguments=arguments,
                    prompt_tokens=prompt["eae_prompt"],
                    prompt_slots=prompt["eae_slots"]
                )
                dataset.append(instance)
        print("{} -> loaded {} instances from {}".format(self.__class__.__name__, len(dataset), path))
        return dataset


    def collate_fn(self, batch):
        doc_ids = [inst.doc_id for inst in batch]
        wnd_ids = [inst.wnd_id for inst in batch]
        tokens = [inst.tokens for inst in batch]
        token_nums = [inst.token_num for inst in batch]
        triggers = [inst.trigger for inst in batch]
        entities = [inst.entities for inst in batch]
        arguments = [inst.arguments for inst in batch]
        prompt_tokens = [inst.prompt_tokens for inst in batch]
        prompt_slots = [inst.prompt_slots for inst in batch]

        return EAETextBatch(
            doc_ids=doc_ids, wnd_ids=wnd_ids, tokens=tokens, token_nums=token_nums,
            triggers=triggers, entities=entities, arguments=arguments,
            prompt_tokens=prompt_tokens, prompt_slots=prompt_slots
        )
    

    def __len__(self): 
        return len(self.dataset)
    

    def __getitem__(self, index):
        return self.dataset[index]


class SwigDatasetED(Dataset):
    def __init__(self, path, space_path, image_dir, mapping=None, max_samples_per_label=None):
        self.dataset = self._load_data(path, space_path, image_dir, mapping)
        self.event_type_set = self._event_type_set()

        if max_samples_per_label:
            dataset_per_et = {}
            dataset_sampled = []
            for sample in self.dataset:
                if not sample.trigger in dataset_per_et:
                    dataset_per_et[sample.trigger] = []
                dataset_per_et[sample.trigger].append(sample)
            
            for et, samples in dataset_per_et.items():
                random.shuffle(samples)
                idx = min(len(samples), max_samples_per_label)
                dataset_sampled.extend(samples[:idx])
            self.dataset = dataset_sampled


    def _event_type_set(self):
        event_type_set = set()
        for instance in self.dataset:
            event_type_set.add(instance.trigger)
        return event_type_set

    
    def _load_data(self, path, space_path, image_dir, mapping):
        with open(path, "r") as fp:
            samples = json.load(fp)

        with open(space_path, "r") as fp:
            self.swig_space = json.load(fp)

        dataset = []
        for image, sample in samples.items():
            image_id = image.split(".")[0]
            image = os.path.join(image_dir, image)

            event_type = sample["verb"]
            if mapping["ed"] is not None:
                event_type = mapping["ed"].get(event_type, "O")


            instance = EDImageInstance(
                doc_id=image_id,
                wnd_id=image_id,
                image=image,
                trigger=event_type
            )
            dataset.append(instance)
        print("{} -> loaded {} instances from {}".format(self.__class__.__name__, len(dataset), path))
        return dataset
    

    def collate_fn(self, batch):
        doc_ids = [inst.doc_id for inst in batch]
        wnd_ids = [inst.wnd_id for inst in batch]
        images = [inst.image for inst in batch]
        triggers = [inst.trigger for inst in batch]

        return EDImageBatch(
            doc_ids=doc_ids, wnd_ids=wnd_ids, images=images, triggers=triggers
        )


    def __len__(self): 
        return len(self.dataset)
    

    def __getitem__(self, index): 
        instance = self.dataset[index]
        if instance.image is not None:
            instance = instance._replace(image=np.asarray(Image.open(instance.image).convert("RGB")))
        return instance


class SwigDatasetEAE(Dataset):
    def __init__(self, path, space_path, image_dir, mapping=None, prompts_path=None):
        self.prompts = self._load_prompts(prompts_path)
        self.dataset = self._load_data(path, space_path, image_dir, mapping)
        self.event_type_set = self._event_type_set()
        self.argument_type_set = self._argument_type_set()


    def _event_type_set(self):
        event_type_set = set()
        for instance in self.dataset:
            event_type_set.add(instance.trigger)
        return event_type_set
    

    def _argument_type_set(self):
        argument_type_set = set()
        for instance in self.dataset:
            for argument in instance.arguments:
                argument_type_set.add(argument[-1])
        return argument_type_set
    

    def _load_prompts(self, path):
        if path:
            with open(path, "r") as fp:
                prompts = json.load(fp)
            return prompts
        return {}

    
    def _load_data(self, path, space_path, image_dir, mapping):
        with open(path, "r") as fp:
            samples = json.load(fp)

        with open(space_path, "r") as fp:
            self.swig_space = json.load(fp)

        dataset = []
        for image, sample in samples.items():
            image_id = image
            image = os.path.join(image_dir, image)

            event_type = sample["verb"]
            if mapping["ed"] is not None:
                event_type = mapping["ed"].get(event_type, "O")
            
            if event_type == "O": continue

            objects = []
            arguments = []
            for _role, bbox in sample["bb"].items():
                obj = copy.deepcopy(bbox)
                obj += [_role]
                if -1 in bbox: continue

                if mapping["eae"] is not None:
                    role = mapping["eae"][event_type].get(_role, "O")
                    
                bbox += [role]
                objects.append(obj)
                if role != "O":
                    arguments.append(bbox)

            if len(self.prompts) > 0:
                assert event_type in self.prompts
            prompt = self.prompts.get(event_type, {"eae_prompt": None, "eae_slots": None})

            instance = EAEImageInstance(
                doc_id=image_id,
                wnd_id=image_id,
                image=image,
                trigger=event_type,
                arguments=arguments,
                objects=objects,
                prompt_tokens=prompt["eae_prompt"],
                prompt_slots=prompt["eae_slots"]
            )
            dataset.append(instance)
        print("{} -> loaded {} instances from {}".format(self.__class__.__name__, len(dataset), path))
        return dataset
    

    def collate_fn(self, batch):
        doc_ids = [inst.doc_id for inst in batch]
        wnd_ids = [inst.wnd_id for inst in batch]
        images = [inst.image for inst in batch]
        triggers = [inst.trigger for inst in batch]
        arguments = [inst.arguments for inst in batch]
        objects = [inst.objects for inst in batch]
        prompt_tokens = [inst.prompt_tokens for inst in batch]
        prompt_slots = [inst.prompt_slots for inst in batch]

        return EAEImageBatch(
            doc_ids=doc_ids, wnd_ids=wnd_ids, images=images, triggers=triggers,
            arguments=arguments, objects=objects, prompt_tokens=prompt_tokens, 
            prompt_slots=prompt_slots
        )


    def __len__(self): 
        return len(self.dataset)
    

    def __getitem__(self, index): 
        instance = self.dataset[index]
        if instance.image is not None:
            instance = instance._replace(image=np.asarray(Image.open(instance.image).convert("RGB")))
        return instance
    

class M2E2TextDatasetED(Dataset):
    def __init__(self, path):
        self.dataset = self._load_data(path)
        self.entity_type_set = self._entity_type_set()
        self.event_type_set = self._event_type_set()


    def _entity_type_set(self):
        entity_type_set = set()
        for instance in self.dataset:
            for entity in instance.entities:
                entity_type_set.add(entity[2])
        return entity_type_set
    

    def _event_type_set(self):
        event_type_set = set()
        for instance in self.dataset:
            for event in instance.trigger:
                event_type_set.add(event[2])
        return event_type_set
    

    def _load_data(self, path):
        with open(path, "r", encoding="utf-8") as fp:
            lines = fp.readlines()
        samples = [json.loads(line) for line in lines]

        dataset = []
        for sample in samples:
            entities = [(
                entity["start"], entity["end"], entity["entity_type"], entity["text"]
            ) for entity in sample["entities"]]

            triggers = [(
                event["start"], event["end"], event["event_type"], event["text"]
            ) for event in sample["events"]]

            instance = EDTextInstance(
                doc_id=sample["doc_id"],
                wnd_id=sample["wnd_id"],
                tokens=sample["tokens"],
                token_num=len(sample["tokens"]),
                trigger=triggers,
                entities=entities
            )
            dataset.append(instance)
        print("{} -> loaded {} instances from {}".format(self.__class__.__name__, len(dataset), path))
        return dataset


    def collate_fn(self, batch):
        doc_ids = [inst.doc_id for inst in batch]
        wnd_ids = [inst.wnd_id for inst in batch]
        tokens = [inst.tokens for inst in batch]
        token_nums = [inst.token_num for inst in batch]
        triggers = [inst.trigger for inst in batch]
        entities = [inst.entities for inst in batch]

        return EDTextBatch(
            doc_ids=doc_ids, wnd_ids=wnd_ids, tokens=tokens, token_nums=token_nums, 
            triggers=triggers, entities=entities
        )


    def __len__(self): 
        return len(self.dataset)
    

    def __getitem__(self, index): 
        instance = self.dataset[index]
        return instance


class M2E2TextDatasetEAE(Dataset):
    def __init__(self, path, prompts_path=None, events=None):
        self.prompts = self._load_prompts(prompts_path)
        self.dataset = self._load_data(path, events)
        self.entity_type_set = self._entity_type_set()
        self.event_type_set = self._event_type_set()
        self.argument_type_set = self._argument_type_set()


    def _entity_type_set(self):
        entity_type_set = set()
        for instance in self.dataset:
            for entity in instance.entities:
                entity_type_set.add(entity[2])
        return entity_type_set
    

    def _event_type_set(self):
        event_type_set = set()
        for instance in self.dataset:
            event_type_set.add(instance.trigger[2])
        return event_type_set
    

    def _argument_type_set(self):
        argument_type_set = set()
        for instance in self.dataset:
            for argument in instance.arguments:
                argument_type_set.add(argument[2])
        return argument_type_set
    

    def _load_prompts(self, path):
        if path:
            with open(path, "r") as fp:
                prompts = json.load(fp)
            return prompts
        return {}
    

    def _load_data(self, path, events):
        with open(path, "r", encoding="utf-8") as fp:
            lines = fp.readlines()
        samples = [json.loads(line) for line in lines]

        dataset = []

        # Use predicted triggers
        if not events is None:
            event_map = collections.defaultdict(list)
            for event in events:
                map_id = (event["doc_id"], event["wnd_id"])
                event_map[map_id].append(event)

            for sample in samples:
                entities = [(
                    entity["start"], entity["end"], entity["entity_type"], entity["text"]
                ) for entity in sample["entities"]]

                map_id = (sample["doc_id"], sample["wnd_id"])
                for event in event_map[map_id]:
                    trigger = (event["start"], event["end"], event["event_type"], event["text"])
                    arguments  = []

                    if len(self.prompts) > 0:
                        assert event["event_type"] in self.prompts
                    prompt = self.prompts.get(event["event_type"], {"eae_prompt": None, "eae_slots": None})

                    instance = EAETextInstance(
                        doc_id=sample["doc_id"],
                        wnd_id=sample["wnd_id"],
                        tokens=sample["tokens"],
                        token_num=len(sample["tokens"]),
                        trigger=trigger,
                        entities=entities,
                        arguments=arguments,
                        prompt_tokens=prompt["eae_prompt"],
                        prompt_slots=prompt["eae_slots"]
                    )
                    dataset.append(instance)

        # Use gold triggers
        else:
            for sample in samples:
                entities = [(
                    entity["start"], entity["end"], entity["entity_type"], entity["text"]
                ) for entity in sample["entities"]]

                for event in sample["events"]:
                    trigger = (event["start"], event["end"], event["event_type"], event["text"])

                    arguments = [(
                        arg["start"], arg["end"], arg["role"], arg["text"]
                    ) for arg in event["arguments"]]

                    if len(self.prompts) > 0:
                        assert event["event_type"] in self.prompts
                    prompt = self.prompts.get(event["event_type"], {"eae_prompt": None, "eae_slots": None})

                    instance = EAETextInstance(
                        doc_id=sample["doc_id"],
                        wnd_id=sample["wnd_id"],
                        tokens=sample["tokens"],
                        token_num=len(sample["tokens"]),
                        trigger=trigger,
                        entities=entities,
                        arguments=arguments,
                        prompt_tokens=prompt["eae_prompt"],
                        prompt_slots=prompt["eae_slots"]
                    )
                    dataset.append(instance)

        print("{} -> loaded {} instances from {}".format(self.__class__.__name__, len(dataset), path))
        return dataset


    def collate_fn(self, batch):
        doc_ids = [inst.doc_id for inst in batch]
        wnd_ids = [inst.wnd_id for inst in batch]
        tokens = [inst.tokens for inst in batch]
        token_nums = [inst.token_num for inst in batch]
        triggers = [inst.trigger for inst in batch]
        entities = [inst.entities for inst in batch]
        arguments = [inst.arguments for inst in batch]
        prompt_tokens = [inst.prompt_tokens for inst in batch]
        prompt_slots = [inst.prompt_slots for inst in batch]

        return EAETextBatch(
            doc_ids=doc_ids, wnd_ids=wnd_ids, tokens=tokens, token_nums=token_nums,
            triggers=triggers, entities=entities, arguments=arguments,
            prompt_tokens=prompt_tokens, prompt_slots=prompt_slots
        )
    

    def __len__(self): 
        return len(self.dataset)
    

    def __getitem__(self, index):
        return self.dataset[index]


class M2E2ImageDatasetED(Dataset):
    def __init__(self, path, image_dir, prompts_path=None):
        self.prompts = self._load_prompts(prompts_path)
        self.dataset = self._load_data(path, image_dir)
        self.event_type_set = self._event_type_set()


    def _event_type_set(self):
        event_type_set = set()
        for instance in self.dataset:
            event_type_set.add(instance.trigger)
        return event_type_set
    

    def _load_prompts(self, path):
        if path:
            with open(path, "r") as fp:
                prompts = json.load(fp)
            return prompts
        return {}
    
    
    def _load_data(self, path, image_dir):
        with open(path, "r", encoding="utf-8") as fp:
            lines = fp.readlines()
        samples = [json.loads(line) for line in lines]

        dataset = []
        for sample in samples:
            image = sample["file"]
            image = os.path.join(image_dir, image)

            event_type = sample["event"].get("event_type", "O")

            instance = EDImageInstance(
                doc_id=sample["doc_id"],
                wnd_id=sample["img_id"],
                image=image,
                trigger=event_type
            )
            dataset.append(instance)
        print("{} -> loaded {} instances from {}".format(self.__class__.__name__, len(dataset), path))
        return dataset
    

    def collate_fn(self, batch):
        doc_ids = [inst.doc_id for inst in batch]
        wnd_ids = [inst.wnd_id for inst in batch]
        images = [inst.image for inst in batch]
        triggers = [inst.trigger for inst in batch]

        return EDImageBatch(
            doc_ids=doc_ids, wnd_ids=wnd_ids, images=images, triggers=triggers
        )


    def __len__(self): 
        return len(self.dataset)
    

    def __getitem__(self, index): 
        instance = self.dataset[index]
        if instance.image is not None:
            instance = instance._replace(image=np.asarray(Image.open(instance.image).convert("RGB")))
        return instance


class M2E2ImageDatasetEAE(Dataset):
    def __init__(self, path, image_dir, prompts_path=None, events=None, objects=None, objects_threshold=0.7):
        self.prompts = self._load_prompts(prompts_path)
        self.dataset = self._load_data(path, image_dir, events, objects, objects_threshold)
        self.event_type_set = self._event_type_set()
        self.argument_type_set = self._argument_type_set()


    def _event_type_set(self):
        event_type_set = set()
        for instance in self.dataset:
            event_type_set.add(instance.trigger)
        return event_type_set
    

    def _argument_type_set(self):
        argument_type_set = set()
        for instance in self.dataset:
            for argument in instance.arguments:
                argument_type_set.add(argument[-1])
        return argument_type_set
    

    def _load_prompts(self, path):
        if path:
            with open(path, "r") as fp:
                prompts = json.load(fp)
            return prompts
        return {}
    
    
    def _load_data(self, path, image_dir, events, objects, objects_threshold):
        with open(path, "r", encoding="utf-8") as fp:
            lines = fp.readlines()
        samples = [json.loads(line) for line in lines]

        dataset = []

        # Use predicted triggers
        if not events is None:
            event_map = dict()
            for event in events:
                map_id = (event["doc_id"], event["wnd_id"])
                event_map[map_id] = event

            for sample in samples:
                map_id = (sample["doc_id"], sample["img_id"])
                if map_id in event_map:
                    event = event_map[map_id]
                    image = sample["file"]
                    image = os.path.join(image_dir, image)
                    
                    event_type = event["trigger"]

                    if event_type == "O":
                        continue

                    candidates = []
                    for item in objects[sample["file"]]:
                        item = copy.deepcopy(item)
                        if item["conf"] >= objects_threshold:
                            bbox = item["bbox"]
                            role = item["class"]
                            bbox += [role]
                            candidates.append(bbox)

                    arguments = []

                    if len(self.prompts) > 0:
                        assert event_type in self.prompts
                    prompt = self.prompts.get(event_type, {"eae_prompt": None, "eae_slots": None})

                    instance = EAEImageInstance(
                        doc_id=sample["doc_id"],
                        wnd_id=sample["img_id"],
                        image=image,
                        trigger=event_type,
                        arguments=arguments,
                        objects=candidates,
                        prompt_tokens=prompt["eae_prompt"],
                        prompt_slots=prompt["eae_slots"]
                    )
                    dataset.append(instance)

        # Use gold triggers:
        else:
            for sample in samples:
                image = sample["file"]
                image = os.path.join(image_dir, image)

                event_type = sample["event"].get("event_type", "O")
                
                if event_type == "O":
                    continue

                candidates = []
                for item in objects[sample["file"]]:
                    item = copy.deepcopy(item)
                    if item["conf"] >= objects_threshold:
                        bbox = item["bbox"]
                        role = item["class"]
                        bbox += [role]
                        candidates.append(bbox)
 

                arguments = []
                for argument in sample["event"]["arguments"]:
                    argument = copy.deepcopy(argument)
                    bbox = argument["bbox"]
                    role = argument["role"]
                    bbox += [role]
                    arguments.append(bbox)

                if len(self.prompts) > 0:
                    assert event_type in self.prompts
                prompt = self.prompts.get(event_type, {"eae_prompt": None, "eae_slots": None})

                instance = EAEImageInstance(
                    doc_id=sample["doc_id"],
                    wnd_id=sample["img_id"],
                    image=image,
                    trigger=event_type,
                    arguments=arguments,
                    objects=candidates,
                    prompt_tokens=prompt["eae_prompt"],
                    prompt_slots=prompt["eae_slots"]
                )
                dataset.append(instance)

        print("{} -> loaded {} instances from {}".format(self.__class__.__name__, len(dataset), path))
        return dataset
    

    def collate_fn(self, batch):
        doc_ids = [inst.doc_id for inst in batch]
        wnd_ids = [inst.wnd_id for inst in batch]
        images = [inst.image for inst in batch]
        triggers = [inst.trigger for inst in batch]
        arguments = [inst.arguments for inst in batch]
        objects = [inst.objects for inst in batch]
        prompt_tokens = [inst.prompt_tokens for inst in batch]
        prompt_slots = [inst.prompt_slots for inst in batch]

        return EAEImageBatch(
            doc_ids=doc_ids, wnd_ids=wnd_ids, images=images, triggers=triggers,
            arguments=arguments, objects=objects, prompt_tokens=prompt_tokens, 
            prompt_slots=prompt_slots
        )


    def __len__(self): 
        return len(self.dataset)
    

    def __getitem__(self, index): 
        instance = self.dataset[index]
        if instance.image is not None:
            instance = instance._replace(image=np.asarray(Image.open(instance.image).convert("RGB")))
        return instance