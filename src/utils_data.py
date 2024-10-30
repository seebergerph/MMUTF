import copy
import collections


# Following previous work UniCL & CAMEL
MY_SWIG2M2E2_MAPPINGS = {
    "floating": "Movement:Transport", 
    "leading": "Contact:Meet", 
    "cheering": "Conflict:Demonstrate",
    "restraining": "Justice:Arrest-Jail", 
    "bulldozing": "Conflict:Attack", 
    "mourning": "Life:Die",
    "tugging": "Conflict:Attack", 
    "signing": "Contact:Meet", 
    "colliding": "Conflict:Attack",
    "weighing": "Movement:Transport", 
    "sleeping": "Life:Die", 
    "falling": "Life:Die",
    "confronting": "Contact:Meet", 
    "gambling": "Transaction:Transfer-Money",
    "pricking": "Transaction:Transfer-Money"
}


def prepare_swig2ace_mapping(filepath):
    swig2ace_ed = dict()
    swig2ace_eae = collections.defaultdict(dict)

    def add_my_mappings(swig2ace_ed):
        swig2ace_ed = copy.deepcopy(swig2ace_ed)

        print("SWiG2ACE Mapping: Refine pre-defined verbs!")
        drop = ['destroying', 'saluting', 'subduing', 'gathering', 
                'ejecting', 'marching', 'aiming', 'confronting',
                'bulldozing']
        for k in drop:
            if k in swig2ace_ed:
                swig2ace_ed.pop(k)

        for key, val in MY_SWIG2M2E2_MAPPINGS.items():
            swig2ace_ed[key] = val
        return swig2ace_ed

    with open(filepath, "r") as file:
        for line in file:
            fields = line.strip().split()
            swig_verb = fields[0]
            swig_arg = fields[1]
            ace_event = fields[2]
            ace_arg = fields[3]

            ace_event = ace_event.replace("||", ":").replace("|", "-")

            if not swig_verb in swig2ace_ed: 
                swig2ace_ed[swig_verb] = ace_event 

            if not swig_arg in swig2ace_eae: 
                swig2ace_eae[ace_event][swig_arg] = ace_arg

    swig2ace_ed = add_my_mappings(swig2ace_ed)
    return {"ed": swig2ace_ed, "eae": swig2ace_eae}