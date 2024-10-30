NONE_label = "O"

M2E2_eventtypes = [
    NONE_label, "Conflict:Attack", "Conflict:Demonstrate", "Contact:Meet", "Contact:Phone-Write", 
    "Justice:Arrest-Jail", "Life:Die", "Movement:Transport", "Transaction:Transfer-Money"
]

M2E2_eventtype2idx = {tag: idx for idx, tag in enumerate(M2E2_eventtypes)}
M2E2_idx2eventtype = {val: key for key, val in M2E2_eventtype2idx.items()}

M2E2_roletypes = [
    NONE_label, "Money", "Target", "Victim", "Instrument", "Agent", "Artifact", "Entity", "Giver", 
    "Recipient", "Attacker", "Vehicle", "Place", "Person"
]

M2E2_roletype2idx = {tag: idx for idx, tag in enumerate(M2E2_roletypes)}
M2E2_idx2roletype = {val: key for key, val in M2E2_roletype2idx.items()}

M2E2_eventtype2roles= {
    "Conflict:Attack": ["Target", "Attacker", "Instrument"],
    "Conflict:Demonstrate": ["Entity", "Place"],
    "Contact:Meet": ["Entity"],
    "Contact:Phone-Write": ["Entity", "Instrument"],
    "Justice:Arrest-Jail": ["Person", "Agent", "Instrument"],
    "Life:Die": ["Victim", "Instrument"],
    "Movement:Transport": ["Agent", "Artifact", "Vehicle"],
    "Transaction:Transfer-Money": ["Money", "Recipient", "Giver"]
}