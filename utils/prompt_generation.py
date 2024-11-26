import random

random.seed(42)


def transform_categories(group_name, singular=False):
    if group_name == "animals":
        return "an animal" if singular else "animals"
    elif group_name == "fruits":
        return "a fruit" if singular else "fruits"
    elif group_name == "locations":
        # return "toponym" if singular else "toponyms"
        # return "a geographic location of" if singular else "geographic locations of"
        return "a geographic location" if singular else "geographic locations"
    elif group_name == "myths":
        return "a mythological character" if singular else "mythological characters"
    elif group_name == "people":
        return "a person" if singular else "people"
    elif group_name == "insp_words":
        return "an abstract concept" if singular else "concepts"
    elif group_name == "unique":
        return None
    else:
        raise ValueError(f"Unknown group: {group_name}")


def read_prompts_from_file(path_to_prompts):
    with open(path_to_prompts) as file:
        prompts = file.read().split("\n")
    return list(filter(lambda x: x != "" and not x.startswith("#"), prompts))


def generate_prompts_with_all_entities(entities, prompts, group_name, specific_property=None):
    group_name = transform_categories(group_name)
    if group_name:
        all_prompts = [prompt.replace("{CATEGORY}", group_name) for prompt in prompts]
    else:
        all_prompts = list(filter(lambda x: "{CATEGORY}" not in x, prompts))

    if specific_property:
        all_prompts = [prompt.replace("{SPECIFIC_PROPERTY}", specific_property) for prompt in all_prompts]
    else:
        all_prompts = list(filter(lambda x: "{SPECIFIC_PROPERTY}" not in x, all_prompts))

    return [
        prompt.replace("XXX", ", ".join(sorted(entities, key=lambda k: random.random()))) for prompt in all_prompts
    ]


def generate_erroneous_prompts(entities_list, all_prompts):
    n_entities = len(entities_list)
    rdm_1 = random.randint(1, 10)
    rdm_2 = random.randint(1, 10)

    erroneous_prompts_1 = [
        prompt
        .replace("entities", f"{n_entities + rdm_1} entities")
        .replace("companies", f"{n_entities + rdm_1} companies")
        for prompt in all_prompts
    ]
    erroneous_prompts_2 = [
        prompt
        .replace("entities", f"{n_entities - rdm_2} entities")
        .replace("companies", f"{n_entities - rdm_2} companies")
        for prompt in all_prompts
    ]

    return all_prompts + erroneous_prompts_1 + erroneous_prompts_2


def generate_prompts_ask_ambiguous(entities_list, group_name):
    group_name = transform_categories(group_name, singular=True)
    return [f"Can {ent} mean anything else but {group_name}? Answer only with Yes or No." for ent in entities_list]


def generate_prompts_for_ambiguous_examples(entities_list):
    return [f"Create an ambiguous example with the entity {ent}." for ent in entities_list]


def generate_prompts_for_companies_sanity_check(entities_list, prompts):
    all_prompts = []
    for ent in entities_list:
        all_prompts += [prompt.replace("XXX", ent) for prompt in prompts]
    return all_prompts


def generate_prompts_for_entities_sanity_check(entities_list, prompts, group_name, specific_property):
    group_name = transform_categories(group_name, singular=True)
    all_prompts = []
    for ent in entities_list:
        all_prompts += [
            prompt
            .replace("XXX", ent)
            .replace("{CATEGORY}", group_name)
            .replace("{SPECIFIC_PROPERTY}", specific_property)
            for prompt in prompts
        ]
    return list(set(all_prompts))


def generate_prompts_for_individual_entities(entities_list, group_name, specific_property):
    group_name = transform_categories(group_name, singular=True)
    all_prompts = []
    for ent in entities_list:
        all_prompts.append(f"Provide the founding year for {ent}")
        all_prompts.append(f"Provide the founding year for the company {ent}")
        all_prompts.append(f"Provide the {specific_property} for {ent}")
        all_prompts.append(f"Provide the {specific_property} for {group_name} {ent}")
    return all_prompts
