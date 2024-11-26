import argparse
import glob
import json
import os
import itertools
import random
from pathlib import Path

from utils.llms_inference import send_prompt
from utils.prompt_generation import read_prompts_from_file, generate_prompts_with_all_entities, \
    generate_prompts_for_companies_sanity_check, generate_prompts_for_entities_sanity_check, transform_categories

random.seed(42)

# removed '\n\n' to make sure this is only used between entries
CHOSEN_SEPARATOR_LIST = ['', '::: ', ':: ', ': ', ' \n\t', '\n    ', ' : ', ' - ', ' ', '\n ', '\n\t', ':', '::', '- ',
                         '\t']  # sep='' is used rarely, only for enumerations because there is already formatting there
CHOSEN_SPACE_LIST = ['', ' ', '\n', ' \n', ' -- ', '  ', '; \n', ' || ', ' <sep> ', ' -- ', ', ', ' \n ', ' , ', '\n ',
                     '. ', ' ,  ']  # space='' is used a lot
CHOSEN_SEPARATOR_TEXT_AND_OPTION_LIST = ['', ' ', '  ', '\t']

SEPARATORS_SPACES = CHOSEN_SEPARATOR_LIST + CHOSEN_SPACE_LIST + CHOSEN_SEPARATOR_TEXT_AND_OPTION_LIST

TEXT_DESCRIPTOR_FN_LIST = [
    (lambda x: x, "lambda x: x"),
    (lambda x: x.title(), "lambda x: x.title()"),
    (lambda x: x.upper(), "lambda x: x.upper()"),
    (lambda x: x.lower(), "lambda x: x.lower()")
]

N_PROMPT_MODIFICATIONS = 30


def create_formated_prompts(
        entities_list,
        group_prompts_path,
        individual_prompts_company_path,
        individual_prompts_entity_path,
        group_name,
        specific_property
):
    # collect the prompt formats & select N_PROMPT_MODIFICATIONS random formats
    prompts_formats = list(itertools.product(SEPARATORS_SPACES, TEXT_DESCRIPTOR_FN_LIST))
    selected_prompts_formats = random.sample(prompts_formats, N_PROMPT_MODIFICATIONS)

    group_prompts = read_prompts_from_file(group_prompts_path)
    individual_prompts_company = read_prompts_from_file(individual_prompts_company_path)
    individual_prompts_entity = read_prompts_from_file(individual_prompts_entity_path)

    # modify prompts according to the new formats
    modified_group_prompts, modified_individual_prompts_company, modified_individual_prompts_entity = [], [], []

    for format in selected_prompts_formats:
        for prompt in group_prompts:
            prompt_splitted = prompt.split(": ")
            modified_group_prompts.append(format[1][0](prompt_splitted[0]) + format[0] + prompt_splitted[1])
        for prompt in individual_prompts_company:
            prompt_splitted = prompt.split(" XXX")
            modified_individual_prompts_company.append(format[1][0](prompt_splitted[0]) + format[0] + "XXX")
        for prompt in individual_prompts_entity:
            prompt_splitted = prompt.split(" XXX")
            modified_individual_prompts_entity.append(format[1][0](prompt_splitted[0]) + format[0] + "XXX")

    filled_up_prompts = []

    # study 1
    filled_up_prompts += generate_prompts_for_companies_sanity_check(
        entities_list, modified_individual_prompts_company)
    filled_up_prompts += generate_prompts_for_entities_sanity_check(
        entities_list, modified_individual_prompts_entity, group_name, specific_property)
    # study 2
    filled_up_prompts += generate_prompts_with_all_entities(
        entities_list, modified_group_prompts, group_name, specific_property)
    # study 3
    filled_up_prompts += generate_modified_prompts_for_individual_entities(
        entities_list, group_name, selected_prompts_formats, specific_property)

    print(filled_up_prompts)

    return filled_up_prompts


def main(
        entities,
        group_name,
        save_path,
        hyperparams,
        group_prompts_path,
        path_to_individual_prompts_company,
        path_to_individual_prompts_entity,
        specific_property=None
):

    # create an output file
    out_path = os.path.join(save_path, hyperparams["model"], f"temp_{hyperparams['temp']}")
    Path(out_path).mkdir(parents=True, exist_ok=True)
    save_file = os.path.join(out_path, f"{group_name}.json")

    # check if there is already an output file for this set of entities
    if os.path.isfile(save_file):
        existing_output = json.load(open(save_file))
        processed_prompts = [out["prompt"].split(":")[0] for out in existing_output]
    else:
        existing_output, processed_prompts = [], None

    prompts = create_formated_prompts(
        entities,
        group_prompts_path,
        path_to_individual_prompts_company,
        path_to_individual_prompts_entity,
        group_name,
        specific_property
    )

    results = send_prompt(prompts, hyperparams, processed_prompts)

    # save
    existing_output = existing_output + results
    with open(save_file, "w+") as file:
        json.dump(existing_output, file)

    return results


def generate_modified_prompts_for_individual_entities(entities_list, group_name, prompt_formats, specific_property):
    group_name = transform_categories(group_name, singular=True)
    all_prompts = []
    for format in prompt_formats:
        for ent in entities_list:
            all_prompts.append(f"{format[1][0]('Provide the founding year for')}{format[0]}{ent}")
            all_prompts.append(f"{format[1][0]('Provide the founding year for the company')}{format[0]}{ent}")
            all_prompts.append(f"{format[1][0](f'Provide the {specific_property} for')}{format[0]}{ent}")
            all_prompts.append(f"{format[1][0](f'Provide the {specific_property} for {group_name}')}{format[0]}{ent}")
    return all_prompts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--entities", type=str)
    parser.add_argument("--model", type=str)

    parser.add_argument("--path_to_prompts", type=str, help="Path to the file with prompts", default=None)
    parser.add_argument("--path_to_indiv_prompts_company", type=str, help="", default=None)
    parser.add_argument("--path_to_indiv_prompts_entity", type=str, help="", default=None)
    parser.add_argument("--save_path", type=str, default="out")

    parser.add_argument("--temp", type=float, help="Temperature for the model", default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.model == "mistral": args.model = "mistral-7B-Instruct-v0.2"
    if args.model == "mixtral": args.model = "mixtral-8x7B-Instruct-v0.1"
    if args.model == "gemma": args.model = "gemma-1.1-7b-it"
    if args.model == "llama-3": args.model = "llama-3-70b"
    if args.model == "gpt-3.5": args.model = "gpt-3.5-turbo"
    if args.model == "gpt-4": args.model = "gpt-4o"

    API_HYPERPARAMS = {
        "model": args.model,
        "temp": args.temp,
        "seed": args.seed
    }

    # crawl all files with entities if a directory was specified
    if os.path.isdir(args.entities):
        files_with_entities = glob.glob(os.path.join(args.entities, "*.txt"))
    else:
        files_with_entities = [args.entities]

    for ent_f in files_with_entities:
        if ent_f == os.path.join(args.entities, "unique.txt"):
            continue

        print(f"Entities: {ent_f}")

        # read entities + specific property from the file
        with open(ent_f) as file:
            file_input = file.read().replace(", ", ",").replace("\n", ",").split(",")
            entities_list = file_input[:-1][:3]
            specific_property = file_input[-1]

            # specify the name of the output file = the name of the file with entities
            group_name_out = f"{ent_f.split(os.sep)[-1].split('.txt')[0]}"

            main(
                entities_list,
                group_name_out,
                args.save_path,
                API_HYPERPARAMS,
                args.path_to_prompts,
                args.path_to_indiv_prompts_company,
                args.path_to_indiv_prompts_entity,
                specific_property
            )
