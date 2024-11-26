import argparse
import random
from typing import List

import glob
import json
import os

from pathlib import Path

from utils.llms_inference import send_prompt
from utils.prompt_generation import generate_prompts_for_ambiguous_examples, \
    generate_prompts_for_companies_sanity_check, \
    generate_prompts_with_all_entities, generate_erroneous_prompts, read_prompts_from_file, \
    generate_prompts_for_entities_sanity_check, generate_prompts_for_individual_entities, generate_prompts_ask_ambiguous

random.seed(42)


def generate_prompts(
        entities_list: List,
        path_to_prompts: str = None,
        path_to_companies_sanity_check_prompts: str = None,
        path_to_entities_sanity_check_prompts: str = None,
        specific_property: str = None,
        processed_prompts: List = None,
        generate_erroneous: bool = False,
        generate_ambiguous: bool = False,
        generate_min: bool = False,
        group_name: str = None,
        ask_whether_ambiguous: bool = False
):
    all_prompts = []

    if ask_whether_ambiguous:
        all_prompts += generate_prompts_ask_ambiguous(entities_list, group_name)

    # read prompts from an external file
    if path_to_prompts:
        prompts = read_prompts_from_file(path_to_prompts)
        all_prompts += generate_prompts_with_all_entities(entities_list, prompts, group_name, specific_property)
        all_prompts += generate_prompts_for_individual_entities(entities_list, group_name, specific_property)

    # add prompts for sanity check of the companies
    if path_to_companies_sanity_check_prompts:
        prompts = read_prompts_from_file(path_to_companies_sanity_check_prompts)
        all_prompts += generate_prompts_for_companies_sanity_check(entities_list, prompts)

    # add prompts for sanity check of the entities
    if path_to_entities_sanity_check_prompts:
        prompts = read_prompts_from_file(path_to_entities_sanity_check_prompts)
        generated_prompts = generate_prompts_for_entities_sanity_check(entities_list, prompts, group_name, specific_property)
        if generated_prompts:
            all_prompts += generated_prompts

    if generate_erroneous:
        all_prompts += generate_erroneous_prompts(entities_list, all_prompts)

    if generate_ambiguous:
        all_prompts += generate_prompts_for_ambiguous_examples(entities_list)

    # filter out from the list the already processed prompts
    return list(
        filter(lambda x: x.split(":")[0] not in processed_prompts, all_prompts)) if processed_prompts else all_prompts


def main(
        entities,
        group_name,
        save_path,
        hyperparams,
        path_to_prompts,
        path_to_companies_sanity_check_prompts,
        path_to_entities_sanity_check_prompts,
        specific_property=None,
        ask_whether_ambiguous=False
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

    # generate prompts
    prompts = generate_prompts(
        entities,
        path_to_prompts=path_to_prompts,
        path_to_companies_sanity_check_prompts=path_to_companies_sanity_check_prompts,
        path_to_entities_sanity_check_prompts=path_to_entities_sanity_check_prompts,
        processed_prompts=processed_prompts,
        specific_property=specific_property,
        group_name=group_name,
        ask_whether_ambiguous=ask_whether_ambiguous
    )

    print(f"Amount of prompts: {len(prompts)}")

    results = send_prompt(prompts, hyperparams, processed_prompts)

    # save
    existing_output = existing_output + results
    with open(save_file, "w+") as file:
        json.dump(existing_output, file)

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--entities", type=str, help="Path to either file with the entities, separated by commas, "
                                                     "or to the folder with such files", default="")
    parser.add_argument(
        "--model", type=str, help="", default="gpt-3.5-turbo",
        choices=[
            "gpt",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-1106",
            "gpt-4",
            "gpt-4-turbo",
            "gpt-4o",
            "llama-3-70b",
            "llama-2-70b-chat-hf",
            "gemma-1.1-7b-it",
            "mistral-7B-Instruct-v0.2",
            "mixtral-8x7B-Instruct-v0.1",
            "gpt-3.5",
            "llama-3",
            "gemma",
            "mistral",
            "mixtral"
        ]
    )
    parser.add_argument("--url", type=str, help="URL for the model", default="http://localhost:8000/")
    parser.add_argument("--temp", type=float, help="Temperature for the model", default=1.0)
    parser.add_argument("--save_path", type=str, help="Path to the directory with the results", default="out")
    parser.add_argument("--seed", type=int, help="NB! This feature is in Beta. If specified, our system will make "
                                                 "a best effort to sample deterministically, such that repeated "
                                                 "requests with the same `seed` and parameters should return the same "
                                                 "result. Determinism is not guaranteed, and you should refer to the "
                                                 "`system_fingerprint` response parameter to monitor changes "
                                                 "in the backend (system_fingerprint: only supported in gpt-4-turbo).",
                        default=42)
    # parser.add_argument("--system_prompt", type=str, default="")
    # parser.add_argument("--group_name", type=str, help="Name of the group of entities", default="")
    parser.add_argument("--path_to_prompts", type=str, help="Path to the file with prompts", default=None)
    parser.add_argument("--path_to_companies_sanity_check_prompts", type=str,
                        help="Path to the file with prompts that will be used for sanity checking whether the model"
                             "knows about the companies.",
                        default=None)
    parser.add_argument("--path_to_entities_sanity_check_prompts", type=str,
                        help="Path to the file with prompts that will be used for sanity checking whether the model "
                             "knows about the entities",
                        default=None)
    parser.add_argument("--ask_whether_ambiguous", type=bool, default=None)

    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)

    if args.model == "mistral": args.model = "mistral-7B-Instruct-v0.2"
    if args.model == "mixtral": args.model = "mixtral-8x7B-Instruct-v0.1"
    if args.model == "gemma": args.model = "gemma-1.1-7b-it"
    if args.model == "llama-3": args.model = "llama-3-70b"
    if args.model == "gpt-3.5": args.model = "gpt-3.5-turbo"

    API_HYPERPARAMS = {
        "model": args.model,
        "temp": args.temp,
        "url": args.url,
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
            entities = file_input[:-1]
            specific_property = file_input[-1]

        # specify the name of the output file = the name of the file with entities
        group_name_out = f"{ent_f.split(os.sep)[-1].split('.txt')[0]}"

        if args.model == "gpt":
            for model in ["gpt-3.5-turbo", "gpt-4o"]:
                API_HYPERPARAMS["model"] = model
                main(
                    entities=entities,
                    group_name=group_name_out,
                    save_path=args.save_path,
                    path_to_prompts=args.path_to_prompts,
                    path_to_companies_sanity_check_prompts=args.path_to_companies_sanity_check_prompts,
                    path_to_entities_sanity_check_prompts=args.path_to_entities_sanity_check_prompts,
                    hyperparams=API_HYPERPARAMS,
                    specific_property=specific_property,
                    ask_whether_ambiguous=args.ask_whether_ambiguous
                )
        else:
            main(
                entities=entities,
                group_name=group_name_out,
                save_path=args.save_path,
                path_to_prompts=args.path_to_prompts,
                path_to_companies_sanity_check_prompts=args.path_to_companies_sanity_check_prompts,
                path_to_entities_sanity_check_prompts=args.path_to_entities_sanity_check_prompts,
                hyperparams=API_HYPERPARAMS,
                specific_property=specific_property,
                ask_whether_ambiguous=args.ask_whether_ambiguous
            )
