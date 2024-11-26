import os
import re
import glob
import argparse
import requests
import mwparserfromhell


def search_by_hyperlinks(wikicode):
    # print("(Based on hyperlinks)")
    parsed_wikicode = mwparserfromhell.parse(wikicode)
    final_amb_list_links = parsed_wikicode.filter_wikilinks(wikicode)
    return [str(link.title) for link in final_amb_list_links]


def search_by_keyword_match(wikicode, comp_name):
    # print("(Based on string matches)")
    parsed_wikicode = mwparserfromhell.parse(wikicode)
    whole_page_text = parsed_wikicode.strip_code().rsplit("See also")[0].split("\n\n")
    list_of_lines = [line.rsplit('\n') for line in whole_page_text]
    final_amb_list = []
    for line_set in list_of_lines:
        for line in line_set:
            if (
                    comp_name.lower() in line.lower() or comp_name.upper() in line.upper()
            ) and "may also refer to" not in line:
                final_amb_list.append(line)
    return final_amb_list


def search_by_lines_and_hyperlinks(wikicode):
    corr_line = re.split("\n==.*==\n", wikicode)
    final_amb_list = sum([line.split("\n") for line in corr_line], [])
    return list(filter(lambda x: x != "" and ("may also refer to" not in x) and ("[[" in x), final_amb_list))


def get_wiki_page(title):
    response = requests.get(
        'https://en.wikipedia.org/w/api.php',
        params={
            'action': 'query',
            'format': 'json',
            'titles': title,
            'prop': 'revisions',
            'rvprop': 'content'
        }).json()
    return next(iter(response['query']['pages'].values()))


def calculate_entity_ambiguity(comp_name, postfix="(disambiguation)"):
    page = get_wiki_page(f'{comp_name} {postfix}')

    if "revisions" not in page.keys():
        print(f"No 'revisions' for entity {comp_name}, abort")
        return None

    # if there is a redirection page, redirect :)
    if "redirect" in page["revisions"][0]['*'].lower():
        page = get_wiki_page(comp_name)

    # extract page content
    wikicode = page['revisions'][0]['*']

    # remove "See also" section
    wikicode = wikicode.rsplit("See also")[0]

    # select lines + check that there are hyperlinks iin them
    final_amb_list = search_by_lines_and_hyperlinks(wikicode)

    # select all hyperlinks
    # final_amb_list = search_by_hyperlinks(wikicode)

    # select lines with the match of the key word
    # final_amb_list = search_by_keyword_match(wikicode, comp_name)

    # final_amb_list_string = "\n".join(final_amb_list)
    # print(f'{final_amb_list_links_string}')

    return len(final_amb_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--entities", type=str,
                        help="Path to either file with the entities, separated by commas, or to the folder with such files",
                        default="")
    args = parser.parse_args()

    # crawl all files with entities if a directory was specified
    if os.path.isdir(args.entities):
        files_with_entities = glob.glob(os.path.join(args.entities, "*.txt"))
    else:
        files_with_entities = [args.entities]

    for ent_f in files_with_entities:
        print(f"Entities: {ent_f}")

        with open(ent_f) as file:
            file_input = file.read().replace(", ", ",").replace("\n", ",").split(",")
            entities = file_input[:-1]
            for ent in entities:
                ent_amb = calculate_entity_ambiguity(ent)
