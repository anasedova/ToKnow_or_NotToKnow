import os
import requests
import pickle

from matplotlib.gridspec import GridSpec

from utils.collect_amb_info import calculate_entity_ambiguity

import numpy as np
import matplotlib.pyplot as plt

from numerize import numerize

with open("utils/wiki_access_token.txt") as f:
    ACCESS_TOKEN = f.read()

WIKI_URL = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user"

START_DATE = "20150701"
END_DATE = "20240520"


def get_wiki_page_views(page_title):
    url = os.path.join(WIKI_URL, page_title, "monthly", f"{START_DATE}00", f"{END_DATE}00")
    headers = {'Authorization': f'Bearer {ACCESS_TOKEN}', 'User-Agent': 'amb'}
    response = requests.get(url, headers=headers)
    data = response.json()["items"]
    return sum([d["views"] for d in data])


class AmbiguousEntity:

    def __init__(self, name, company_page_title, other_page_title, ambiguity=None):
        self.name = name
        self.company_page_title = company_page_title
        self.other_page_title = other_page_title

        self.company_popularity = get_wiki_page_views(self.company_page_title) if self.company_page_title else 0
        self.other_popularity = get_wiki_page_views(self.other_page_title) if self.other_page_title else 0

        self.ambiguity = ambiguity if type(ambiguity) == int else calculate_entity_ambiguity(self.name)

    def __repr__(self):
        return self.name


def process_entities():
    entities = {
        "fruits": [
            AmbiguousEntity("Apple", "Apple_Inc.", "Apple"),  # , 48),
            AmbiguousEntity("Fig", "Fig_(company)", "Fig"),  # , 15),
            AmbiguousEntity("Mango", "Mango_(retailer)", "Mango"),  # , 43),
            AmbiguousEntity("Kiwi", "Kiwi.com", "Kiwifruit"),  # , 36),
            # AmbiguousEntity("Peach", "Peach_Aviation", "Peach"),  # , 20),
            AmbiguousEntity("Papaya", None, "Papaya"),  # , 12),
            AmbiguousEntity("Orange", "Orange_S.A.", "Orange_(fruit)"),  # , 103)
        ],
        "animals": [
            AmbiguousEntity("Jaguar", "Jaguar_Cars", "Jaguar"),  # , 53),
            AmbiguousEntity("Puma", "Puma_(brand)", "Cougar"),  # , 45),
            AmbiguousEntity("Penguin", "Penguin_Random_House", "Penguin"),  # , 55),
            AmbiguousEntity("Greyhound", "Greyhound_Lines", "Greyhound"),  # , 35),
            AmbiguousEntity("Dove", "Dove_(brand)", "Columbidae"),  # , 50),
            AmbiguousEntity("Fox", "Fox_Corporation", "Fox"),  # , 89),
            AmbiguousEntity("Lynx", "Axe_(brand)", "Lynx"),  # , 78),
            # AmbiguousEntity("Jellyfish", "Jellyfish.com", "Jellyfish"),  # , 12),
            # AmbiguousEntity("Sparrow", "Sparrow_Health_System", "Sparrow"),  # , 62),
            # AmbiguousEntity("Shell", "Shell_corporation", "Seashell"),  # , 56),
        ],
        "myths": [
            AmbiguousEntity("Amazon", "Amazon_(company)", "Amazons"),  # , 63),
            AmbiguousEntity("Nike", "Nike,_Inc.", "Nike_(mythology)"),  # , 34),
            # AmbiguousEntity("Oracle", "Oracle_Corporation", "Oracle"),  # , 58),
            # AmbiguousEntity("Pandora", "Pandora_(jewelry)", "Pandora"),  # , 101),
            AmbiguousEntity("Midas", "Midas_(automotive service)", "Midas"),  # , 38),
            AmbiguousEntity("Mars", "Mars_Inc.", "Mars"),  # , 134),
            AmbiguousEntity("Hermes", "Hermès", "Hermes"),  # , 57),
            AmbiguousEntity("Hyperion", "Hyperion_Solutions", "Hyperion_(Titan)"),  # , 62),
            AmbiguousEntity("Vulcan", "Vulcan_Inc.", "Vulcan_(mythology)"),  # , 79),
            AmbiguousEntity("Pegasus", "Pegasus_Airlines", "Pegasus"),  # , 86),
            # AmbiguousEntity("Minerva", None, "Minerva"),  # , 100),
        ],
        "abstract concepts": [
            AmbiguousEntity("Triumph", "Triumph_International", "Roman_triumph"),  # , 45),
            AmbiguousEntity("Harmony", "Harmony_Gold_(mining)", "Harmony"),  # , 119),
            AmbiguousEntity("Genesis", "Genesis_Motor", "Book_of_Genesis"),  # , 141),
            AmbiguousEntity("Vision", "Vision_Research_(company)", "Visual_perception"),  # , 101),
            AmbiguousEntity("Pioneer", "Pioneer_Corporation", "American_pioneer"),  # , 95),
            AmbiguousEntity("Vanguard", "The_Vanguard_Group", "Vanguard"),  # , 128),
            AmbiguousEntity("Zenith", "Zenith_Electronics", "Zenith"),  # , 64),
            AmbiguousEntity("Allure", "Allure_(magazine)", "Interpersonal_attraction"),  # , 17),
            AmbiguousEntity("Tempo", "Tempo_(brand)", "Tempo"),  # , 59),
            AmbiguousEntity("Fidelity", "Fidelity_Investments", "Fidelity")  # , 29)
        ],
        "locations": [
            AmbiguousEntity("Amazon", "Amazon_(company)", "Amazon_River"),
            AmbiguousEntity("Cisco", "Cisco", None),
            AmbiguousEntity("Montblanc", "Montblanc_(company)", "Mont_Blanc"),
            AmbiguousEntity("Patagonia", "Patagonia,_Inc.", "Patagonia"),
            # AmbiguousEntity("Philadelphia", "Philadelphia_Cream_Cheese", "Philadelphia"),
            AmbiguousEntity("Hershey", "The_Hershey_Company", "Hershey,_Pennsylvania"),
            AmbiguousEntity("Nokia", "Nokia", "Nokia,_Finland"),
            AmbiguousEntity("Eagle Creek", "Eagle_Creek_(company)", "Eagle_Creek_River_(Arizona)"),
            AmbiguousEntity("Prosper", "Prosper_Marketplace", "Prosper,_Texas")
        ],
        "people": [
            AmbiguousEntity("Ford", "Ford_Motor_Company", "Henry_Ford"),
            AmbiguousEntity("Disney", "The_Walt_Disney_Company", "Walt_Disney"),
            AmbiguousEntity("Tesla", "Tesla,_Inc.", "Nikola_Tesla"),
            AmbiguousEntity("Boeing", "Boeing", "William_E._Boeing"),
            AmbiguousEntity("Dell", "Dell", "Michael_Dell"),
            AmbiguousEntity("Ferrero", "Ferrero_SpA", "Pietro_Ferrero"),
            # AmbiguousEntity("Merck", "Merck_Group", "Friedrich_Jacob_Merck"),
            AmbiguousEntity("Versace", "Versace", "Gianni_Versace"),
            AmbiguousEntity("Philips", "Philips", "Gerard_Philips"),
            AmbiguousEntity("Levi", "Levi_Strauss_%26_Co.", "Levi_Strauss"),
            AmbiguousEntity("Benetton", "Benetton_Group", "Benetton_family"),
        ],
        # "unambiguous_companies": [
        #     AmbiguousEntity("AirBnB", "AirBnB", None, 0),
        #     AmbiguousEntity("Hulu", "Hulu", None, 0),
        #     AmbiguousEntity("Skype", "Skype", None, 0),
        #     AmbiguousEntity("Fiverr", "Fiverr", None, 0),
        #     AmbiguousEntity("Etsy", "Etsy", None, 0),
        #     AmbiguousEntity("Zillow", "Zillow", None, 0),
        #     AmbiguousEntity("Groupon", "Groupon", None, 0),
        #     AmbiguousEntity("Zynga", "Zynga", None, 0),
        #     AmbiguousEntity("Squarespace", "Squarespace", None, 0),
        #     AmbiguousEntity("Mailchimp", "Mailchimp", None, 0),
        #     AmbiguousEntity("Zapier", "Zapier", None, 0),
        #     AmbiguousEntity("Yext", "Yext", None, 0),
        #     AmbiguousEntity("Wrike", "Wrike", None, 0),
        #     AmbiguousEntity("AppLovin", "AppLovin", None, 0),
        #     AmbiguousEntity("Addepar", "Addepar", None, 0),
        #     AmbiguousEntity("Snyk", "Snyk", None, 0),
        #     AmbiguousEntity("Synack", "Synack", None, 0),
        #     AmbiguousEntity("RealD", "RealD", None, 0),
        #     AmbiguousEntity("Alation", "Alation", None, 0),
        #     AmbiguousEntity("Molekule", "Molekule", None, 0),
        #     AmbiguousEntity("GoRuck", "GoRuck", None, 0),
        #     AmbiguousEntity("Auth0", "Auth0", None, 0)
        # ]
    }
    # companies = {
    #     "fruits": [
    #         AmbiguousEntity("Apple", "Apple_Inc.", "Apple"),  # , 48),
    #         AmbiguousEntity("Fig", "Fig_(company)", "Fig"),  # , 15),
    #         AmbiguousEntity("Mango", "Mango_(retailer)", "Mango"),  # , 43),
    #         AmbiguousEntity("Kiwi", "Kiwi.com", "Kiwifruit"),  # , 36),
    #         AmbiguousEntity("Peach", "Peach_Aviation", "Peach"),  # , 20),
    #         AmbiguousEntity("Papaya", None, "Papaya"),  # , 12),
    #         AmbiguousEntity("Orange", "Orange_S.A.", "Orange_(fruit)"),  # , 103)
    #     ],
    #     "animals": [
    #         AmbiguousEntity("Jaguar", "Jaguar_Cars", "Jaguar"),  # , 53),
    #         AmbiguousEntity("Puma", "Puma_(brand)", "Cougar"),  # , 45),
    #         AmbiguousEntity("Penguin", "Penguin_Random_House", "Penguin"),  # , 55),
    #         AmbiguousEntity("Greyhound", "Greyhound_Lines", "Greyhound"),  # , 35),
    #         AmbiguousEntity("Dove", "Dove_(brand)", "Columbidae"),  # , 50),
    #         AmbiguousEntity("Fox", "Fox_Corporation", "Fox"),  # , 89),
    #         AmbiguousEntity("Lynx", "Axe_(brand)", "Lynx"),  # , 78),
    #         AmbiguousEntity("Jellyfish", "Jellyfish.com", "Jellyfish"),  # , 12),
    #         AmbiguousEntity("Sparrow", "Sparrow_Health_System", "Sparrow"),  # , 62),
    #         AmbiguousEntity("Shell", "Shell_corporation", "Seashell"),  # , 56),
    #     ],
    #     "myths": [
    #         AmbiguousEntity("Amazon", "Amazon_(company)", "Amazons"),  # , 63),
    #         AmbiguousEntity("Nike", "Nike,_Inc.", "Nike_(mythology)"),  # , 34),
    #         AmbiguousEntity("Oracle", "Oracle_Corporation", "Oracle"),  # , 58),
    #         AmbiguousEntity("Pandora", "Pandora_(jewelry)", "Pandora"),  # , 101),
    #         AmbiguousEntity("Midas", "Midas_(automotive service)", "Midas"),  # , 38),
    #         AmbiguousEntity("Mars", "Mars_Inc.", "Mars"),  # , 134),
    #         AmbiguousEntity("Hermes", "Hermès", "Hermes"),  # , 57),
    #         AmbiguousEntity("Hyperion", "Hyperion_Solutions", "Hyperion_(Titan)"),  # , 62),
    #         AmbiguousEntity("Vulcan", "Vulcan_Inc.", "Vulcan_(mythology)"),  # , 79),
    #         AmbiguousEntity("Pegasus", "Pegasus_Airlines", "Pegasus"),  # , 86),
    #         AmbiguousEntity("Minerva", None, "Minerva"),  # , 100),
    #     ],
    #     "inspiration words": [
    #         AmbiguousEntity("Triumph", "Triumph_International", "Roman_triumph"),  # , 45),
    #         AmbiguousEntity("Harmony", "Harmony_Gold_(mining)", "Harmony"),  # , 119),
    #         AmbiguousEntity("Genesis", "Genesis_Motor", "Book_of_Genesis"),  # , 141),
    #         AmbiguousEntity("Vision", "Vision_Research_(company)", "Visual_perception"),  # , 101),
    #         AmbiguousEntity("Pioneer", "Pioneer_Corporation", "American_pioneer"),  # , 95),
    #         AmbiguousEntity("Vanguard", "The_Vanguard_Group", "Vanguard"),  # , 128),
    #         AmbiguousEntity("Zenith", "Zenith_Electronics", "Zenith"),  # , 64),
    #         AmbiguousEntity("Allure", "Allure_(magazine)", "Interpersonal_attraction"),  # , 17),
    #         AmbiguousEntity("Tempo", "Tempo_(brand)", "Tempo"),  # , 59),
    #         AmbiguousEntity("Fidelity", "Fidelity_Investments", "Fidelity")  # , 29)
    #     ],
    #     "locations": [
    #         AmbiguousEntity("Amazon", "Amazon_(company)", "Amazon_River"),
    #         AmbiguousEntity("Cisco", "Cisco", None),
    #         AmbiguousEntity("Montblanc", "Montblanc_(company)", "Mont_Blanc"),
    #         AmbiguousEntity("Patagonia", "Patagonia,_Inc.", "Patagonia"),
    #         AmbiguousEntity("Philadelphia", "Philadelphia_Cream_Cheese", "Philadelphia"),
    #         AmbiguousEntity("Hershey", "The_Hershey_Company", "Hershey,_Pennsylvania"),
    #         AmbiguousEntity("Nokia", "Nokia", "Nokia,_Finland"),
    #         AmbiguousEntity("Eagle Creek", "Eagle_Creek_(company)", "Eagle_Creek_River_(Arizona)"),
    #         AmbiguousEntity("Prosper", "Prosper_Marketplace", "Prosper,_Texas")
    #     ],
    #     "proper_names": [
    #         AmbiguousEntity("Ford", "Ford_Motor_Company", "Henry_Ford"),
    #         AmbiguousEntity("Disney", "The_Walt_Disney_Company", "Walt_Disney"),
    #         AmbiguousEntity("Tesla", "Tesla,_Inc.", "Nikola_Tesla"),
    #         AmbiguousEntity("Boeing", "Boeing", "William_E._Boeing"),
    #         AmbiguousEntity("Dell", "Dell", "Michael_Dell"),
    #         AmbiguousEntity("Ferrero", "Ferrero_SpA", "Pietro_Ferrero"),
    #         AmbiguousEntity("Merck", "Merck_Group", "Friedrich_Jacob_Merck"),
    #         AmbiguousEntity("Versace", "Versace", "Gianni_Versace"),
    #         AmbiguousEntity("Philips", "Philips", "Gerard_Philips"),
    #         AmbiguousEntity("Levi", "Levi_Strauss_%26_Co.", "Levi_Strauss"),
    #         AmbiguousEntity("Benetton", "Benetton_Group", "Benetton_family"),
    #     ],
    #     "unambiguous_companies": [
    #         AmbiguousEntity("AirBnB", "AirBnB", None, 0),
    #         AmbiguousEntity("Hulu", "Hulu", None, 0),
    #         AmbiguousEntity("Skype", "Skype", None, 0),
    #         AmbiguousEntity("Fiverr", "Fiverr", None, 0),
    #         AmbiguousEntity("Etsy", "Etsy", None, 0),
    #         AmbiguousEntity("Zillow", "Zillow", None, 0),
    #         AmbiguousEntity("Groupon", "Groupon", None, 0),
    #         AmbiguousEntity("Zynga", "Zynga", None, 0),
    #         AmbiguousEntity("Squarespace", "Squarespace", None, 0),
    #         AmbiguousEntity("Mailchimp", "Mailchimp", None, 0),
    #         AmbiguousEntity("Zapier", "Zapier", None, 0),
    #         AmbiguousEntity("Yext", "Yext", None, 0),
    #         AmbiguousEntity("Wrike", "Wrike", None, 0),
    #         AmbiguousEntity("AppLovin", "AppLovin", None, 0),
    #         AmbiguousEntity("Addepar", "Addepar", None, 0),
    #         AmbiguousEntity("Snyk", "Snyk", None, 0),
    #         AmbiguousEntity("Synack", "Synack", None, 0),
    #         AmbiguousEntity("RealD", "RealD", None, 0),
    #         AmbiguousEntity("Alation", "Alation", None, 0),
    #         AmbiguousEntity("Molekule", "Molekule", None, 0),
    #         AmbiguousEntity("GoRuck", "GoRuck", None, 0),
    #         AmbiguousEntity("Auth0", "Auth0", None, 0)
    #     ]
    # }
    with open("utils/entities.lib", "wb") as file:
        pickle.dump(entities, file)
    print("Entities saved")


def create_popularity_plots(entities):
    plt.rcParams["font.family"] = "avenir"
    color_1, color_2 = "dodgerblue", "darkorange"

    fig = plt.figure(tight_layout=True, figsize=(25, 25))
    gs = GridSpec(4, 2, figure=fig)

    axs = [
        # fig.add_subplot(gs[4, :]),
        fig.add_subplot(gs[3, :]),
        fig.add_subplot(gs[2, 1]),
        fig.add_subplot(gs[2, 0]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 0]),
    ]

    all_names, all_company_pops, all_other_pops = [], [], []

    for group_id, (group_name, group) in enumerate(entities.items()):

        group.sort(key=lambda ent: ent.company_popularity, reverse=True)
        names, company_pops, other_pops = [], [], []
        for ent in group:
            names.append(ent.name)
            company_pops.append(ent.company_popularity)
            other_pops.append(ent.other_popularity)
        all_names += names
        all_company_pops += company_pops
        all_other_pops += other_pops

        ax = axs.pop()
        X_axis = np.arange(len(names))

        # define Y axis ticks
        one_tenth = round((max(company_pops + other_pops) - min(company_pops + other_pops)) // 10, 0)
        Y_axis = list(range(min(company_pops + other_pops), max(company_pops + other_pops), one_tenth))

        Y_axis_labels = [numerize.numerize(n, decimals=0) if n_id % 2 == 0 else "" for n_id, n in enumerate(Y_axis)]

        ax.bar(X_axis - 0.2, company_pops, 0.4, label="Company", color=color_1)
        if len(set(other_pops)) > 1:
            ax.bar(X_axis + 0.2, other_pops, 0.4, label=group_name, color=color_2)
        ax.set_xticks(X_axis, names, rotation=45, horizontalalignment='right', fontsize=25)
        ax.set_yticks(Y_axis, Y_axis_labels, fontsize=25)

        ax.set_ylabel("Popularity", fontsize=25)

        ax.set_title(f"{group_name}", fontsize=40)
        ax.legend(fontsize=25)
        ax.grid(True, axis='y', linestyle=':')

        if len(axs) == 1:
            ax.set_title(f"Abstract", fontsize=40)
            # ax.set_xlabel("Entities", fontsize=25)

    # additionally create a plot with all entities
    # plt.figure(figsize=(20, 4))
    X_axis = np.arange(len(all_names))
    # Y_axis = list(range(min(all_company_pops + all_other_pops), max(all_company_pops + all_other_pops), 2500000))
    Y_axis = list(range(min(all_company_pops), max(all_company_pops), 2500000))
    Y_axis_labels = [numerize.numerize(n) if n_id % 2 == 0 else "" for n_id, n in enumerate(Y_axis)]

    ax = axs.pop()
    ax.bar(X_axis - 0.2, all_company_pops, 0.4, label="Company", color=color_1)
    # ax.bar(X_axis + 0.2, all_other_pops, 0.4, label="Other Entity", color=color_2)

    ax.set_xticks(X_axis, all_names, rotation=45, horizontalalignment='right', fontsize=20)
    ax.set_yticks(Y_axis, Y_axis_labels)

    # ax.set_xlabel("Entities")
    ax.set_ylabel("Popularity", fontsize=25)

    ax.legend(fontsize=25)
    ax.set_title("All Entities", fontsize=20)
    ax.grid(True, axis='y', linestyle=':')
    plt.show()

    # create a plot with how many entities are there for each popularity bucket
    # from matplotlib import pyplot
    # all_company_pops = list(filter(lambda x: x != 0, all_company_pops))
    # all_other_pops = list(filter(lambda x: x != 0, all_other_pops))
    # pyplot.hist(all_company_pops, bins=5, alpha=0.5, label='companies')
    # pyplot.hist(all_other_pops, bins=5, alpha=0.5, label='others')
    #
    # Y_axis = list(range(min(all_company_pops + all_other_pops), max(all_company_pops + all_other_pops), 5))
    # Y_axis_labels = [numerize.numerize(n) if n_id % 2 == 0 else "" for n_id, n in enumerate(Y_axis)]
    # pyplot.yticks(Y_axis, Y_axis_labels)
    #
    # pyplot.legend(loc='upper right')
    # pyplot.show()


# if we need to process entities first
# process_entities()

# otherwise
with open("utils/entities.lib", "rb") as file:
    companies = pickle.load(file)

companies_1 = {
    "Fruit": companies["fruits"],
    "Animal": companies["animals"],
    "Myth": companies["myths"],
    "Person": companies["people"],
    "Location": companies["locations"],
    "Abstract": companies["abstract concepts"],
}

create_popularity_plots(companies_1)
