import Network_Based_Recommendation_System_FUNCTIONS as homework_2
import csv
import networkx as nx
import sys


# it creates a dictionary with the keys 'graph' (graph - a nx.Graph object), 'categories' (set of all the categories),
# 'items' (set of all the movies) and 'categories_movies_dict' (dictionary with the movies per category). It receives
# as parameter a file where each line represents a category and each column (separated by \t) carries the movie_id
def create_category_category_graph(category_movie_ranking_file):
    graph_category_category = {}
    all_items = set()
    all_categories = set(range(1, 6))  # categories IDs
    g = nx.Graph()
    try:
        with open(category_movie_ranking_file, 'r') as input_file:
            input_file_csv_reader = csv.reader(input_file, delimiter='\t')

            # keeps track of all the movies already visited per category
            visited = {k: [] for k in all_categories}  # {cat_id : [visited items]}

            # iterate through the category_movies file
            curr_cat_id = 1
            # Rows are categories
            for row in input_file_csv_reader:
                # Columns are movies
                for col in row:
                    item_id = int(col)
                    all_items.add(item_id)
                    visited[curr_cat_id].append(item_id)  # mark the item as visited for the current category
                    # check if the item is present in othe categories
                    for other_cat_id in (all_categories - {curr_cat_id}):
                        # creates an edge or updates the weight attribute
                        if item_id in visited[other_cat_id]:
                            if g.has_edge(curr_cat_id, other_cat_id):
                                g[curr_cat_id][other_cat_id]['weight'] += 1
                            else:
                                g.add_edge(curr_cat_id, other_cat_id, weight=1)
                curr_cat_id += 1
        graph_category_category['graph'] = g
        graph_category_category['categories'] = all_categories
        graph_category_category['items'] = all_items
        graph_category_category['categories_movies_dict'] = visited
    finally:
        input_file.close()
    return graph_category_category


# creates a dictionary mimicking a preferences_vector, in which each key is a category, and the value is the weigh of
# the user preferences for the category.
# As the user doesn't evaluate categories, but just movies, it was necessary to cycle through the 'graph_users_items'
# to check the rating that a user gave to a movie, and then attribute this rating to the categories that this movie
# belongs to.
def create_preference_vector_for_teleporting_category_based(user_id, graph_users_items, graph_category_category,
                                                            categories_movies_dict):
    preference_vector = {}
    for category in graph_category_category['categories']:
        preference_vector[category] = 0.0

    edges = graph_users_items["graph"].edges(user_id, data=True)
    total_weight = float(sum(x[2]['weight'] for x in edges))  # sum all the ratings given by the user (to all movies)
    for edge in edges:
        item = edge[1]
        weight = float(edge[2]["weight"])  # rating given by 'user_id' to 'item'

        # count the categories that a movie belongs to. Used to divide the weigh and avoid double counting the rating,
        # which would make the preference_vector doesn't sum to 1
        n_categories = 0
        for category in graph_category_category['categories']:
            if item in categories_movies_dict[category]:
                n_categories += 1

        # update the weight that the user attributed to the category of the movie (can be more than one)
        for category in graph_category_category['categories']:
            # check the categories that the movie belongs to and divide the rating between them
            if item in categories_movies_dict[category]:
                preference_vector[category] += (weight/n_categories) / total_weight
    return preference_vector


def output(recommended_items_for_recommended_category):
    for recommended in recommended_items_for_recommended_category:
        print recommended[0], recommended[1]

def main():
    try:
        user_id = int(sys.argv[1])
    except:
        user_id = 1683  # the first user_id in u1_base_homework_format.txt

    # NORMAL FLOW FOR FINDING THE PAGERANK IN A ITEM_USER BASE
    training_set_file = './input_data/u1_base_homework_format.txt'
    test_set_file = './input_data/u1_test_homework_format.txt'
    category_movies_filename = './datasets/category_movies.txt'

    test_graph_users_items = homework_2.create_graph_set_of_users_set_of_items(test_set_file)
    training_graph_users_items = homework_2.create_graph_set_of_users_set_of_items(training_set_file)
    item_item_graph = homework_2.create_item_item_graph(training_graph_users_items)
    graph_category_category = create_category_category_graph(category_movies_filename)

    items_nodelist = item_item_graph.nodes()
    N1 = len(items_nodelist)

    categories_nodelist = graph_category_category["graph"].nodes()
    N2 = len(categories_nodelist)

    # calculate the preferences-vector in a user-movies universe
    preferences_user_items = homework_2.create_preference_vector_for_teleporting(user_id,
                                                                            training_graph_users_items)

    M1 = nx.to_scipy_sparse_matrix(item_item_graph, nodelist=items_nodelist, weight='weight', dtype=float)

    personalized_pagerank_vector_of_items = homework_2.pagerank(M1, N1, items_nodelist, alpha=0.85,
                                                                personalization=preferences_user_items)

    # Recommended items in a user-movies universe
    sorted_list_of_recommended_items_form_PERSONAL_recommendation = homework_2.create_ranked_list_of_recommended_items(
        personalized_pagerank_vector_of_items, user_id, training_graph_users_items)

    # RECOMMENDED CATEGORIES
    M2 = nx.to_scipy_sparse_matrix(graph_category_category["graph"], nodelist=categories_nodelist, weight='weight',
                                   dtype=float)

    # calculate the preferences-vector in a user-CATEGORY universe
    preferences_user_category = create_preference_vector_for_teleporting_category_based(user_id,
                                                                                        training_graph_users_items,
                                                                                        graph_category_category,
                                                                                        graph_category_category['categories_movies_dict'])

    # personalized pagerank biased by the most evaluated category by the user
    personalized_pagerank_vector_of_items = homework_2.pagerank(M2, N2, categories_nodelist,
                                                                personalization=preferences_user_category)

    recommended_category = sorted(personalized_pagerank_vector_of_items,
                                  key=personalized_pagerank_vector_of_items.get, reverse=True)[0]

    # just add to the 'recommended' list if the movie is in the recommended category
    # in other words, from the recommendation of the normal user-item pagerank, filter the items that don't
    # belong to the recommended category
    recommended_items_for_recommended_category = []
    for item in sorted_list_of_recommended_items_form_PERSONAL_recommendation:
        if item[0] in graph_category_category['categories_movies_dict'][recommended_category]:
            recommended_items_for_recommended_category.append(item)

    output(recommended_items_for_recommended_category)

if __name__ == "__main__":
    main()
