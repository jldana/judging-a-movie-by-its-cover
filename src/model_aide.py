import pandas as pd
from collections import Counter

def movie_db():
    movie_name_data = '/Users/jdilla/Desktop/Galvanize/Week_7/Day35_031117/alt-recommender-case-study/data/movies/movies.csv'
    return pd.read_csv(movie_name_data)

def base_class(df_mov):
    base_class = []
    df2 = df_mov['genres']
    for i in df2:
        base_class.append(i.split('|'))
    return base_class

def class_handler(base_class):
    # Remove IMAX - not a genre
    n_l = [[x for x in genre if x != 'IMAX'] for genre in base_class]
    # Remove crime - not a genre
    nl1 = [['Action' if x == 'Crime' else x for x in genre ] for genre in n_l]
    # Remove crime - not a genre
    nl1_1 = [['Action'  if x == 'War' else x for x in genre] for genre in nl1]
    # Change no genre to UNKNOWN - easier to type...
    n_l2 = [[r if r != '(no genres listed)' else 'UNKNOWN' for r in genre] for genre in nl1_1]
    # If children was tagged called it a children's movie.
    nl3 = [['Children'] if 'Children' in genre else genre for genre in n_l2]
    # Remove War Tag. War movies are action or drama movies that happen in the scetting of war.
    #nl4 = [[x for x in genre if x != 'War' and len(genre) > 1] for genre in nl3]
    # It's a comedy if comedy comes up in titles with more than 2 genres.
    nl5 = [['Comedy'] if 'Comedy' in genre and len(genre) > 2 else genre for genre in nl3]
    # It's a comedy if it has 2 genres and the other isn't Drama. Comedy Drama is its own category.
    nl5_5 = [['Comedy'] if 'Comedy' in genre and 'Drama' not in genre else genre for genre in nl5]
    # It's an Action Thriller if both Action and Thriller are present.
    nl6 = [['Action', 'Thriller'] if 'Thriller' in genre and 'Action' in genre else genre for genre in nl5_5]
    # It's action Adventure if both are in genres.
    nl6_6 = [['Action', 'Adventure'] if 'Adventure' in genre and 'Action' in genre else genre for genre in nl6]
    # Drama if Drama and Romance exist.
    nl7 = [['Drama'] if 'Drama' in genre and 'Romance' in genre else genre for genre in nl6_6]
    # # Crime movies are not a genre.
    # nl8 = [[x for x in genre if x != 'Crime' and len(genre) > 1] for genre in nl7]
    # Romantic Comedies are comedies.
    nl9 = [['Comedy'] if 'Comedy' in genre and 'Romance' in genre else genre for genre in nl7]
    # Musicals are musicals - Rent vs. Cats vs. Oklahoma
    nl10 = [['Musical'] if 'Musical' in genre else genre for genre in nl9]
    #Animated Movies are animated. Most seem to be childrens movies, but some are for adults...
    nl11 = [['Animation'] if 'Animation' in genre else genre for genre in nl10]
    # Documentaries are Documentaries.
    nl12 = [['Documentary'] if 'Documentary' in genre else genre for genre in nl11]
    # Thriller is the most prominent of the subgenres.
    nl13 = [['Thriller'] if 'Thriller' in genre and 'Action' not in genre else genre for genre in nl12]
    # Horror tags indicate some level of horror.
    nl14 = [['Horror'] if 'Horror' in genre else genre for genre in nl13]
    # If there's only two subs Sci-Fi wins.
    nl15 = [['Sci-Fi'] if 'Sci-Fi' in genre and len(genre) > 2 else genre for genre in nl14]
    # Westerns share a certain je ne sais qua?
    nl16 = [['Western'] if 'Western' in genre else genre for genre in nl15]
    # Films-noir are of an ilk.
    nl17 = [['Film-Noir'] if 'Film-Noir' in genre else genre for genre in nl16]
    #Drama is the dominant class in Adventrue and Action Dramas.
    nl18 = [['Drama'] if 'Drama' in genre and 'Action' in genre else genre for genre in nl17]
    #Drama is the dominant class in Adventrue and Action Dramas.
    nl19 = [['Drama'] if 'Drama' in genre and 'Adventure' in genre else genre for genre in nl18]
    # Mystery is a subclass.
    for i in nl19:
        if len(i) > 1:
            if 'Mystery' in i:
                i = i.remove('Mystery')
    #Fantasy is a subclass
    for i in nl19:
        if len(i) > 1:
            if 'Fantasy' in i:
                i = i.remove('Fantasy')

    # At this stage romance is a subclass.
    for i in nl19:
        if len(i) > 1:
            if 'Romance' in i:
                i = i.remove('Romance')
    # At this stage sci-fi is also a subclass.
    for i in nl19:
        if len(i) > 1:
            if 'Sci-Fi' in i:
                i = i.remove('Sci-Fi')
    for i in nl19:
        if len(i) == 2:
            if i[1] == 'Action':
                i.pop(1)
    for i in nl19:
        if len(i) > 1:
            if 'Action' in i:
                if 'Thriller' in i:
                    i = i.remove('Thriller')
    classes = nl19
    return classes

def reframe(classes):
    classes = [' '.join(genre) for genre in classes]
    new_ser = pd.Series(classes, name='genres')
    return pd.DataFrame(new_ser)

if __name__ == '__main__':
    df_mov = movie_db()
    # for i, j in enumerate(df_mov['genres']):
    #     if j == '':
    #         print(i, j)

    base_class = base_class(df_mov)
    classes = class_handler(base_class)
    class_df = reframe(classes)





#
# len(unigen)
# len(uni_gen)
# for i in uni_gen:
#     print(i)
# for i in uni_gen:
#     i.split('|')
# uni_gen
# for i in uni_gen:
#     i = [i.split('|')]
# uni_gen
# for i in uni_gen:
#     i = [i.split('|')]
#     print(i)
# for i in uni_gen:
#     i = i.split('|')
#     print(i)
# uni_gen
# uni_gen_list
# uni_gen_list = []
# for i in uni_gen:
#     i = i.split('|')
#     uni_gen_list.append(i)
# uni_gen_list
# history
