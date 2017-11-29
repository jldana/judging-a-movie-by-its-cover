from bs4 import BeautifulSoup
import requests
import time as tm
from urllib import request
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import os


def movie_parser():
    mov_list = []
    movie_name_data = '/Users/jdilla/Desktop/Galvanize/Week_7/Day35_031117/alt-recommender-case-study/data/movies/movies.csv'
    df = pd.read_csv(movie_name_data)
    for i in df['title']:
        i = i.rstrip(' (1234567890)')
        i = i.lower()
        i = i.replace(':', '')
        i = i.replace("'", '')
        i = i.replace(',', '')
        i = i.replace('.', '')
        i = i.replace('&', 'and')
        if i.find('(') != -1:
            spot = (i.index('(') - 1)
            extra = i[spot:]
            i = i.replace(extra, '')
        i = i.replace(' ', '_')
        if i[-4:] == '_the':
            i = i.rstrip('_the')
            mov_list.append(i)
            i = 'the_' + i
        mov_list.append(i)
    return mov_list

def scrape_linker(movie):
    link = 'https://www.rottentomatoes.com/m/{}'.format(movie)
    return link

def query(link):
    #This function takes a link http://something.smt and returns the html
    # for that page.
    response = requests.get(link)
    if response.status_code != 200:
        return '404'
    else:
        return response.content

def movie_file_name(movie):
    movieimg = movie + '.jpg'
    movie_file_name = "/Users/jdilla/Desktop/Galvanize/CapStone/judging-a-movie-by-its-cover/movie_posters/{}".format(movieimg)
    return movie_file_name

def bad_post_fname(movie):
    movieimg = movie + '.jpg'
    bad_poster_file_dir = '/Users/jdilla/Desktop/Galvanize/CapStone/judging-a-movie-by-its-cover/bad_poster/{}'.format(movieimg)
    return bad_poster_file_dir

def second_pass(no_mov):
    a_movies = []
    for i in no_mov:
        if i[-2:] == '_a':
            i = i.rstrip('_a')
            a_movies.append(i)
            i = 'a_' + i
            a_movies.append(i)
    return a_movies

def scrape(mov_list):
    no_url = []
    final_film_list = []
    for movie in mov_list:
        if os.path.isfile(movie_file_name(movie)) == True or os.path.isfile(bad_post_fname(movie)) == True:
            final_film_list.append(movie)
            print("Got it already!")
        else:
            link = scrape_linker(movie)
            html = query(link)
            if html == '404':
                print('{} not found.'.format(movie))
                no_url.append(movie)
                tm.sleep(5)
            else:
                print('Saving {}.'.format(movie))
                final_film_list.append(movie)
                file_name = movie_file_name(movie)
                soup = BeautifulSoup(html, 'html.parser')
                post_div = soup.find_all(class_="posterImage")
                post_url = post_div[0]['src']
                poster = request.urlopen(post_url)
                request.urlretrieve(post_url, file_name)
                tm.sleep(10)
    print(no_url)
    pass

def the_dont_haves(mov_list):
    no_mov = []
    for movie in mov_list:
        if os.path.isfile(movie_file_name(movie)) == False and os.path.isfile(bad_post_fname(movie)) == False:
            no_mov.append(movie)
    return no_mov

if __name__ == '__main__':


    mov_list = movie_parser()
    no_mov = the_dont_haves(mov_list)
    a_movies = second_pass(no_mov)
    # # for j, i in enumerate(mov_list):
    # #     if i == 'student_of_the_year':
    # #         print(j)
    # mov_list=(mov_list[9797:])
    scrape(a_movies)
