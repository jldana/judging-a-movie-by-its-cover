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
            i = i.replace('_the', '')
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

def scrape(mov_list):
    no_url = []
    for movie in mov_list:
        if os.path.isfile(movie_file_name(movie)) == True:
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
                file_name = movie_file_name(movie)
                soup = BeautifulSoup(html, 'html.parser')
                post_div = soup.find_all(class_="posterImage")
                post_url = post_div[0]['src']
                poster = request.urlopen(post_url)
                request.urlretrieve(post_url, file_name)
                tm.sleep(10)
    print(no_url)
    print(got_it)
    pass


if __name__ == '__main__':

    mov_list = movie_parser()
    # for j, i in enumerate(mov_list):
    #     if i == 'stella_dallas':
    #         print(j)
    mov_list=(mov_list[7300:])
    scrape(mov_list)
