from bs4 import BeautifulSoup
import requests
import time as tm
from urllib import request
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import os

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
    pass

def movie_file_name(movie, genre_class = 'Animation'):
    movieimg = movie + '.jpg'
    movie_file_name = "/Users/jdilla/Desktop/Galvanize/CapStone/judging-a-movie-by-its-cover/{}/{}".format(genre_class, movieimg)
    return movie_file_name

def namer(movie):
    movie = movie.lower()
    movie = movie.replace("'", '')
    movie = movie.replace('&', 'and')
    movie = movie.replace(':', '')
    movie = movie.replace('!', '')
    movie = movie.replace('.', '')
    movie = movie.replace(',', '')
    movie = movie.replace('/', '')
    movie = movie.replace('(', '')
    movie = movie.replace(')', '')
    movie = movie.replace('-', '')
    movie = movie.replace(' ', '_')
    movie = movie.replace('__', '_')
    return movie

def the_scrape_part_ix_the_reckoning(the_list, input1):
    counter = 0
    for item in the_list:
        movie = namer(item[0])
        post_url = item[1]
        file_name = movie_file_name(movie, genre_class = input1)
        if os.path.isfile(file_name) != True:
            counter += 1
            print('Saving {}: {}.'.format(movie, counter))
            poster = request.urlopen(post_url)
            request.urlretrieve(post_url, file_name)
            tm.sleep(9)
        else:
            counter += 1
            print("Got that one already!: {}".format(counter))
    pass

def extract(html):
    soup = BeautifulSoup(open(html), 'html.parser')
    post_div = soup.find_all(class_='poster')
    post_urls=[]
    post_names = []
    for i in post_div:
        if i != 'https://staticv2-4.rottentomatoes.com/static/images/redesign/poster_default.gif':
            post_urls.append(i['src'])
            post_names.append(i['alt'])
    return zip(post_names, post_urls)

if __name__ == '__main__':
    input1 = input("What's the genre?")
    html = '/Users/jdilla/Downloads/{}.html'.format(input1)
    the_list = extract(html)
    the_scrape_part_ix_the_reckoning(the_list, input1)
