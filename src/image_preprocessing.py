from skimage import data, io, filters, novice
import pandas as pd
import os
import shutil
from scraper import *
from model_aide import *

def bad_poster_mover(mov_list):
    file_dir = '/Users/jdilla/Desktop/Galvanize/CapStone/judging-a-movie-by-its-cover/bad_poster/'
    for movie in mov_list:
        im_path = movie_file_name(movie)
        if os.path.isfile(im_path) == True:
            picture = novice.open(im_path)
            if picture.size != (206, 305):
                ex_path = file_dir + movie + '.jpg'
                shutil.move(im_path, ex_path)
    pass

def output_path(class_name, movie):
    class_dir = '/Users/jdilla/Desktop/Galvanize/CapStone/judging-a-movie-by-its-cover/{}/{}.jpg'.format(class_name, movie)
    return class_dir

def class_serperator(class_df):
    unique_classes = list(class_df.genres.unique())
    for clas in unique_classes:
        move_list = class_df[class_df['genres'] == clas].title.values
        class_name = clas.lower().replace(' ', '_').replace('-', '_')
        dir_path = '/Users/jdilla/Desktop/Galvanize/CapStone/judging-a-movie-by-its-cover/{}'.format(class_name)
        os.mkdir(dir_path)
        for movie in move_list:
            for i in range(3):
                articles = ['', 'the_', 'a_']
                im_path = movie_file_name(articles[i]+movie)
                if os.path.isfile(im_path) == True:
                    ex_path = output_path(class_name, movie)
                    shutil.move(im_path, ex_path)
        # if os.path.isfile(im_path) == True:
        #     picture = novice.open(im_path)
        #     if picture.size != (206, 305):
        #         ex_path = file_dir + movie + '.jpg'
        #         shutil.move(im_path, ex_path)
    pass




if __name__ == '__main__':

    mov_list = movie_parser()
    bad_poster_mover(mov_list)
    # df_mov = movie_db()
    # # for i, j in enumerate(df_mov['genres']):
    # #     if j == '':
    # #         print(i, j)
    #
    # base_class, names = base_class(df_mov)
    # classes = class_handler(base_class)
    # class_df = reframe(classes, names)
    # class_serperator(class_df)
    # # print(class_df.head(20))
