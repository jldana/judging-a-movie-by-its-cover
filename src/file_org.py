import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def comp_dir(comed, comp2):
    comp1_files = os.listdir(comed)
    comp2_files = os.listdir(comp2)
    return set(comp1_files) , set(comp2_files)

def del_func(list_to_del, where_from):
    for im in list_to_del:
        fil_pat = where_from + '/' + im
        os.remove(fil_pat)
        print('{} removed from {}.'.format(im, where_from))
    pass

if __name__ == '__main__':

    act_thrill = '../less_data/data/train/Action_Thriller'
    comed = '../less_data/data/train/Comedy'
    drama = '../less_data/data/train/Drama'
    comp2 = input('What should we compare to Comedy? ')
    if comp2 == 'Drama':
        comp3 = drama
    else:
        comp3 = act_thrill
    comed_files, comp3_files = comp_dir(comed, comp3)
    com_files = comed_files & comp3_files
    print(com_files)
    del_func(com_files, act_thrill)
