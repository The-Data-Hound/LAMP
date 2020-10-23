#this may be all that is needed
import os
def load_thesis_data():
    '''Returns dictionary of Absolute File Paths for TSV files'''
    print('Dictionary of Thesis Data as Absolute File Locations\n')
    this_file = os.path.abspath(__file__)
    this_dir = os.path.dirname(this_file)
    parent = os.path.abspath(os.path.join(this_dir, os.pardir))
    grandparent = os.path.abspath(os.path.join(parent, os.pardir))
    loc = grandparent+'/data/'
    d = {}
    d['add_drop']=loc+'add_drop.tsv'
    d['bacteria_lamp_labeled_data'] = loc+'bacteria_lamp_labeled_data.tsv'
    d['species_list']=loc+'species_list.tsv'
    d['bacteria_lamp_pretrained']=loc+'bacteria_lamp_pretrained'
    return d
