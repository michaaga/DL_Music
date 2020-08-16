# -*- coding: utf-8 -*-

from music21 import converter
import EncDec_music21_helper
import pickle
import os

directory_name = '/Users/tzvikif/Documents/Msc/Deep Learning/Project/MID/Bach'
directory = os.fsencode(directory_name)

counter = 0
total_vec = []
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    holder = converter.parse(directory_name+'/'+filename)
    total_vec.append(EncDec_music21_helper.organize_song_midi_length(holder))
    
    counter+= 1
    print(counter)
    
with open(directory_name + 'classical_notes2.pkl', 'wb') as handle:
        pickle.dump(total_vec, handle, protocol= pickle.HIGHEST_PROTOCOL )
