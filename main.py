from music21 import converter

data_dir = './music/liz_donjuan'

# Parse MIDI file and convert notes to chords
score = converter.parse(data_dir+'.mid').chordify()

# Display as sheet music
print('start show')
print(score.show('text'))
print('done')