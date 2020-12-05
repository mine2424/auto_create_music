from music21 import converter, pitch, interval, instrument, note, chord, stream
import os
import numpy as np
import tensorflow as tf

save_dir = './music/'

songList = os.listdir(save_dir)

# Create empty list for scores
originalScores = []

# Load and make list of stream objects
for song in songList:
    score = converter.parse(save_dir+song)
    originalScores.append(score)

# Define empty lists of lists
originalChords = [[] for _ in originalScores]
originalDurations = [[] for _ in originalScores]
originalKeys = []

# Extract notes, chords, durations, and keys
for i, song in enumerate(originalScores):
    originalKeys.append(str(song.analyze('key')))
    for element in song:
        if isinstance(element, note.Note):
            originalChords[i].append(element.pitch)
            originalDurations[i].append(element.duration.quarterLength)
        else:
            originalChords[i].append('.'.join(str(n) for n in element.pitches))
            originalDurations[i].append(element.duration.quarterLength)
    print(str(i))

# print(str(originalChords))
cMajorChords = [c for (c, k) in zip(originalChords, originalKeys)]
cMajorDurations = [c for (c, k) in zip(originalDurations, originalKeys)]

# Map unique chords to integers
uniqueChords = np.unique([i for s in originalChords for i in s])
chordToInt = dict(zip(uniqueChords, list(range(0, len(uniqueChords)))))

# Map unique durations to integers
uniqueDurations = np.unique([i for s in originalDurations for i in s])
durationToInt = dict(zip(uniqueDurations, list(range(0, len(uniqueDurations)))))

# Invert chord and duration dictionaries
intToChord = {i: c for c, i in chordToInt.items()}
intToDuration = {i: c for c, i in durationToInt.items()}

# Define sequence length
sequenceLength = 32

# Define empty arrays for train data
trainChords = []
trainDurations = []

# Construct training sequences for chords and durations
for s in range(len(cMajorChords)):
    chordList = [chordToInt[c] for c in cMajorChords[s]]
    durationList = [durationToInt[d] for d in cMajorDurations[s]]
    for i in range(sequenceLength - len(chordList)):
        trainChords.append(chordList[i:i+sequenceLength])
        trainDurations.append(durationList[i:i+sequenceLength])



# Convert to one-hot encoding and swap chord and sequence dimensions
trainChords = tf.keras.utils.to_categorical(trainChords).transpose(0,2,1)

# Convert data to numpy array of type float
trainChords = np.array(trainChords, np.float)

# Define number of samples, chords and notes, and input dimension
nSamples = trainChords.shape[0]
nChords = trainChords.shape[1]
inputDim = nChords * sequenceLength

# Set number of latent features
latentDim = 2

# Flatten sequence of chords into single dimension
trainChordsFlat = trainChords.reshape(nSamples, inputDim)

# Define encoder input shape
encoderInput = tf.keras.layers.Input(shape = (inputDim))

# Define decoder input shape
latent = tf.keras.layers.Input(shape = (latentDim))

# Define dense encoding layer connecting input to latent vector
encoded = tf.keras.layers.Dense(latentDim, activation = 'tanh')(encoderInput)

# Define dense decoding layer connecting latent vector to output
decoded = tf.keras.layers.Dense(inputDim, activation = 'sigmoid')(latent)

# Define the encoder and decoder models
encoder = tf.keras.Model(encoderInput, encoded)
decoder = tf.keras.Model(latent, decoded)

# Define autoencoder model
autoencoder = tf.keras.Model(encoderInput, decoder(encoded))

# Compile autoencoder model
autoencoder.compile(loss = 'binary_crossentropy', learning_rate = 0.01, optimizer='rmsprop')

# Train autoencoder
autoencoder.fit(trainChordsFlat, trainChordsFlat, epochs = 500)

# Generate chords from randomly generated latent vector
generatedChords = decoder(np.random.normal(size=(1,latentDim))).numpy().reshape(nChords, sequenceLength).argmax(0)

# Identify chord sequence from integer sequence
chordSequence = [intToChord[c] for c in generatedChords]

# Set location to save generated music
generated_dir = '../generated/'

# Generate stream with guitar as instrument
generatedStream = stream.Stream()
generatedStream.append(instrument.Guitar())

# Append notes and chords to stream object
for j in range(len(chordSequence)):
    try:
        generatedStream.append(note.Note(chordSequence[j].replace('.', ' ')))
    except:
        generatedStream.append(chord.Chord(chordSequence[j].replace('.', ' ')))

generatedStream.write('midi', fp=generated_dir+'autoGenerate.mid')

print('done')