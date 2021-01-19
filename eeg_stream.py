from pylsl import StreamInlet, resolve_stream
import numpy as np

print('looking for an EEG stream...')
stream = resolve_stream()
print(stream)

inlet = StreamInlet(stream[0])
chunk = np.zeros([10, 10, 11])
counter = 0
#   0,  1,  2,   3,  4,  5,  6,  7, -6, -5,  -4, -3, -2,  -1
# AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4

def convert2D(data, Y=10, X=11):
    data_2D = np.zeros([Y, X])
    data_2D[0] = (       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0)
    data_2D[1] = (       0,       0,       0,       0, data[0],       0,data[-1],       0,       0,       0,       0)
    data_2D[2] = (       0, data[1],       0, data[2],       0,       0,       0,data[-3],       0,data[-2],       0)
    data_2D[3] = (       0,       0, data[3],       0,       0,       0,       0,       0,data[-4],       0,       0)
    data_2D[4] = (       0, data[4],       0,       0,       0,       0,       0,       0,       0,data[-5],       0)
    data_2D[5] = (       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0)
    data_2D[6] = (       0, data[5],       0,       0,       0,       0,       0,       0,       0,data[-6],       0)
    data_2D[7] = (       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0)
    data_2D[8] = (       0,       0,       0,       0, data[6],       0, data[7],       0,       0,       0,       0)
    data_2D[9] = (       0,       0,       0,       0,       0,       0,       0,       0,       0,       0,       0)
    return data_2D

while True:
    sample, timestamp = inlet.pull_sample()
    sample = np.asarray(sample[3:-1]) - 4100
    chunk[counter] = convert2D(sample)
    counter += 1
    if counter == 10:
        print('READY')
        counter = 0
        print(chunk)
        chunk = np.zeros([10, 10, 11])
        quit()
    print(timestamp, sample[3:-1])
