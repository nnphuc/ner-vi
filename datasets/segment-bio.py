def process_line(f,line):
    tokens = line.split('\t')
    if len(tokens)<2:
        f.write("\n")
    else:
        w0 = tokens[0]
        ws = w0.split(" ")
        N = len(ws)
        if N == 1:
            f.write("{}\t{}\n".format(w0,"O"))
        else:
            f.write("{}\t{}\n".format(ws[0],"B"))
            for i in range(1,N):
                f.write("{}\t{}\n".format(ws[i],"I"))
        #f.write("\n")

import sys

with open("train-swg.txt","w") as f:
    fin = open("train.txt")
    for line in fin.readlines():
        line = line.strip()
        process_line(f,line)

