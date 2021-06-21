
def readLines(filename):
    list = []
    count = 0;
    with open(filename, encoding="utf8") as f:

        while True:
            count += 1

            # Get next line from file
            line = f.readline()
            list.append(line)
            # if line is empty
            # end of file is reached
            if not line:
                break
            # print("Line{}: {}".format(count, line.strip()))

        f.close()
    return list

if __name__ == '__main__':
    bert_vocab = readLines("vocab-bert.txt")
    original_vocab = readLines("learning/treelstm/data/qald/vocab-cased-qald.txt")


    s = set(bert_vocab)
    temp3 = [x for x in original_vocab if x not in s]
    print(len(temp3))

    print("")

    with open("listvocab_qald.txt", "w") as output:
        for i in temp3:
            output.write(i)




