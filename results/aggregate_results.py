def main():
    file1 = open('s1hash.txt', 'r')
    lines = file1.readlines()

    num_cases = 1000
    tt = 0
    tne = 0
    tth = 0

    #     \item \textit{AT: } Average Time per Test Case
    # \item \textit{TT: } Total Time to Run All Test Cases
    # \item \textit{ANE: } Average Nodes Explored per Test Case
    # \item \textit{TNE: }Total Number of Nodes Explored across all test cases
    # \item \textit{NES: } Nodes Explored per Second (ANE/AT)
    # \item \textit{TNES: }Total Nodes Explored per Second across all test cases (TNE/TT)
    # \item \textit{ATH: }Average Number of Transposition Table Hits (for Transposition Table Implementation)
    # \item \textit{TTH: }Total Number of Transposition Table Hits (for Transposition Table Implementation)


    for line in lines:
        values = line.split()
        if len(values) == 5:
            tth += int(values[4])
        
        tt += int(values[2])
        tne += int(values[3])

    at = tt/num_cases
    ane = tne/num_cases
    nes = tne/tt
    ath = tth/num_cases

    print("AT: ", at)
    print("TT: ", tt)
    print("ANE: ", ane)
    print("TNE: ", tne)
    print("NES: ", nes)
    print("ATH: ", ath)
    print("TTH: ", tth)

main()