def changeDataFile(filename):
    with open(filename) as f:
        while True:
            line = f.readline()
            if not line:
                break
            if line.__contains__("Scanning"):
                print("**************")
                print()
            if line.__contains__("SignalLevel"):
                ()
            else: print(line.strip())