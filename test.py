import random
from datetime import datetime

from main import NEURALNET
from testdata import tests as testdata


def stopwatch(f):
    def wrap(*args, **kw):
        start = datetime.now()
        result = f(*args, **kw)
        end = datetime.now()
        print(end - start)
        return result

    return wrap


random.seed(10)


@stopwatch
def testrun(x):
    record = 0
    for run in range(x):
        points = 0
        for test in testdata:
            nn = NEURALNET()
            outcome = nn.feed(test[0])
            if outcome == 8:
                outcome = "r"
            elif outcome == 7:
                outcome = "l"
            elif outcome <= 6:
                outcome += 1
            if test[1] == outcome:
                points += 1
            #print(test[1], outcome)
        if points > record:
            record = points
            nn.saveState()
        print("run:", run, "points:", points, "/", len(testdata))
    print("record: ", record)


# print(outcome)

testrun(100)
