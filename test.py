import random
from datetime import datetime

from main import NEURALNET


def stopwatch(f):
    def wrap(*args, **kw):
        start = datetime.now()
        result = f(*args, **kw)
        end = datetime.now()
        print(end - start)
        return result
    return wrap


random.seed(1)
testdata = [random.randint(0, 2) for x in range(49)]
nn = NEURALNET()
outcome = nn.feed(testdata)
print(outcome)
