tests = [
    [[2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], '3', 0],
    [[2, 2, 2, 1, 1, 1, 0, 1, 1, 2, 1, 2, 2, 0, 1, 1, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], '5', 1],
    [[1, 2, 1, 2, 2, 2, 0, 2, 2, 1, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'r', 2],
    [[1, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 2, 0, 0, 0, 0], 'r', 3],
    [[1, 1, 2, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 2, 1, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], '2', 4],
    [[1, 2, 1, 2, 1, 0, 0, 2, 1, 2, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 1, 1, 1, 2, 0, 0, 2, 2, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0], 'l', 5],
    [[2, 2, 2, 1, 1, 2, 1, 1, 2, 1, 2, 0, 0, 0, 1, 1, 1, 2, 0, 0, 0, 2, 1, 2, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'l', 6],
    [[2, 1, 1, 1, 0, 0, 0, 1, 2, 2, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0], '1', 7],
    [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 1, 2, 2, 1, 1, 0], '2', 8],
    [[2, 1, 1, 1, 2, 2, 2, 1, 2, 1, 1, 0, 0, 0, 1, 2, 2, 0, 0, 0, 0, 1, 2, 1, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0], 'r', 9],
    [[1, 2, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], '4', 10],
    [[2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 2, 1, 2, 2, 2, 0, 0], '4', 11],
    [[1, 2, 1, 1, 1, 2, 1, 2, 1, 2, 2, 2, 1, 2, 2, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 0, 0, 0, 2, 1, 2, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 1, 2, 1, 2, 0, 0], 'r', 12],
    [[1, 0, 0, 0, 0, 0, 0, 2, 1, 1, 0, 0, 0, 0, 1, 2, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 2, 1, 2, 2, 1, 2, 0], 'r', 13],
    [[2, 1, 2, 1, 0, 0, 0, 2, 2, 1, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], '3', 14],
    [[1, 2, 1, 1, 1, 0, 0, 2, 2, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 2, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 1, 2, 1, 2, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0], '3', 15],
    [[1, 2, 1, 2, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 2, 1, 2, 1, 2, 0, 0, 2, 2, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], '2', 16],
    [[1, 2, 1, 2, 1, 2, 1, 1, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0], 'r', 17],
    [[1, 1, 2, 1, 2, 0, 0, 1, 1, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], '2', 18],
    [[0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 2, 1, 0, 0, 0, 0], '3', 19],
    [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 2, 1, 1, 1, 0, 0, 0], '7', 20],
    [[0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 2, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 2, 1, 1, 1, 0, 0, 0, 1, 1, 1, 2, 2, 1, 1], '6', 21],
    [[1, 2, 2, 1, 2, 2, 0, 1, 1, 1, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], '2', 22],
    [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 2, 1, 1, 0, 0, 0, 2, 2, 1, 2, 0, 0, 0, 1, 1, 1, 2, 1, 1, 2], 'l', 23],
    [[1, 1, 2, 1, 1, 2, 0, 1, 2, 2, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0], 'r', 24],
    [[2, 2, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], '2', 25],
    [[2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 2, 2, 0, 0, 0, 0, 1, 2, 1, 1, 1, 0, 0, 2, 1, 2, 1, 2, 2, 0], '6', 26],
    [[2, 1, 2, 2, 1, 2, 1, 2, 1, 1, 1, 0, 0, 0, 1, 1, 2, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], '2', 27],
    [[1, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 2, 2, 2, 0, 0, 0, 2, 1, 2, 1, 0, 0, 0, 2, 2, 1, 1, 1, 0, 0], '4', 28],
    [[1, 1, 1, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 2, 2, 2, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'l', 29],
    [[2, 1, 2, 1, 1, 2, 0, 2, 2, 1, 2, 1, 0, 0, 1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0], 'r', 30],
    [[2, 2, 2, 1, 2, 2, 2, 1, 1, 2, 1, 2, 1, 0, 1, 2, 2, 1, 2, 1, 0, 2, 2, 1, 2, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], '5', 31],
    [[2, 2, 2, 1, 1, 0, 0, 1, 2, 1, 0, 0, 0, 0, 1, 2, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0], 'l', 32],
    [[2, 1, 2, 1, 1, 2, 2, 2, 2, 1, 2, 1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1, 2, 2, 0, 0, 0, 0, 2, 1, 1, 1, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 2, 1, 2, 1, 0, 0, 0], '5', 33],
    [[2, 1, 2, 2, 1, 0, 0, 1, 2, 2, 1, 2, 0, 0, 2, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], '3', 34],
    [[1, 1, 1, 2, 2, 2, 0, 2, 2, 1, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], '3', 35],
    [[1, 1, 2, 1, 2, 1, 2, 1, 1, 0, 0, 0, 0, 0, 2, 1, 1, 2, 1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 1, 2, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0], 'l', 36],
    [[1, 1, 1, 2, 2, 0, 0, 1, 2, 2, 0, 0, 0, 0, 2, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 2, 2, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 2, 2, 1, 0, 0, 0, 0], '5', 37],
    [[2, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 1, 1, 0, 0, 0, 0, 1, 2, 2, 1, 1, 1, 2], 'l', 38],
    [[2, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], '1', 39],
    [[0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 2, 1, 2, 1, 2, 2, 0, 1, 1, 2, 1, 2, 0, 0, 1, 2, 1, 1, 2, 2, 1], 'l', 40],
    [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 1, 2, 0, 0, 2, 1, 1, 2, 2, 1, 1, 2, 1, 1, 2, 1, 2, 2], 'l', 41],
    [[2, 1, 1, 1, 2, 2, 0, 1, 2, 2, 1, 2, 0, 0, 1, 1, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0], '4', 42],
    [[1, 1, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 1, 1, 2, 1, 0, 0, 0, 1, 2, 2, 1, 0, 0, 0, 2, 1, 1, 1, 2, 0, 0, 2, 2, 1, 2, 2, 1, 0, 2, 2, 1, 2, 2, 1, 1], 'l', 43],
    [[0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 1, 1, 2, 0, 0, 0, 0, 2, 1, 2, 0, 0, 0, 0, 2, 2, 1, 1, 1, 0, 0], '7', 44],
    [[1, 1, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0], '1', 45],
    [[1, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 1, 2, 1, 2, 1, 1, 0, 1, 1, 1, 2, 0, 0, 0, 2, 2, 1, 2, 2, 1, 2], 'r', 46],
    [[1, 2, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 1, 1, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 1, 2, 2, 1, 0, 0], 'r', 47],
    [[1, 2, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 2, 2, 0, 0, 0, 0], 'l', 48],
    [[2, 1, 2, 2, 2, 1, 0, 1, 1, 2, 1, 2, 0, 0, 2, 1, 2, 0, 0, 0, 0, 1, 2, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'r', 49],
    [[1, 1, 1, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], '1', 50],
    [[2, 1, 0, 0, 0, 0, 0, 1, 1, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 2, 1, 1, 0, 0, 0, 0, 1, 2, 2, 1, 2, 2, 1, 2, 1, 1, 2, 0, 0, 0], 'r', 51],
    [[2, 1, 2, 1, 2, 1, 2, 2, 2, 1, 1, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0], '3', 52],
    [[0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 1, 1, 1, 2, 1, 2, 0], '4', 53],
    [[2, 2, 1, 2, 2, 0, 0, 1, 2, 1, 0, 0, 0, 0, 2, 1, 2, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'r', 54],
    [[1, 2, 2, 2, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'r', 55],
    [[2, 2, 2, 1, 2, 2, 0, 2, 1, 1, 1, 0, 0, 0, 1, 1, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], '2', 56],
    [[2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 2, 1, 2, 0, 0, 0, 0, 1, 2, 2, 1, 2, 2, 0], '5', 57],
    [[2, 2, 1, 1, 2, 0, 0, 1, 2, 1, 2, 2, 0, 0, 2, 2, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], 'r', 58],
    [[2, 1, 2, 2, 2, 1, 0, 1, 2, 1, 1, 1, 0, 0, 2, 1, 2, 2, 2, 0, 0, 1, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'r', 59],
    [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 2, 2, 1, 2, 0, 0, 0, 1, 1, 2, 2, 0, 0, 0], 'l', 60],
    [[0, 0, 0, 0, 0, 0, 0, 2, 2, 1, 2, 0, 0, 0, 2, 1, 1, 1, 0, 0, 0, 1, 2, 2, 0, 0, 0, 0, 2, 1, 1, 1, 2, 1, 0, 1, 1, 2, 2, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2], '4', 61],
    [[1, 2, 2, 2, 1, 0, 0, 1, 2, 2, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], '7', 62],
    [[2, 1, 2, 2, 2, 1, 2, 2, 1, 1, 1, 2, 1, 2, 2, 1, 2, 2, 2, 0, 0, 1, 2, 1, 2, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 2, 1, 2, 0, 0, 0, 0, 1, 2, 1, 0, 0, 0, 0], '5', 63],
    [[2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 0, 0, 0, 0, 1, 2, 2, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 1, 2, 1, 1, 0, 0, 0], '6', 64],
    [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 1, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 1, 2, 1, 0, 0, 0, 0, 2, 1, 2, 2, 2, 0, 0], 'r', 65],
    [[1, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 2, 1, 1, 1, 0, 0, 0, 1, 2, 2, 2, 0, 0, 0, 2, 2, 1, 1, 0, 0, 0, 1, 2, 1, 2, 0, 0, 0], '3', 66],
    [[2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 2, 1, 0, 0, 0, 0, 1, 2, 2, 1, 0, 0, 0], '5', 67],
    [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 1, 1, 2, 0, 0, 0, 0, 1, 1, 2, 0, 0, 0, 0], '4', 68],
    [[2, 2, 2, 1, 1, 0, 0, 2, 2, 2, 1, 2, 0, 0, 1, 2, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], '3', 69],
    [[0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 1, 1, 2, 2, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 2, 2, 2, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 2, 2, 1, 0, 0, 0], '6', 70],
    [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0], '7', 71],
    [[1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 2, 2, 1, 0, 0, 0, 0, 1, 2, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0], '3', 72],
    [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 2, 2, 2, 1, 2, 2, 0], '3', 73],
    [[2, 1, 1, 2, 1, 2, 1, 1, 2, 1, 1, 0, 0, 0, 2, 1, 1, 1, 2, 2, 2, 1, 2, 2, 1, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], '4', 74],
    [[1, 1, 2, 2, 1, 0, 0, 2, 2, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], '7', 75],
    [[0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 2, 2, 1, 1, 0, 0, 0], 'r', 76],
    [[1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 1, 2, 2, 0, 0, 0, 0, 2, 1, 1, 1, 2, 0, 0, 1, 2, 2, 2, 1, 0, 0], 'l', 77],
    [[2, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 2, 2, 1, 1, 2, 0, 2, 1, 1, 0, 0, 0, 0, 2, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], '7', 78],
    [[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0], '7', 79],
    [[2, 1, 2, 2, 1, 1, 2, 1, 2, 1, 1, 0, 0, 0, 2, 2, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], '4', 80],
    [[1, 2, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 2, 1, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], '5', 81],
    [[2, 1, 1, 1, 2, 0, 0, 1, 1, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'r', 82],
    [[2, 1, 2, 2, 1, 2, 0, 1, 1, 2, 2, 1, 1, 0, 2, 1, 1, 1, 0, 0, 0, 1, 2, 2, 0, 0, 0, 0, 1, 1, 2, 2, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0], '3', 83],
    [[2, 2, 2, 1, 2, 1, 2, 1, 2, 2, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'r', 84],
    [[2, 2, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 1, 2, 2, 1, 2, 0, 0, 2, 1, 2, 1, 2, 1, 2], 'l', 85],
    [[2, 2, 1, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 2, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], '6', 86],
    [[2, 1, 2, 2, 2, 1, 2, 1, 2, 1, 2, 1, 1, 2, 2, 2, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], '3', 87],
    [[1, 2, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 2, 2, 0, 0, 0, 0, 2, 2, 1, 2, 0, 0, 0, 2, 1, 2, 1, 1, 2, 1, 1, 1, 1, 2, 2, 1, 2], '4', 88],
    [[2, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 2, 1, 1, 0, 0, 0, 1, 2, 2, 1, 0, 0, 0], 'r', 89],
    [[2, 2, 1, 2, 1, 2, 1, 1, 2, 2, 1, 0, 0, 0, 1, 1, 2, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0], '5', 90],
    [[0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 2, 1, 2, 2, 1, 0, 0, 1, 1, 1, 2, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2], 'l', 91],
    [[2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 0, 0, 0, 0, 2, 2, 1, 2, 1, 2, 0, 1, 2, 1, 1, 2, 1, 0], 'r', 92],
    [[2, 2, 1, 2, 2, 1, 2, 1, 2, 1, 2, 0, 0, 0, 1, 1, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], 'l', 93],
    [[2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 2, 2, 2, 1, 0, 0, 0, 1, 1, 1, 2, 2, 0, 0, 1, 1, 1, 2, 1, 0, 0], 'l', 94],
    [[2, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 2, 1, 1, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0], 'r', 95],
    [[2, 2, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0], '7', 96],
    [[1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 2, 2, 1, 0, 0, 0, 0, 2, 1, 1, 2, 1, 0, 0, 1, 2, 2, 1, 2, 2, 2], 'l', 97],
    [[2, 2, 1, 1, 1, 2, 2, 1, 2, 0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], '4', 98],
    [[2, 1, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 0, 0, 1, 1, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], '5', 99],
]
