import numpy as np

mp2dlib_correspondence = [

    ## Face Contour
    [127],  # 1
    [234],  # 2
    [93],  # 3
    [132, 58],  # 4
    [58, 172],  # 5
    [136],  # 6
    [150],  # 7
    [176],  # 8
    [152],  # 9
    [400],  # 10
    [379],  # 11
    [365],  # 12
    [397, 288],  # 13
    [361],  # 14
    [323],  # 15
    [454],  # 16
    [356],  # 17

    ## Right Brow
    [70],  # 18
    [63],  # 19
    [105],  # 20
    [66],  # 21
    [107],  # 22

    ## Left Brow
    [336],  # 23
    [296],  # 24
    [334],  # 25
    [293],  # 26
    [300],  # 27

    ## Nose
    [168, 6],  # 28
    [197, 195],  # 29
    [5],  # 30
    [4],  # 31
    [75],  # 32
    [97],  # 33
    [2],  # 34
    [326],  # 35
    [305],  # 36

    ## Right Eye
    [33],  # 37
    [160],  # 38
    [158],  # 39
    [133],  # 40
    [153],  # 41
    [144],  # 42

    ## Left Eye
    [362],  # 43
    [385],  # 44
    [387],  # 45
    [263],  # 46
    [373],  # 47
    [380],  # 48

    ## Upper Lip Contour Top
    [61],  # 49
    [39],  # 50
    [37],  # 51
    [0],  # 52
    [267],  # 53
    [269],  # 54
    [291],  # 55

    ## Lower Lip Contour Bottom
    [321],  # 56
    [314],  # 57
    [17],  # 58
    [84],  # 59
    [91],  # 60

    ## Upper Lip Contour Bottom
    [78],  # 61
    [82],  # 62
    [13],  # 63
    [312],  # 64
    [308],  # 65

    ## Lower Lip Contour Top
    [317],  # 66
    [14],  # 67
    [87],  # 68
]

for ri in range(68):
    if len(mp2dlib_correspondence[ri]) == 1:
        idx = mp2dlib_correspondence[ri][0]
        mp2dlib_correspondence[ri] = [idx, idx]


def convert_landmarks_mediapipe_to_dlib(lmks_mp: np.array):
    """
    Convert the 478 Mediapipe dense landmarks to
    the 68 Dlib sparse landmarks
    input:
        - lmks_mp: Mediapipe landmarks, [478, 2] or [478, 3]
    return:
        - lmks_mp2dlib: Converted Dlib landmarks, [68, 2] or [68, 3]
    """
    # convert the landmarks
    lmks_mp2dlib = lmks_mp[mp2dlib_correspondence].mean(axis=1)

    return lmks_mp2dlib
