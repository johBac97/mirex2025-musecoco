## MuseCoco Input Details
Here are possible input tokens to MuseCoco
    "I1s2": "Instrument", a 28-dim multi-hot vector. ("乐器个列表，每个列表长度为3，依次为是、否、NA")
        Format: I1s2_[inst_id]_0  
        inst_id: [0-27]
        There are condition tokens like I1s2_[inst_id]_1 and ..._2. Deprecated for simplicity.
    "R1": "Rhythm Danceability", 
        R1_0: dancable
        R1_1: not dancable
        R1_2: NA
    "R3": "Rhythm Intensity",  
        R3_0: not intense
        R3_1: med
        R3_2: intense
    "S2s1": "Artist", 
        Format: S2s1_[id]
        id's value:
            'beethoven': 0,
            'mozart': 1,
            'chopin': 2,
            'schubert': 3,
            'schumann': 4,
            'bach-js': 5,
            'haydn': 6,
            'brahms': 7,
            'Handel': 8,
            'tchaikovsky': 9,
            'mendelssohn': 10,
            'dvorak': 11,
            'liszt': 12,
            'stravinsky': 13,
            'mahler': 14,
            'prokofiev': 15,
            'shostakovich': 16,
    "S4": "Genre",
        S4_[gid]_0
        gid's value:
            'New Age': 0,
            'Electronic': 1,
            'Rap': 2,
            'Religious': 3,
            'International': 4,
            'Easy_Listening': 5,
            'Avant_Garde': 6,
            'RnB': 7,
            'Latin': 8,
            'Children': 9,
            'Jazz': 10,
            'Classical': 11,
            'Comedy_Spoken': 12,
            'Pop_Rock': 13,
            'Reggae': 14,
            'Stage': 15,
            'Folk': 16,
            'Blues': 17,
            'Vocal': 18,
            'Holiday': 19,
            'Country': 20,
            'Symphony': 21,
        Similarly, S4_[gid]_1 and S4_[gid]_2 are deprecated.
    "B1s1": "Bar", represent bar个数区间的id
        B1s1_[bid]
        bid's value:
            0：1-4，
            1：5-8，
            2：9-12，
            3：13-16
    "TS1s1": "Time Signature",
        TS1s1_[tsid]
        tsid's value:
            0: (4, 4), 
            1: (2, 4), 
            2: (3, 4), 
            3: (1, 4), 
            4: (6, 8), 
            5: (3, 8)
    "K1": "Key",
        K1_0: major
        K1_1: minor
        K1_2: unknown
    "T1s1": "Tempo",
        T1s1_[tid]
        tid's value:
            0表示慢，
            1表示适中
            2表示快。
    "P4": "Pitch Range", n_octaves
        P4_[0-12]
        0个8度，1个8度，...，11个8度, NA
    "EM1": "Emotion", but don't know the mapping. Detail not specified. Deprecated.
        EM1_[0-4] 
    "TM1": “Time", output duration in seconds (Deprecated)
        TM1_[0-5]
            0表示(0-15]秒，
            1表示(15-30]秒，
            2表示30-45秒，
            3表示45-60秒，
            4表示60秒以上


 # piano 0:
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0,

        # keyboard 1:
        6: 1,
        7: 1,
        8: 1,
        9: 1,

        # percussion 2:
        11: 2,
        12: 2,
        13: 2,
        14: 2,
        15: 2,
        47: 2,
        55: 2,
        112: 2,
        113: 2,
        115: 2,
        117: 2,
        119: 2,

        # organ 3:
        16: 3,
        17: 3,
        18: 3,
        19: 3,
        20: 3,
        21: 3,
        22: 3,
        23: 3,

        # guitar 4:
        24: 4,
        25: 4,
        26: 4,
        27: 4,
        28: 4,
        29: 4,
        30: 4,
        31: 4,

        # bass 5:
        32: 5,
        33: 5,
        34: 5,
        35: 5,
        36: 5,
        37: 5,
        38: 5,
        39: 5,
        43: 5,

        # violin 6:
        40: 6,

        # viola 7:
        41: 7,

        # cello 8:
        42: 8,

        # harp 9:
        46: 9,

        # strings 10:
        44: 10,
        45: 10,
        48: 10,
        49: 10,
        50: 10,
        51: 10,

        # voice 11:
        52: 11,
        53: 11,
        54: 11,

        # trumpet 12:
        56: 12,
        59: 12,

        # trombone 13:
        57: 13,

        # tuba 14:
        58: 14,

        # horn 15:
        60: 15,
        69: 15,

        # brass 16:
        61: 16,
        62: 16,
        63: 16,

        # sax 17:
        64: 17,
        65: 17,
        66: 17,
        67: 17,

        # oboe 18:
        68: 18,

        # bassoon 19:
        70: 19,

        # clarinet 20:
        71: 20,

        # piccolo 21:
        72: 21,

        # flute 22:
        73: 22,
        75: 22,

        # pipe 23:
        74: 23,
        76: 23,
        77: 23,
        78: 23,
        79: 23,

        # synthesizer 24:
        80: 24,
        81: 24,
        82: 24,
        83: 24,
        84: 24,
        85: 24,
        86: 24,
        87: 24,
        88: 24,
        89: 24,
        90: 24,
        91: 24,
        92: 24,
        93: 24,
        94: 24,
        95: 24,

        # ethnic instrument 25:
        104: 25,
        105: 25,
        106: 25,
        107: 25,
        108: 25,
        109: 25,
        110: 25,
        111: 25,

        # sound effect 26:
        10: 26,
        120: 26,
        121: 26,
        122: 26,
        123: 26,
        124: 26,
        125: 26,
        126: 26,
        127: 26,
        96: 26,
        97: 26,
        98: 26,
        99: 26,
        100: 26,
        101: 26,
        102: 26,
        103: 26,

        # drum 27:
        128: 27,
        118: 27,
        114: 27,
        116: 27,
    }

    inst_class_id_to_inst_class_name = {
        # piano 0:
        0: 'piano',

        # keyboard 1:
        1: 'keyboard',

        # percussion 2:
        2: 'percussion',

        # organ 3:
        3: 'organ',

        # guitar 4:
        4: 'guitar',

        # bass 5:
        5: 'bass',

        # violin 6:
        6: 'violin',

        # viola 7:
        7: 'viola',

        # cello 8:
        8: 'cello',

        # harp 9:
        9: 'harp',

        # strings 10:
        10: 'strings',

        # voice 11:
        11: 'voice',

        # trumpet 12:
        12: 'trumpet',

        # trombone 13:
        13: 'trombone',

        # tuba 14:
        14: 'tuba',

        # horn 15:
        15: 'horn',

        # brass 16:
        16: 'brass',

        # sax 17:
        17: 'sax',

        # oboe 18:
        18: 'oboe',

        # bassoon 19:
        19: 'bassoon',

        # clarinet 20:
        20: 'clarinet',

        # piccolo 21:
        21: 'piccolo',

        # flute 22:
        22: 'flute',

        # pipe 23:
        23: 'pipe',

        # synthesizer 24:
        24: 'synthesizer',

        # ethnic instrument 25:
        25: 'ethnic instrument',

        # sound effect 26:
        26: 'sound effect',

        # drum 27:
        27: 'drum',
    }


# CTP:Instruments: I1s2_0_0, I1s2_11_0, >32 bars: B1s1_4, >60s: TM1_4  Vocal: S4_18_0
init_seq = [
    'I1s2_0_0', 'I1s2_1_2', 'I1s2_2_2', 'I1s2_3_2', 'I1s2_4_2', 'I1s2_5_2', 'I1s2_6_2', 'I1s2_7_2', 'I1s2_8_2', 
    'I1s2_9_2', 'I1s2_10_2', 'I1s2_11_0', 'I1s2_12_2', 'I1s2_13_2', 'I1s2_14_2', 'I1s2_15_2', 'I1s2_16_2', 'I1s2_17_2', 
    'I1s2_18_2', 'I1s2_19_2', 'I1s2_20_2', 'I1s2_21_2', 'I1s2_22_2', 'I1s2_23_2', 'I1s2_24_2', 'I1s2_25_2', 'I1s2_26_2', 
    'I1s2_27_2', 
    'I4_28', 'C1_4', 
    'R1_2', 'R3_3', 'S2s1_17', 
    'S4_0_2', 'S4_1_2', 'S4_2_2', 'S4_3_2', 'S4_4_2', 'S4_5_2', 
    'S4_6_2', 'S4_7_2', 'S4_8_2', 'S4_9_2', 'S4_10_2', 'S4_11_2', 'S4_12_2', 'S4_13_2', 'S4_14_2', 'S4_15_2', 'S4_16_2', 
    'S4_17_2', 'S4_18_0', 'S4_19_2', 'S4_20_2', 'S4_21_2', 
    'B1s1_4', 'TS1s1_0', 'K1_2', 'T1s1_1', 'P4_12', 
    'ST1_14', 'EM1_4', 
    'TM1_4', '<sep>']


# Lsh :Instruments: I1s2_0_0, I1s2_11_0, 8 bars: B1s1_1, >60s: TM1_1  Vocal: S4_18_0
init_seq = [
    'I1s2_0_0', 'I1s2_1_2', 'I1s2_2_2', 'I1s2_3_2', 'I1s2_4_2', 'I1s2_5_2', 'I1s2_6_2', 'I1s2_7_2', 'I1s2_8_2', 
    'I1s2_9_2', 'I1s2_10_2', 'I1s2_11_0', 'I1s2_12_2', 'I1s2_13_2', 'I1s2_14_2', 'I1s2_15_2', 'I1s2_16_2', 'I1s2_17_2', 
    'I1s2_18_2', 'I1s2_19_2', 'I1s2_20_2', 'I1s2_21_2', 'I1s2_22_2', 'I1s2_23_2', 'I1s2_24_2', 'I1s2_25_2', 'I1s2_26_2', 
    'I1s2_27_2', 
    'I4_28', 'C1_4', 
    'R1_2', 'R3_3', 'S2s1_17', 
    'S4_0_2', 'S4_1_2', 'S4_2_2', 'S4_3_2', 'S4_4_2', 'S4_5_2', 
    'S4_6_2', 'S4_7_2', 'S4_8_2', 'S4_9_2', 'S4_10_2', 'S4_11_2', 'S4_12_2', 'S4_13_2', 'S4_14_2', 'S4_15_2', 'S4_16_2', 
    'S4_17_2', 'S4_18_0', 'S4_19_2', 'S4_20_2', 'S4_21_2', 
    'B1s1_1', 'TS1s1_0', 'K1_2', 'T1s1_1', 'P4_12', 
    'ST1_14', 'EM1_4', 
    'TM1_1', '<sep>']


# Acc :Instruments: I1s2_0_0, I1s2_11_2, 8 bars: B1s1_1, >60s: TM1_1  Vocal: S4_18_0
init_seq = [
    'I1s2_0_0', 'I1s2_1_2', 'I1s2_2_2', 'I1s2_3_2', 'I1s2_4_2', 'I1s2_5_2', 'I1s2_6_2', 'I1s2_7_2', 'I1s2_8_2', 
    'I1s2_9_2', 'I1s2_10_2', 'I1s2_11_2', 'I1s2_12_2', 'I1s2_13_2', 'I1s2_14_2', 'I1s2_15_2', 'I1s2_16_2', 'I1s2_17_2', 
    'I1s2_18_2', 'I1s2_19_2', 'I1s2_20_2', 'I1s2_21_2', 'I1s2_22_2', 'I1s2_23_2', 'I1s2_24_2', 'I1s2_25_2', 'I1s2_26_2', 
    'I1s2_27_2', 
    'I4_28', 'C1_4', 
    'R1_2', 'R3_3', 'S2s1_17', 
    'S4_0_2', 'S4_1_2', 'S4_2_2', 'S4_3_2', 'S4_4_2', 'S4_5_2', 
    'S4_6_2', 'S4_7_2', 'S4_8_2', 'S4_9_2', 'S4_10_2', 'S4_11_2', 'S4_12_2', 'S4_13_2', 'S4_14_2', 'S4_15_2', 'S4_16_2', 
    'S4_17_2', 'S4_18_2', 'S4_19_2', 'S4_20_2', 'S4_21_2', 
    'B1s1_1', 'TS1s1_0', 'K1_2', 'T1s1_1', 'P4_12', 
    'ST1_14', 'EM1_4', 
    'TM1_1', '<sep>']