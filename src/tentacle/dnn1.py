from tentacle.dnn3 import DCNN3


class DCNN1(DCNN3):
    pass


if __name__ == "__main__":
    n1 = DCNN1(is_revive=True)
    n1.run()
