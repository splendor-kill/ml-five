from tentacle.dnn3 import DCNN3


class DCNN2(DCNN3):
    pass


if __name__ == "__main__":
    n = DCNN2(is_revive=False)
    n.run()
