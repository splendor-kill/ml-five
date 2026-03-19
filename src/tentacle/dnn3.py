from tentacle.dnn import Pre


class DCNN3(Pre):
    def __init__(self, is_train=True, is_revive=False, is_rl=False):
        super().__init__(is_train, is_revive, is_rl)
        self.test_stat = None


if __name__ == "__main__":
    n = DCNN3(is_revive=False)
    n.run()
