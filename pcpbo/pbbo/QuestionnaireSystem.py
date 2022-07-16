class QuestionnaireSystem:
    def __init__(self, cfg, method):
        self.method = method

    def gen_query(self):
        raise NotImplementedError()

    def get_answer(self, answer):
        self.method.get_answer(answer)

    def terminate(self):
        self.method.terminate()


class TrjBasedQuestionnaireSystem(QuestionnaireSystem):
    def __init__(self, cfg, method):
        super(TrjBasedQuestionnaireSystem, self).__init__(cfg, method)

    def gen_query(self):
        trj_query, _ = self.method.gen_query()
        return trj_query


class ImgBasedQuestionnaireSystem(QuestionnaireSystem):
    def __init__(self, cfg, method):
        super(ImgBasedQuestionnaireSystem, self).__init__(cfg, method)

    def gen_query(self):
        _, img_query = self.method.gen_query()
        return img_query
