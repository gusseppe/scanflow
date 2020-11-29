class Node(object):
    """
        Abstract base Node class.

    """
    def __init__(self, name: str):
        self.name = name

class Checker(Node):

    def __init__(self,
                 name:str = None,
                 port:int = 8001):

        super(Checker, self).__init__(name=name)
        # self.port = self.choose_port(port)
        self._to_dict = locals()
