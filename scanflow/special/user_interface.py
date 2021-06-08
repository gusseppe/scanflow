import logging

class Node(object):
    """
        Abstract base Node class.

    """
    def __init__(self, name: str):
        self.name = name

class UserInterface(Node):

    def __init__(self,
                 image:str,
                 file:str,
                 # function:str,
                 # model:str,
                 name:str = 'user_interface',
                 # version:int = 1,
                 # stage:str ='Production',
                 port:int = 8010):

        super(UserInterface, self).__init__(name=name)
        self.image = image
        # self.function = function
        # self.model = model
        # self.version = version
        self.port = self.choose_port(port)
        self._to_dict = locals()

    def check_port_in_use(self, port: int):
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0

    def choose_port(self, port: int):
        chosen_port = port
        if self.check_port_in_use(chosen_port):
            logging.info(f"[UserInterface] Port {chosen_port} is in use by UserInterface.")
        else:
            logging.info(f"[UserInterface] Port {chosen_port} is set successfully.")

        return chosen_port

    @property
    def to_dict(self):
        tmp_dict = self._to_dict
        tmp_dict.pop('self', None)
        tmp_dict.pop('__class__', None)
        tmp_dict = {k: v for k, v in tmp_dict.items() if v is not None}
        return tmp_dict
